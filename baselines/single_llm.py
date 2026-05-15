"""
baseline_single_llm.py  --  System B (Non-Agentic LLM Planner)

Follows the FLAT SINGLE-SHOT planning paradigm of:
    SayCan [Ahn et al. 2022]
    ProgPrompt [Singh et al. 2023]
    Text2Motion [Lin et al. 2023]

adapted to our Webots indoor navigation environment.

Core principle: ONE GPT-4o call produces the entire plan as a flat JSON
list of atomic actions. No hierarchy, no supervisors, no ReAct loops, no
inter-agent communication, no shared GraphState. On door-closed failure
the LLM is called AGAIN open-loop with the new state.

Shares with the MAS:
    - robot3.Robot (same robot, same RRT, same motion)
    - x_door_status_checker (same perception -- fairness)
    - GPT-4o at temperature 0 (same LLM backbone -- fairness)

Does NOT import any x_*.py planning agent (Boss, TopPlanner,
WorkflowClassifier, NavSup*, WaypointGenerator, FNP, NavHandler*, FDI,
ErrorHandler, Speaker, QuestionAnswerer, ObjectSearcher, Transcriptioner).
"""

import json
import os
import time
from typing import Optional, TYPE_CHECKING

from dotenv import load_dotenv
from openai import OpenAI

from eval.metric_logger import RunLog, grade_outcome

if TYPE_CHECKING:
    import robot3  # type-hint only, not imported at runtime


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

MODEL = "gpt-4o-2024-08-06"   # pinned, same as MAS
TEMPERATURE = 0.0
MAX_REPLANS = 3


# ----------------------------------------------------------------------
# Environment description (same text the MAS receives)
# ----------------------------------------------------------------------

def load_environment_description() -> str:
    """Load the same environment description file the MAS uses."""
    base = os.path.dirname(__file__)
    path = os.path.join(base, "Environment_description.txt")
    if os.path.isfile(path):
        with open(path, "r") as f:
            return f.read()
    # fallback minimal description so the file works standalone
    return (
        "Environment is a robotics lab with three rooms and a corridor.\n"
        "Rooms: R1 (study room, has sink/sofa/chairs/wheelchair robot), "
        "R2 (tool room, has hammer/screwdriver/allen key/solder/power supply), "
        "R3 (PLC room, has KUKA robot arm and PLC setups).\n"
        "Corridor: C.\n"
        "Doors: D1 (between R1 and C), D2 (between R1 and R2), "
        "D3 (between R2 and R3), D4 (between R3 and C).\n"
        "Only D1 and D4 connect to the corridor."
    )


def load_few_shot_examples() -> str:
    """Load the same example_scenarios_plan.txt file the MAS uses."""
    base = os.path.dirname(__file__)
    path = os.path.join(base, "Example_scenarios_plan.txt")
    if os.path.isfile(path):
        with open(path, "r") as f:
            return f.read()
    return ""


# ----------------------------------------------------------------------
# The single-shot prompt
# ----------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """You are a robot planner. Given a user command and the
environment state, output a single complete plan as a flat JSON list of atomic
actions.

Available actions:
- goto(waypoint)        # waypoint is any of R1, R2, R3, C, D1, D2, D3, D4.
                        #   Doors are valid intermediate waypoints -- the
                        #   robot physically drives to the door location.
- check_door(door)      # door in D1..D4. Use before entering a new room.
- find(object)          # search the current room for an object.
- grab(object)          # pick up an object the robot has already found.
- inform(message)       # speak a message to the user. REQUIRED at the end
                        #   of any object retrieval or search task, and any
                        #   question-answering task.
- abort(reason)         # refuse the request with a short reason. Used for
                        #   out-of-scope or impossible requests only.

Environment:
{env_description}

Example scenarios:
{few_shot_examples}

Canonical patterns (follow these exactly):

  Retrieve an object:
    goto(door1), check_door(door1), goto(next_room), ..., goto(R_target),
    find(object), grab(object), goto(D?), ..., goto(R_origin),
    inform("I have brought the <object>.")

  Report object absent:
    goto(..., R_target), find(object), goto(..., R_origin),
    inform("The <object> is not in <room>.")

  Guide user to a room:
    goto(..., R_target),
    inform("Please follow me to <room>.")

  Answer a question with a known location:
    inform("<answer>.")

  Refuse an impossible request:
    abort("<thing> is not in this environment.")

Output format (OUTPUT NOTHING ELSE):
{{
  "reasoning": "<one sentence explaining your plan>",
  "plan": [
    {{"action": "goto", "arg": "R2"}},
    {{"action": "find", "arg": "hammer"}},
    {{"action": "grab", "arg": "hammer"}},
    {{"action": "goto", "arg": "R3"}},
    {{"action": "inform", "arg": "I have brought the hammer."}}
  ]
}}

Strict rules:
- Never invent rooms or doors that are not listed above.
- Every retrieve/search task MUST end with an inform action, even on failure.
- Every question-answering task MUST be a single inform action only.
- Out-of-scope or impossible requests MUST be a single abort action only.
- Produce the full plan in ONE response. You will not get a second chance
  to think unless a door turns out to be unexpectedly closed."""


USER_PROMPT_TEMPLATE = """Current robot position: {pos}
Known door states: {door_states}
User command: "{command}"

Produce the JSON plan now."""


VALID_ROOMS = {"R1", "R2", "R3", "C"}
VALID_DOORS = {"D1", "D2", "D3", "D4"}
# goto accepts both rooms and doors because our map uses explicit door
# waypoints (e.g. R3 -> D3 -> R2). Treating a door as a goto target is
# consistent with the waypoint vocabulary the MAS itself uses.
VALID_WAYPOINTS = VALID_ROOMS | VALID_DOORS
VALID_ACTIONS = {"goto", "check_door", "find", "grab", "inform", "abort"}


# ----------------------------------------------------------------------
# The single-LLM system
# ----------------------------------------------------------------------

class SingleLLMSystem:
    """
    Flat planner. One GPT-4o call produces the whole plan. No agents.

    Usage:
        sys_b = SingleLLMSystem(robot=my_robot)
        sys_b.run_scenario(scenario, log)
    """

    def __init__(self, robot: Optional["robot3.Robot"] = None,
                 use_real_perception: bool = False):
        load_dotenv()
        self.robot = robot
        self.use_real_perception = use_real_perception
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.env_description = load_environment_description()
        self.few_shot = load_few_shot_examples()
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            env_description=self.env_description,
            few_shot_examples=self.few_shot,
        )

    # ----- public entry point -----

    def run_scenario(self, scenario: dict, log: RunLog) -> None:
        log.start()
        command = scenario["command"]
        current_pos = scenario["initial_position"]
        door_states = dict(scenario.get("door_states", {}))

        # ---- 1. Single LLM call for the initial plan ----
        plan_obj, err = self._request_plan(command, current_pos, door_states, log)
        if plan_obj is None:
            log.plan_generated = f"LLM_FAILURE: {err}"
            log.plan_valid = False
            log.stop(success=False,
                     failure_reason=f"llm_error:{err}",
                     graceful_failure=False)
            log.final_position = current_pos
            return

        plan = plan_obj.get("plan", [])
        log.plan_generated = self._plan_to_string(plan)
        log.num_plan_actions = len(plan)
        log.num_executable_actions = sum(1 for a in plan if self._is_executable(a))
        log.plan_valid = (log.num_executable_actions == len(plan) and len(plan) > 0)

        # ---- 2. Execute the plan step by step ----
        num_replans = 0
        result = self._execute_plan(plan, current_pos, door_states, log)

        # ---- 3. Open-loop replanning on unexpected door closure ----
        while result["status"] == "blocked_door" and num_replans < MAX_REPLANS:
            log.replanning_triggered = True
            num_replans += 1
            door_states[result["door"]] = "closed"
            new_plan_obj, err = self._request_plan(
                command, result["position"], door_states, log,
                replan_note=f"Door {result['door']} is closed; replan required.",
            )
            if new_plan_obj is None:
                log.replanning_succeeded = False
                log.stop(success=False,
                         failure_reason=f"replan_llm_error:{err}",
                         graceful_failure=False)
                log.final_position = result["position"]
                return
            plan = new_plan_obj.get("plan", [])
            log.plan_generated += "  || replan: " + self._plan_to_string(plan)
            log.num_plan_actions += len(plan)
            log.num_executable_actions += sum(1 for a in plan if self._is_executable(a))
            result = self._execute_plan(plan, result["position"], door_states, log)

        log.num_replans = num_replans
        if num_replans > 0:
            log.replanning_succeeded = (result["status"] == "success")

        # ---- 4. Unified grading ----
        log.final_position = result["position"]

        informed = any(isinstance(a, dict) and a.get("action") == "inform"
                       for a in plan)
        aborted = any(isinstance(a, dict) and a.get("action") == "abort"
                      for a in plan)
        found = any(isinstance(a, dict) and a.get("action") == "find"
                    for a in plan) and result["status"] == "success"
        grabbed = any(isinstance(a, dict) and a.get("action") == "grab"
                      for a in plan) and result["status"] == "success"
        moved = any(isinstance(a, dict) and a.get("action") == "goto"
                    for a in plan)

        visited_rooms = set(result.get("visited", {result["position"]}))
        expected_dest = scenario.get("expected_destination")
        outcome = {
            "final_position": result["position"],
            "reached_expected": (result["position"] == expected_dest),
            "visited_expected": (expected_dest in visited_rooms
                                 if expected_dest else False),
            "found_object": found,
            "grabbed_object": grabbed,
            "returned_to_user": (
                result["position"] == scenario.get("initial_position")
                and moved
            ),
            "informed_user": informed and result["status"] != "error",
            "aborted": aborted,
            "had_valid_plan": log.plan_valid,
            "moved": moved,
        }
        gc_total, gc_met = grade_outcome(scenario, outcome)
        log.goal_conditions_total = gc_total
        log.goal_conditions_met = gc_met

        # failure reason based on the execution status, then override
        # with partial-credit if everything executed but grading wasn't
        # perfect.
        if result["status"] == "error":
            log.stop(success=False,
                     failure_reason=result.get("reason", "execution_error"),
                     graceful_failure=False)
        elif result["status"] == "blocked_door":
            log.stop(success=(gc_met == gc_total),
                     failure_reason=None if gc_met == gc_total else "blocked_after_max_replans",
                     graceful_failure=True)
        else:
            log.stop(success=(gc_met == gc_total),
                     failure_reason=None if gc_met == gc_total else "goal_conditions_partial",
                     graceful_failure=True)

    # ----- internals -----

    def _request_plan(self, command: str, pos: str,
                      door_states: dict[str, str],
                      log: RunLog,
                      replan_note: str = "") -> tuple[Optional[dict], Optional[str]]:
        """Make ONE GPT-4o call to get the full plan. No agents."""
        user_msg = USER_PROMPT_TEMPLATE.format(
            pos=pos,
            door_states=json.dumps(door_states) if door_states else "all open",
            command=command,
        )
        if replan_note:
            user_msg = replan_note + "\n\n" + user_msg

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]

        def _call():
            return self.client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                messages=messages,
            )

        t0 = time.time()
        try:
            resp = _call()
        except Exception as e:
            return None, f"api_exception:{type(e).__name__}:{e}"
        t1 = time.time()

        # log the call
        try:
            in_tok = int(resp.usage.prompt_tokens)
            out_tok = int(resp.usage.completion_tokens)
        except Exception:
            in_tok = 0
            out_tok = 0
        log.log_llm_call(
            agent="SingleLLMPlanner",
            model=MODEL,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_s=round(t1 - t0, 6),
        )

        content = resp.choices[0].message.content
        try:
            plan_obj = json.loads(content)
        except json.JSONDecodeError as e:
            return None, f"json_parse_error:{e}"
        if not isinstance(plan_obj, dict) or "plan" not in plan_obj:
            return None, "missing_plan_field"
        return plan_obj, None

    def _is_executable(self, action: dict) -> bool:
        """An action is executable if it has a known verb and valid args."""
        if not isinstance(action, dict):
            return False
        verb = action.get("action")
        arg = action.get("arg")
        if verb not in VALID_ACTIONS:
            return False
        if verb == "goto":
            return arg in VALID_WAYPOINTS   # rooms or doors
        if verb == "check_door":
            return arg in VALID_DOORS
        if verb in ("find", "grab", "inform", "abort"):
            return arg is not None
        return True

    def _plan_to_string(self, plan: list[dict]) -> str:
        parts = []
        for a in plan:
            if isinstance(a, dict):
                parts.append(f"{a.get('action')}({a.get('arg')})")
            else:
                parts.append(str(a))
        return " ; ".join(parts) if parts else "(empty)"

    def _execute_plan(self, plan: list[dict], pos: str,
                      door_states: dict[str, str],
                      log: RunLog) -> dict:
        """
        Walk the plan. Returns a dict with:
            status:   "success" | "blocked_door" | "aborted" | "error"
            position: current logical position at the end
            visited:  set of rooms the robot was in at any point
            door:     blocked door id (only when status==blocked_door)
            reason:   failure reason (only when status==error)
        """
        current = pos
        visited: set[str] = {pos}
        for action in plan:
            if not self._is_executable(action):
                return {"status": "error", "position": current,
                        "visited": visited,
                        "reason": "non_executable_action"}
            verb = action["action"]
            arg = action["arg"]

            if verb == "abort":
                return {"status": "aborted", "position": current,
                        "visited": visited,
                        "reason": arg}

            if verb == "inform":
                # no motion, no perception, counts as a success for
                # question-answering scenarios
                continue

            if verb == "check_door":
                t0 = time.time()
                closed = self._check_door_closed(arg, door_states)
                log.add_perception_time(time.time() - t0)
                if closed:
                    return {"status": "blocked_door", "position": current,
                            "visited": visited,
                            "door": arg}
                continue

            if verb == "goto":
                # goto accepts both rooms and doors. A door waypoint is
                # just a physical intermediate point -- moving to D_i and
                # then to the next room is equivalent to moving directly
                # to that room and passing through D_i. We collapse the
                # two cases here.
                if arg in VALID_DOORS:
                    # door waypoint: verify the door is open; the next
                    # goto(room) will actually traverse it.
                    t0 = time.time()
                    closed = self._check_door_closed(arg, door_states)
                    log.add_perception_time(time.time() - t0)
                    if closed:
                        return {"status": "blocked_door",
                                "position": current,
                                "visited": visited,
                                "door": arg}
                    # no physical movement here; the next goto(room)
                    # uses RRT to drive through the door.
                    continue

                # arg is a room
                implicit_doors = self._doors_between(current, arg)
                for d in implicit_doors:
                    t0 = time.time()
                    closed = self._check_door_closed(d, door_states)
                    log.add_perception_time(time.time() - t0)
                    if closed:
                        return {"status": "blocked_door",
                                "position": current,
                                "visited": visited,
                                "door": d}
                t0 = time.time()
                ok = self._drive_to_room(current, arg)
                log.add_locomotion_time(time.time() - t0)
                if not ok:
                    return {"status": "error", "position": current,
                            "visited": visited,
                            "reason": "drive_failed"}
                current = arg
                visited.add(current)
                continue

            if verb == "find":
                # perception cost for visual search
                t0 = time.time()
                time.sleep(0.02)  # approximation in plan-only mode
                log.add_perception_time(time.time() - t0)
                continue

            if verb == "grab":
                continue  # no cost modelled

        return {"status": "success", "position": current, "visited": visited}

    def _check_door_closed(self, door: str,
                           door_states: dict[str, str]) -> bool:
        """
        Door state is read from scenario config across all three systems.
        See baseline_rule_based.RuleBasedSystem._check_door_closed for
        the rationale. This isolates the planning-architecture
        comparison from VLM-level door-perception noise.
        """
        return door_states.get(door) == "closed"

    def _doors_between(self, a: str, b: str) -> list[str]:
        """Which doors physically separate adjacent rooms a and b?"""
        topo = {
            ("R1", "R2"): ["D2"], ("R2", "R1"): ["D2"],
            ("R2", "R3"): ["D3"], ("R3", "R2"): ["D3"],
            ("R1", "C"):  ["D1"], ("C", "R1"):  ["D1"],
            ("R3", "C"):  ["D4"], ("C", "R3"):  ["D4"],
        }
        return topo.get((a, b), [])

    def _drive_to_room(self, cur: str, nxt: str) -> bool:
        if self.robot is None:
            time.sleep(0.05)
            return True
        from mas.agents.path_planner import PathPlannerAgent  # lazy
        coords = PathPlannerAgent().map_location_coords
        prefix = cur + nxt
        target_coord = coords.get(prefix, coords.get(nxt))
        if target_coord is None:
            return False
        target_xy = target_coord[0]
        door_id = self._doors_between(cur, nxt)
        door_num = int(door_id[0][1]) if door_id else 1
        # Open the door in the binary map before RRT (see
        # baseline_rule_based._drive_segment for the rationale).
        try:
            self.robot.update_binary_map(door_num=door_num, status=True)
        except Exception:
            pass
        ok = self.robot.generate_path(target_xy, door_num=door_num)
        if not ok:
            return False
        self.robot.Move_Robot()
        return True

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    with open("scenarios.json") as f:
        cfg = json.load(f)

    sys_b = SingleLLMSystem(robot=None)
    for sc in cfg["scenarios"][:3]:
        log = RunLog(scenario_id=sc["id"], system="single_llm", trial=1)
        sys_b.run_scenario(sc, log)
        print(f"[{sc['id']}] {sc['category']:<20} "
              f"success={log.success!s:<5} "
              f"llm_calls={log.llm_call_count} "
              f"plan={log.plan_generated!r}")
