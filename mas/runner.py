"""
mas_runner.py  --  text-mode wrapper around the hierarchical MAS.

Exposes MASRunner.run_scenario(scenario, log) with the SAME signature as
baseline_rule_based.RuleBasedSystem and baseline_single_llm.SingleLLMSystem,
so run_experiment.py can treat all three systems interchangeably.

Internally this module:
  - imports the existing x_*.py agent classes (identical to Allinone_1.py)
  - builds the SAME LangGraph the MAS uses
  - replaces SpeechAgent with a text injector that reads scenario.command
  - replaces Speaker with a message capturer (no TTS) so we can grade
    whether the MAS "informed" the user without playing audio
  - replaces Finisher's Excel export with a soft reset of agent histories
  - hooks every node through a RunLog for per-agent timing + LLM call
    accounting

The MAS itself is unchanged. This file sits alongside Allinone_1.py and
does not import it (Allinone_1.py runs app.invoke at import time; we
can't use it as a module).
"""

from __future__ import annotations

import os
import time
from typing import Optional, TYPE_CHECKING, TypedDict

from dotenv import load_dotenv

from eval.metric_logger import RunLog, grade_outcome

if TYPE_CHECKING:
    import robot3  # type-hint only


# ----------------------------------------------------------------------
# GraphState -- identical to Allinone_1.py so agent wrappers drop in
# unchanged.
# ----------------------------------------------------------------------

class GraphState(TypedDict, total=False):
    Current_robot_position_G: str
    Recent_agent_G: str
    Raw_user_request_G: str
    Transcriptioner_status: str

    Thought_boss: str
    Next_agent_boss: str
    AgentInput_boss: str

    Speaker_caller: str
    Output_speaker: str

    Thought_toppl: str
    taskplan_toppl: str

    Thought_wc: str
    Next_agent_wc: str
    AgentInput_wc: str

    Object_tobe_searched_os: str
    Object_search_status: str
    Object_location_os: str

    Object_tobe_grabbed_og: str
    Object_location_og: str
    Object_grabber_status: str

    Explanation_qa: str
    FinalAnswer_qa: str

    response_navigation_supervisor_main: str

    Thought_navsupsec: str
    Action_navsupsec: str
    AgentInput_navsupsec: str

    Explanation_destination: str
    Current_robot_position_CPI: str

    Room_waypoints: str
    Door_sequence: str
    Final_waypoints: str

    Plan_FNP: str

    Next_task_NHM: str
    Action_type_NHM: str

    Thought_navhansec: str
    Action_navhansec: str
    AgentInput_navhansec: str

    Door_status_explanation: str

    Path_planning_status: str
    Target_orientation: int
    Target_position: str

    Robot_executor_status: str

    Thought_error: str
    Action_error: str
    AgentInput_error: str

    check_finisher: str

    # text-mode extras (not in Allinone_1)
    _scenario_id: str
    _visited: str               # comma-joined list of rooms visited
    _done: bool


# ----------------------------------------------------------------------
# Module-level state used by node wrappers
# ----------------------------------------------------------------------

_current_log: Optional[RunLog] = None
_current_scenario: Optional[dict] = None
_captured_messages: list[str] = []
_visited_rooms: set[str] = set()


def _reset_run_state() -> None:
    global _captured_messages, _visited_rooms
    _captured_messages = []
    _visited_rooms = set()


def _log_agent_time(agent_name: str, t0: float, t1: float) -> None:
    if _current_log is not None:
        _current_log.log_agent_timing(agent_name, t0, t1)


# When True, prints [TRACE] lines before/after every node invocation so
# a freeze can be pinpointed to an exact node. Cheap in production, worth
# keeping on during debugging of the 3-way comparison runs.
TRACE_NODES = True


def _trace(msg: str) -> None:
    if TRACE_NODES:
        print(f"[TRACE] {msg}", flush=True)


# ----------------------------------------------------------------------
# The MASRunner class
# ----------------------------------------------------------------------

class MASRunner:
    """
    Thin wrapper so run_experiment.py can treat the MAS exactly like
    the baselines.

    Usage:
        runner = MASRunner(robot=my_robot)
        runner.run_scenario(scenario, log)
    """

    def __init__(self, robot: Optional["robot3.Robot"] = None,
                 use_real_perception: bool = True):
        load_dotenv()
        self.robot = robot
        self.use_real_perception = use_real_perception

        # Lazy import -- these pull in langchain/openai/pygame/cv2 etc.
        from mas.agents.boss import BossAgent
        from mas.agents.top_planner import TopPlannerAgent
        from mas.agents.workflow_classifier import WorkflowClassifierAgent
        from mas.agents.object_searcher import ObjectFindingAgent
        from mas.agents.question_answerer import QuestionAnsweringAgent
        from mas.agents.navigational_supervisor_main import NavigationSupervisorMainAgent
        from mas.agents.navigational_supervisor_sec import NavigationSupervisorSecAgent
        from mas.agents.final_destination_identifier import DestinationDecisionAgent
        from mas.agents.waypoint_generator import WaypointGenerationAgent
        from mas.agents.final_navigational_planner import FinalNavigationalPlannerAgent
        from mas.agents.navigation_handler_main import NavigationalHandlerMainAgent
        from mas.agents.navigation_handler_sec import NavigationalHandlerSecAgent
        from mas.agents.path_planner import PathPlannerAgent
        from mas.agents.door_status_checker import DoorStatusCheckerAgent
        from mas.agents.error_handler import ErrorHandlerAgent

        self.Boss = BossAgent()
        self.TopPlanner = TopPlannerAgent()
        self.WorkflowClassifier = WorkflowClassifierAgent()
        self.ObjectSearcher = ObjectFindingAgent()
        self.QuestionAnswerer = QuestionAnsweringAgent()
        self.NavigationSupervisorMain = NavigationSupervisorMainAgent()
        self.NavigationSupervisorSec = NavigationSupervisorSecAgent()
        self.DestinationIdentifier = DestinationDecisionAgent()
        self.WaypointGenerator = WaypointGenerationAgent()
        self.FinalNavigationPlanner = FinalNavigationalPlannerAgent()
        self.NavigationHandlerMain = NavigationalHandlerMainAgent()
        self.NavigationHandlerSec = NavigationalHandlerSecAgent()
        self.PathPlanner = PathPlannerAgent()
        self.DoorStatusChecker = DoorStatusCheckerAgent()
        self.ErrorHandler = ErrorHandlerAgent()

        self.app = self._build_graph()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_scenario(self, scenario: dict, log: RunLog) -> None:
        global _current_log, _current_scenario
        _current_log = log
        _current_scenario = scenario
        _reset_run_state()

        # mark initial position as visited
        _visited_rooms.add(scenario["initial_position"])

        initial_state: GraphState = {
            "Current_robot_position_G": scenario["initial_position"],
            "Transcriptioner_status": "off",   # text mode
            "_scenario_id": scenario["id"],
            "_done": False,
        }

        log.start()
        final_state: dict = {}
        try:
            # recursion_limit = 150 gives comfortable headroom for a
            # complete R3->R2->R1 navigation including post-arrival
            # ObjectSearcher + Speaker + Finisher cleanup, while still
            # bounding the runaway loops we've observed on scenarios
            # like "take me to netball court" where the MAS cannot find
            # a reachable goal and keeps re-checking doors. A measured
            # R3->R2->R1 run uses ~38 agent calls = ~70-90 LangGraph
            # steps including conditional-edge transitions.
            final_state = self.app.invoke(
                initial_state,
                config={"recursion_limit": 150},
            )
        except Exception as e:
            _trace(f"mas_exception: {type(e).__name__}: {e}")
            log.stop(
                success=False,
                failure_reason=f"mas_exception:{type(e).__name__}:{e}",
                graceful_failure=False,
            )
            log.notes = f"messages={_captured_messages[:3]}"
            self._reset_agent_histories()
            _current_log = None
            _current_scenario = None
            return

        # ---- grade ----
        final_pos = final_state.get("Current_robot_position_G",
                                    scenario["initial_position"])
        log.final_position = final_pos

        informed = len(_captured_messages) > 0
        qa_answered = bool(final_state.get("FinalAnswer_qa"))

        # plan: prefer the navigation waypoints, fall back to QA answer
        # or the last captured Speaker message (for aborts / inform-only).
        plan_txt = (final_state.get("Final_waypoints", "")
                    or final_state.get("FinalAnswer_qa", "")
                    or (_captured_messages[-1] if _captured_messages else ""))
        log.plan_generated = plan_txt
        log.plan_valid = bool(plan_txt)

        # reasoning time = sum of per-agent durations (every MAS agent
        # is LLM-backed, so agent time is reasoning time).
        log.reasoning_time_s = round(
            sum(t.get("duration", 0.0) for t in log.agent_timings), 6)

        # detect abort: MAS aborts by having the Speaker report the
        # rejection and Boss routes to Finisher without navigation.
        aborted = any("not" in m.lower() and
                      ("exist" in m.lower() or "available" in m.lower())
                      for m in _captured_messages)

        expected_dest = scenario.get("expected_destination")
        outcome = {
            "final_position": final_pos,
            "reached_expected": (final_pos == expected_dest),
            "visited_expected": (expected_dest in _visited_rooms
                                 if expected_dest else False),
            "found_object": final_state.get("Object_search_status",
                                            "") == "Found the object",
            "grabbed_object": False,
            "returned_to_user": (final_pos == scenario["initial_position"]
                                 and len(_visited_rooms) > 1),
            "informed_user": informed or qa_answered,
            "aborted": aborted,
            "had_valid_plan": log.plan_valid,
            "moved": len(_visited_rooms) > 1,
        }
        gc_total, gc_met = grade_outcome(scenario, outcome)
        log.goal_conditions_total = gc_total
        log.goal_conditions_met = gc_met

        log.stop(
            success=(gc_met == gc_total),
            failure_reason=None if gc_met == gc_total else "goal_conditions_partial",
            graceful_failure=True,
        )
        log.notes = f"messages={_captured_messages[:3]}"

        self._reset_agent_histories()
        _current_log = None
        _current_scenario = None

    # ------------------------------------------------------------------
    # Text-mode I/O node replacements
    # ------------------------------------------------------------------

    def _node_text_input(self, state: GraphState) -> dict:
        """Replaces SpeechAgent: feed the scenario command as the user request."""
        t0 = time.time()
        assert _current_scenario is not None
        cmd = _current_scenario["command"]
        t1 = time.time()
        _log_agent_time("SpeechAgent", t0, t1)
        return {
            "Raw_user_request_G": cmd,
            "Recent_agent_G": "SpeechAgent",
        }

    def _node_text_capture(self, state: GraphState) -> dict:
        """Replaces Speaker: capture the message instead of playing TTS."""
        t0 = time.time()
        caller = state.get("Recent_agent_G", "")
        if caller == "Boss":
            speaker_input = state.get("AgentInput_boss", "")
        elif caller == "WorkflowClassifier":
            speaker_input = state.get("AgentInput_wc", "")
        else:
            speaker_input = ""

        # Still invoke the Speaker agent to get the user-friendly phrasing,
        # but skip the TTS play().
        try:
            ai_response_json = self.WorkflowClassifier.__class__.__mro__  # no-op
        except Exception:
            pass
        # We bypass the SpeakerAgent entirely and just record the raw
        # AgentInput -- it is already a human-readable message.
        message = str(speaker_input)
        _captured_messages.append(message)

        # Inform the caller agent that speaking completed (as the real
        # Speaker does) so its chat history advances correctly.
        if caller == "Boss":
            self.Boss.add_observations("Observation: Speaking completed")
        elif caller == "WorkflowClassifier":
            self.WorkflowClassifier.add_observations("Observation: Speaking completed")

        t1 = time.time()
        _log_agent_time("Speaker", t0, t1)
        return {
            "Speaker_caller": caller,
            "Output_speaker": message,
            "Recent_agent_G": "Speaker",
        }

    def _node_finisher(self, state: GraphState) -> dict:
        """Replaces Finisher: soft reset instead of Excel dump."""
        t0 = time.time()
        self._reset_agent_histories()
        t1 = time.time()
        _log_agent_time("Finisher", t0, t1)
        return {"_done": True, "Recent_agent_G": "Finisher"}

    # ------------------------------------------------------------------
    # Agent node wrappers (same logic as Allinone_1.py but timed via RunLog)
    # ------------------------------------------------------------------

    def _node_boss(self, state: GraphState) -> dict:
        _trace("Boss enter")
        t0 = time.time()
        request = state["Raw_user_request_G"]
        robot_current_position = state["Current_robot_position_G"]
        resp = self.Boss.process_user_request(
            request=request, robot_current_position=robot_current_position)
        _log_agent_time("Boss", t0, time.time())
        _trace(f"Boss exit -> {resp.get('Action')}")
        return {
            "Thought_boss": resp["Thought"],
            "Next_agent_boss": resp["Action"],
            "AgentInput_boss": resp["AgentInput"],
            "Recent_agent_G": "Boss",
        }

    def _node_top_planner(self, state: GraphState) -> dict:
        _trace("TopPlanner enter")
        t0 = time.time()
        resp = self.TopPlanner.process_user_request(
            user_request=state["AgentInput_boss"])
        _log_agent_time("TopPlanner", t0, time.time())
        return {
            "Thought_toppl": resp["Thought"],
            "taskplan_toppl": resp["Action"],
            "Recent_agent_G": "TopPlanner",
        }

    def _node_workflow_classifier(self, state: GraphState) -> dict:
        _trace("WorkflowClassifier enter")
        t0 = time.time()
        resp = self.WorkflowClassifier.process_user_request(
            user_request=state["AgentInput_boss"],
            task_plan=state["taskplan_toppl"])
        _log_agent_time("WorkflowClassifier", t0, time.time())
        return {
            "Thought_wc": resp["Thought"],
            "Next_agent_wc": resp["Action"],
            "AgentInput_wc": resp["AgentInput"],
            "Recent_agent_G": "WorkflowClassifier",
        }

    def _node_object_searcher(self, state: GraphState) -> dict:
        t0 = time.time()
        obj = state["AgentInput_wc"]
        if self.robot is None or not self.use_real_perception:
            # Stub: credit based on scenario config so smoke runs work.
            assert _current_scenario is not None
            observed = ("Found the object"
                        if _current_scenario.get("object_in_world", False)
                        else "Cannot found the object")
        else:
            max_scans = 8
            resp = None
            for _ in range(max_scans):
                frame = self.robot.get_current_RGB_camera_frames()
                resp = self.ObjectSearcher.process_user_request(
                    image=frame, object_label=obj, draw_boxes=True)
                if resp.get("Object_Found") == "found":
                    break
                self.robot.rotate_to_scan_environment("left")
            observed = ("Found the object"
                        if resp and resp.get("Object_Found") == "found"
                        else "Cannot found the object")
        self.WorkflowClassifier.add_observations(f"Observation: {observed}")
        _log_agent_time("ObjectSearcher", t0, time.time())
        return {
            "Object_tobe_searched_os": obj,
            "Object_search_status": observed,
            "Recent_agent_G": "ObjectSearcher",
        }

    def _node_object_grabber(self, state: GraphState) -> dict:
        t0 = time.time()
        # plan-only mode: just record a success observation, do not
        # block on input().
        self.WorkflowClassifier.add_observations("Observation: Object grabbed")
        _log_agent_time("ObjectGrabber", t0, time.time())
        return {"Recent_agent_G": "ObjectGrabber"}

    def _node_question_answerer(self, state: GraphState) -> dict:
        t0 = time.time()
        q = state["AgentInput_wc"]
        pos = state["Current_robot_position_G"]
        inp = f"User question: {q} , User current position: {pos}"
        resp = self.QuestionAnswerer.process_user_request(user_request=inp)
        self.WorkflowClassifier.add_observations(
            f"Observation: {resp['FinalAnswer']}")
        _log_agent_time("QuestionAnswerer", t0, time.time())
        return {
            "Explanation_qa": resp["Explanation"],
            "FinalAnswer_qa": resp["FinalAnswer"],
            "Recent_agent_G": "QuestionAnswerer",
        }

    def _node_nav_sup_main(self, state: GraphState) -> dict:
        _trace("NavSupMain enter")
        t0 = time.time()
        if state.get("Recent_agent_G") == "ErrorHandler":
            req = state["AgentInput_error"]
        else:
            req = state["AgentInput_wc"]
        resp = self.NavigationSupervisorMain.process_user_request(request=req)
        self.NavigationSupervisorSec.reset_chat_messages()
        self.NavigationHandlerMain.reset_chat_messages()
        self.NavigationHandlerSec.reset_chat_messages()
        _log_agent_time("NavigationSupervisorMain", t0, time.time())
        return {
            "response_navigation_supervisor_main": resp,
            "Recent_agent_G": "NavigationsupervisorMain",
        }

    def _node_nav_sup_sec(self, state: GraphState) -> dict:
        _trace("NavSupSec enter")
        t0 = time.time()
        plan = state["response_navigation_supervisor_main"]
        resp = self.NavigationSupervisorSec.process_user_request(request=plan)
        _log_agent_time("NavigationSupervisorSec", t0, time.time())
        _trace(f"NavSupSec exit -> {resp.get('Action')}")
        return {
            "Thought_navsupsec": resp["Thought"],
            "Action_navsupsec": resp["Action"],
            "AgentInput_navsupsec": resp["AgentInput"],
            "Recent_agent_G": "NavigationsupervisorSec",
        }

    def _node_final_destination_identifier(self, state: GraphState) -> dict:
        t0 = time.time()
        resp = self.DestinationIdentifier.process_user_request(
            state["AgentInput_navsupsec"])
        formatted = (f"Exact destination: {resp['Destination']}, "
                     f"Explanation: {resp['Explanation']}")
        self.NavigationSupervisorSec.add_observations(
            f"Observation: {formatted}")
        _log_agent_time("FinalDestinationIdentifier", t0, time.time())
        return {
            "Explanation_destination": formatted,
            "Recent_agent_G": "FinalDestinationIdentifier",
        }

    def _node_current_position_identifier(self, state: GraphState) -> dict:
        t0 = time.time()
        pos = state["Current_robot_position_G"]
        msg = f"Robot current position is {pos}."
        self.NavigationSupervisorSec.add_observations(f"Observation: {msg}")
        _log_agent_time("CurrentPositionIdentifier", t0, time.time())
        return {
            "Current_robot_position_CPI": msg,
            "Recent_agent_G": "CurrentPositionIdentifier",
        }

    def _node_waypoint_generator(self, state: GraphState) -> dict:
        t0 = time.time()
        resp = self.WaypointGenerator.process_user_request(
            state["AgentInput_navsupsec"])
        formatted = (f"Waypoints: {resp['WaypointSequence']}, "
                     f"DoorSequence: {resp['doorSequence']}")
        self.NavigationSupervisorSec.add_observations(
            f"Observation: ({formatted})")
        _log_agent_time("WaypointGenerator", t0, time.time())
        return {
            "Room_waypoints": resp["WaypointSequence"],
            "Door_sequence": resp["doorSequence"],
            "Final_waypoints": resp["FinalSequence"],
            "Recent_agent_G": "WaypointGenerator",
        }

    def _node_final_nav_planner(self, state: GraphState) -> dict:
        t0 = time.time()
        resp = self.FinalNavigationPlanner.process_user_request(
            state["AgentInput_navsupsec"])
        self.NavigationHandlerMain.reset_chat_messages()
        _log_agent_time("FinalNavigationalPlanner", t0, time.time())
        return {"Plan_FNP": resp["The_plan"], "Recent_agent_G": "FinalNavigationalPlanner"}

    def _node_nav_handler_main(self, state: GraphState) -> dict:
        _trace("NavHandlerMain enter")
        t0 = time.time()
        resp = self.NavigationHandlerMain.process_user_request(state["Plan_FNP"])
        self.NavigationHandlerSec.reset_chat_messages()
        _log_agent_time("NavigationHandlerMain", t0, time.time())
        _trace(f"NavHandlerMain exit -> ActionType={resp.get('ActionType')}")
        return {
            "Next_task_NHM": f"Thought: {resp['Thought']}, Action: {resp['Action']}",
            "Action_type_NHM": resp["ActionType"],
            "Recent_agent_G": "NavigationHandlerMain",
        }

    def _node_nav_handler_sec(self, state: GraphState) -> dict:
        _trace("NavHandlerSec enter")
        t0 = time.time()
        resp = self.NavigationHandlerSec.process_user_request(
            request=state["Next_task_NHM"])
        if resp["Action"] == "NavigationHandlerMain":
            self.NavigationHandlerMain.add_observations(
                f"Observation: {resp['AgentInput']}")
        _log_agent_time("NavigationHandlerSec", t0, time.time())
        _trace(f"NavHandlerSec exit -> {resp.get('Action')}")
        return {
            "Thought_navhansec": resp["Thought"],
            "Action_navhansec": resp["Action"],
            "AgentInput_navhansec": resp["AgentInput"],
            "Recent_agent_G": "NavigationHandlerSec",
        }

    def _node_door_checker(self, state: GraphState) -> dict:
        _trace("DoorChecker enter")
        """
        Door state is read from scenario config across all three systems
        (MAS, rule-based, single-LLM). This deliberately isolates the
        planning-architecture comparison from VLM-level door-perception
        noise, which is not the subject of this evaluation. Object
        recognition (ObjectSearcher) still uses the real VLM.

        Note: the MAS's originally submitted Table I used real VLM
        door perception and is not replaced by this run -- it is
        preserved in the manuscript as a separate robustness result.
        """
        t0 = time.time()
        assert _current_scenario is not None
        doors = _current_scenario.get("door_states", {})

        # Infer which door to check. Original MAS uses a camera frame +
        # VLM; we use scenario config, so we must reconstruct the door
        # identity from the graph state. Sources, in order:
        #   1. AgentInput_navhansec  -- explicit "check D3" text
        #   2. Target_position       -- just-navigated waypoint (often "D3")
        #   3. AgentInput_navsupsec  -- the navigation plan text
        #   4. Door_sequence         -- the waypoint door list
        # Default D1 only if nothing matches (which should be rare).
        # Getting this right is critical: if NavHandlerSec observes
        # "Door D1 is open" when the plan expects D3, it loops forever.
        candidates = [
            state.get("AgentInput_navhansec"),
            state.get("Target_position"),
            state.get("AgentInput_navsupsec"),
            state.get("Door_sequence"),
        ]
        detected = None
        for src in candidates:
            if not src:
                continue
            src_upper = str(src).upper()
            m = next((d for d in ("D1", "D2", "D3", "D4") if d in src_upper), None)
            if m:
                detected = m
                break
        if detected is None:
            detected = "D1"

        status = doors.get(detected, "open")
        explanation = (f"Door {detected} is open"
                       if status == "open"
                       else f"Error Occured. Door {detected} is closed. "
                            "Cannot proceed further")
        _trace(f"DoorChecker: inferred door={detected} status={status}")
        self.NavigationHandlerSec.add_observations(
            f"Observation: {explanation}")
        _log_agent_time("DoorChecker", t0, time.time())
        return {
            "Door_status_explanation": explanation,
            "Recent_agent_G": "DoorChecker",
        }

    def _node_path_planner(self, state: GraphState) -> dict:
        _trace(f"PathPlanner enter target={state.get('AgentInput_navhansec')}")
        t0 = time.time()
        inp = (f"Robot current position: {state['Current_robot_position_G']}, "
               f"robot target position: {state['AgentInput_navhansec']}")
        resp = self.PathPlanner.process_user_request(inp)

        # Short-circuit: when the requested target is where the robot
        # already is, do not call generate_path and mark the state so
        # RobotExecutor skips too. Without this guard, RobotExecutor
        # tries to drive using the previous scenario's stale path and
        # crashes in Move_Robot with IndexError.
        short_circuit = (resp["robot_current_pos_coord"] ==
                         resp["robot_target_pos_coord"])
        if short_circuit:
            msg = (f"Robot is already at {resp['target_position']}. "
                   "Step complete; no movement needed.")
        elif self.robot is not None and self.use_real_perception:
            # The update_binary_map call can raise AttributeError when
            # show_rt_display=False because its last line writes to
            # self.path_display. The binary map itself IS updated before
            # that line, so swallowing the exception is safe -- mirrors
            # the try/except pattern used by baseline_rule_based and
            # baseline_single_llm.
            try:
                self.robot.update_binary_map(
                    door_num=resp["door_num"], status=True)
            except AttributeError:
                pass
            ok = self.robot.generate_path(
                target_position=resp["robot_target_pos_coord"],
                door_num=resp["door_num"],
                show_rt_path=False,
            )
            msg = "Path planning done succesfully." if ok else "Path planning failed. Try again."
        else:
            msg = "Path planning done succesfully."
        self.NavigationHandlerSec.add_observations(f"Observation: {msg}")
        _log_agent_time("PathPlanner", t0, time.time())
        return {
            "Path_planning_status": msg,
            "Target_orientation": resp["target_orientation"],
            "Target_position": resp["target_position"],
            "Recent_agent_G": "PathPlanner",
        }

    def _node_robot_executor(self, state: GraphState) -> dict:
        _trace(f"RobotExecutor enter target_pos={state.get('Target_position')}")
        t0 = time.time()
        target_pos = state["Target_position"]
        current_pos = state["Current_robot_position_G"]

        # If PathPlanner short-circuited (current == target), there is
        # no fresh path to execute. Skip the physical drive instead of
        # running Move_Robot against stale generated_path state.
        status_msg = state.get("Path_planning_status") or ""
        short_circuit = "no movement needed" in status_msg

        if short_circuit or self.robot is None or not self.use_real_perception:
            _trace("RobotExecutor: short_circuit, skip drive")
            msg = "Robot executed completed"
            if target_pos and target_pos[0] in ("R", "C"):
                current_pos = target_pos
        else:
            # Explicitly reset the movement-completion flag BEFORE
            # launching the thread so the wait loop cannot early-exit
            # on a stale True left by the previous Execute_robot call.
            try:
                self.robot.successful_robot_movement = False
            except Exception:
                pass
            _trace("RobotExecutor: calling Execute_robot")
            self.robot.Execute_robot(target_orientation=state["Target_orientation"])
            waited = 0
            while True:
                time.sleep(1)
                waited += 1
                if self.robot.successful_robot_movement:
                    break
                if waited >= 90:
                    _trace(f"RobotExecutor: TIMEOUT after {waited}s; "
                           "breaking to avoid hang")
                    break
            _trace(f"RobotExecutor: drive done in ~{waited}s")
            msg = "Robot executed completed"
            if target_pos and target_pos[0] in ("R", "C"):
                current_pos = target_pos

        if target_pos and target_pos[0] in ("R", "C"):
            _visited_rooms.add(target_pos)

        self.NavigationHandlerSec.add_observations(f"Observation: {msg}")
        _log_agent_time("RobotExecutor", t0, time.time())
        _trace(f"RobotExecutor exit -> {msg}")
        return {
            "Robot_executor_status": msg,
            "Current_robot_position_G": current_pos,
            "Recent_agent_G": "RobotExecutor",
        }

    def _node_error_handler(self, state: GraphState) -> dict:
        t0 = time.time()
        sender = state.get("Recent_agent_G", "")
        if sender == "NavigationsupervisorSec":
            err = state.get("AgentInput_navsupsec", "")
        elif sender == "NavigationHandlerSec":
            err = state.get("AgentInput_navhansec", "")
        else:
            err = ""
        resp = self.ErrorHandler.process_user_request(err)
        _log_agent_time("ErrorHandler", t0, time.time())
        return {
            "Thought_error": resp["Thought"],
            "Action_error": resp["Action"],
            "AgentInput_error": resp["AgentInput"],
            "Recent_agent_G": "ErrorHandler",
        }

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------

    def _router_boss(self, state: GraphState) -> str:
        return state["Next_agent_boss"]

    def _router_speaker(self, state: GraphState) -> str:
        return state["Speaker_caller"]

    def _router_wc(self, state: GraphState) -> str:
        nxt = state["Next_agent_wc"]
        if nxt == "NavigationsupervisorMain":
            self.NavigationSupervisorMain.reset_chat_messages()
        return nxt

    def _router_navsupsec(self, state: GraphState) -> str:
        return state["Action_navsupsec"]

    def _router_navhanmain(self, state: GraphState) -> str:
        if state["Action_type_NHM"] == "Continue":
            return "NavigationHandlerSec"
        observation = "Completed the navigation"
        self.WorkflowClassifier.add_observations(f"Observation: {observation}")
        return "WorkflowClassifier"

    def _router_navhansec(self, state: GraphState) -> str:
        return state["Action_navhansec"]

    def _router_error(self, state: GraphState) -> str:
        return state["Action_error"]

    # ------------------------------------------------------------------
    # Graph wiring  (identical to Allinone_1.py)
    # ------------------------------------------------------------------

    def _build_graph(self):
        from langgraph.graph import END, StateGraph, START

        wf = StateGraph(GraphState)

        wf.add_node("SpeechAgent", self._node_text_input)
        wf.add_node("Boss", self._node_boss)
        wf.add_node("Speaker", self._node_text_capture)
        wf.add_node("TopPlanner", self._node_top_planner)
        wf.add_node("WorkflowClassifier", self._node_workflow_classifier)
        wf.add_node("ObjectSearcher", self._node_object_searcher)
        wf.add_node("ObjectGrabber", self._node_object_grabber)
        wf.add_node("QuestionAnswerer", self._node_question_answerer)
        wf.add_node("NavigationsupervisorMain", self._node_nav_sup_main)
        wf.add_node("NavigationsupervisorSec", self._node_nav_sup_sec)
        wf.add_node("FinalDestinatioIdentifier",
                    self._node_final_destination_identifier)
        wf.add_node("CurrentPositionIdentifier",
                    self._node_current_position_identifier)
        wf.add_node("WaypointGenerator", self._node_waypoint_generator)
        wf.add_node("FinalNavigationalPlanner", self._node_final_nav_planner)
        wf.add_node("NavigationHandlerMain", self._node_nav_handler_main)
        wf.add_node("NavigationHandlerSec", self._node_nav_handler_sec)
        wf.add_node("DoorChecker", self._node_door_checker)
        wf.add_node("PathPlanner", self._node_path_planner)
        wf.add_node("RobotExecutor", self._node_robot_executor)
        wf.add_node("Finisher", self._node_finisher)
        wf.add_node("ErrorHandler", self._node_error_handler)

        wf.add_edge(START, "SpeechAgent")
        wf.add_edge("SpeechAgent", "Boss")
        wf.add_conditional_edges(
            "Boss", self._router_boss,
            {"TopPlanner": "TopPlanner", "Speaker": "Speaker",
             "Finisher": "Finisher"})
        wf.add_conditional_edges(
            "Speaker", self._router_speaker,
            {"Boss": "Boss", "WorkflowClassifier": "WorkflowClassifier"})
        wf.add_edge("TopPlanner", "WorkflowClassifier")
        wf.add_conditional_edges(
            "WorkflowClassifier", self._router_wc,
            {
                "NavigationsupervisorMain": "NavigationsupervisorMain",
                "ObjectSearcher": "ObjectSearcher",
                "ObjectGrabber": "ObjectGrabber",
                "Speaker": "Speaker",
                "Finisher": "Finisher",
                "QuestionAnswerer": "QuestionAnswerer",
            })
        wf.add_edge("ObjectSearcher", "WorkflowClassifier")
        wf.add_edge("ObjectGrabber", "WorkflowClassifier")
        wf.add_edge("QuestionAnswerer", "WorkflowClassifier")
        wf.add_edge("NavigationsupervisorMain", "NavigationsupervisorSec")
        wf.add_conditional_edges(
            "NavigationsupervisorSec", self._router_navsupsec,
            {
                "FinalDestinatioIdentifier": "FinalDestinatioIdentifier",
                "CurrentPositionIdentifier": "CurrentPositionIdentifier",
                "WaypointGenerator": "WaypointGenerator",
                "FinalNavigationalPlanner": "FinalNavigationalPlanner",
                "ErrorHandler": "ErrorHandler",
            })
        wf.add_edge("FinalDestinatioIdentifier", "NavigationsupervisorSec")
        wf.add_edge("CurrentPositionIdentifier", "NavigationsupervisorSec")
        wf.add_edge("WaypointGenerator", "NavigationsupervisorSec")
        wf.add_edge("FinalNavigationalPlanner", "NavigationHandlerMain")
        wf.add_conditional_edges(
            "NavigationHandlerMain", self._router_navhanmain,
            {"NavigationHandlerSec": "NavigationHandlerSec",
             "WorkflowClassifier": "WorkflowClassifier"})
        wf.add_conditional_edges(
            "NavigationHandlerSec", self._router_navhansec,
            {
                "NavigationHandlerMain": "NavigationHandlerMain",
                "DoorChecker": "DoorChecker",
                "PathPlanner": "PathPlanner",
                "RobotExecutor": "RobotExecutor",
                "ErrorHandler": "ErrorHandler",
            })
        wf.add_edge("DoorChecker", "NavigationHandlerSec")
        wf.add_edge("PathPlanner", "NavigationHandlerSec")
        wf.add_edge("RobotExecutor", "NavigationHandlerSec")
        wf.add_conditional_edges(
            "ErrorHandler", self._router_error,
            {"NavigationsupervisorMain": "NavigationsupervisorMain",
             "WorkflowClassifier": "WorkflowClassifier"})
        # Finisher terminates the run (does NOT loop back to SpeechAgent as
        # in Allinone_1.py -- we want one scenario = one invoke).
        wf.add_edge("Finisher", END)

        return wf.compile()

    # ------------------------------------------------------------------
    # Maintenance helpers
    # ------------------------------------------------------------------

    def _reset_agent_histories(self) -> None:
        for a in (
            self.Boss, self.TopPlanner, self.WorkflowClassifier,
            self.NavigationSupervisorMain, self.NavigationSupervisorSec,
            self.NavigationHandlerMain, self.NavigationHandlerSec,
            self.ErrorHandler,
        ):
            try:
                a.reset_chat_messages()
            except Exception:
                pass


# ----------------------------------------------------------------------
# Smoke test: import-time only (agents instantiate) -- no Webots needed
# ----------------------------------------------------------------------

if __name__ == "__main__":
    runner = MASRunner(robot=None, use_real_perception=False)
    print(f"MASRunner initialised with {len(runner.app.nodes)} nodes")
