# mas-llm-robotics

[![IEEE Access Paper](https://img.shields.io/badge/IEEE_Access-Paper-00629B.svg)](https://ieeexplore.ieee.org/abstract/document/11559625)

Implementation of **AgenticNav**, a Hierarchical Multi-Agentic System
(MAS) for LLM-driven autonomous problem-solving in robotics.

This repository contains the source code for AgenticNav and the two
comparison baselines, together with the per-agent prompts, the Webots
world and Pioneer 3-AT controller, and the TCP/JSON bridge between the
LLM-side orchestrator and the Webots-side robot.

For the evaluation harness, scenarios, results, grading rubric, and the
documentation that explains each evaluation, see the companion repository:

> **[mas-llm-robotics-eval](https://github.com/AgenticNav/mas-llm-robotics-eval)**

## Repository layout

- [`mas/`](mas/): AgenticNav. 17 agents in [`mas/agents/`](mas/agents/), 21-node LangGraph state machine in [`mas/app.py`](mas/app.py), shared topological/RRT tools in [`mas/tools/`](mas/tools/).
- [`baselines/`](baselines/): Baseline A ([`rule_based.py`](baselines/rule_based.py)) and Baseline B ([`single_llm.py`](baselines/single_llm.py)).
- [`prompts/`](prompts/): one prompt per LLM-driven agent.
- [`webots/`](webots/): Webots world ([`webots/worlds/home.wbt`](webots/worlds/home.wbt)) and the Pioneer 3-AT controller.
- [`comm/`](comm/), [`robot_server/`](robot_server/): TCP/JSON bridge between the LLM-side orchestrator and the Webots-side robot.
- [`requirements.txt`](requirements.txt): pinned Python dependencies.

## Architecture overview

AgenticNav is organised as a hierarchical Multi-Agentic System with three
logical layers: an interpretation layer that parses the user command into
a strategic plan; a supervisory layer that decomposes the plan into
sub-tasks and handles errors escalated from below; and an
execution-and-perception layer that interfaces with the robot and reports
perception events back upward. A failure observed at the execution layer
(e.g., an unexpectedly closed door) is propagated upward together with
the full mission context, allowing the supervisory layer to re-invoke the
strategic planner under the updated environmental constraints rather than
abort.

## Running end-to-end in Webots

End-to-end re-execution requires both repositories (this one for the
system code, the companion repo for the eval harness), Webots R2023b+,
an OpenAI API key, and a Python env with the pinned dependencies.
Estimated cost: ~3 hours of Webots wall-clock plus ~$5–10 in GPT-4o API.

### Setup

```bash
# Clone both repos side by side
git clone https://github.com/AgenticNav/mas-llm-robotics
git clone https://github.com/AgenticNav/mas-llm-robotics-eval
cd mas-llm-robotics-eval

# Conda env (or any equivalent venv)
conda create -n agent python=3.11
conda activate agent
pip install -r ../mas-llm-robotics/requirements.txt

# Make the code repo importable
export PYTHONPATH=$PYTHONPATH:$(pwd)/../mas-llm-robotics

# OpenAI API key
export OPENAI_API_KEY=sk-...
```

### Three-terminal launch

| # | Terminal | Command | What it does |
|---|----------|---------|--------------|
| 1 | Webots GUI | open `../mas-llm-robotics/webots/worlds/home.wbt` from inside Webots | Starts the simulator, the Pioneer 3-AT robot, and the controller in `webots/controllers/project_12_pioneercontroller_1/`. The controller connects to the TCP bridge in step 2. |
| 2 | Linux terminal | `python -m robot_server.mainserver` | Starts the bridge server on port 5000. Both the Webots controller and the orchestrator connect here. |
| 3 | Linux terminal | `python -m eval.run_experiment --systems rule_based single_llm mas --trials 1 --scenarios-file scenarios/scenarios.json --results-dir results/` | Runs each scenario on each of the three systems and writes per-scenario JSON logs. |

Verify connectivity before starting scenarios: terminal 2 should print
*"Server listening on 0.0.0.0:5000"* followed by *"Client 'webots'
connected from ..."* once the Webots controller has come up.

### Manual door step (before each scenario)

Each scenario in
[scenarios/scenarios.json](https://github.com/AgenticNav/mas-llm-robotics-eval/blob/master/scenarios/scenarios.json)
declares a `door_states` field. Doors not listed are open by default.
Before running a scenario, manually set the door positions in the Webots
GUI to match its `door_states` and place the robot at the scenario's
`initial_position`.

| Scenario | initial_position | door_states |
|---|---|---|
| s01, s02, s05, s06, s07, s08, s09, s10, s11, s12, s13, s15, s18, s19, s20 | R3 | (all open) |
| s03 | R3 | D3 closed |
| s04 | R3 | D3 closed, D4 closed |
| s14 | R2 | D2 closed |
| s16 | R1 | D2 closed |
| s17 | R2 | (all open) |

This manual step is the main reproduction friction. It can be automated
via the Webots Supervisor API but is left manual in this release to keep
the orchestrator independent of any Webots-version-specific binding.

### Useful invocation patterns

Single-scenario plan-only smoke test (rule_based and single_llm only,
no robot):

```bash
python -m eval.run_experiment --systems rule_based single_llm \
    --trials 1 --dry --scenarios s07 \
    --scenarios-file scenarios/scenarios.json \
    --results-dir results/_test
```

Re-aggregate results into the canonical CSVs (overwrites in place):

```bash
python -m eval.aggregate_results --results-dir results/
python scripts/replay_grade.py --verify-jsons
```

### Known caveats

- **MAS scenarios s05, s08, s14, s16, s17, s18 were not re-run for the
  comparative evaluation.** The original Table 1 assessment for those
  rows was carried over (`source=table_i` in `master_per_scenario.csv`).
  See `docs/comparative_evaluation.md` §5 in the companion repo. Re-running
  them in Tier 1 will produce richer logs.
- **MAS s04 fresh run can hit the LangGraph recursion limit (150).**
  Increase the `recursion_limit` argument in [`mas/runner.py`](mas/runner.py)
  if needed, or rely on the carry-over.
- **MAS fresh runs for s19 and s20 emit a hallucinated arrival pattern**
  (*"successfully arrived at the netball court"* / *"...canteen"*) rather
  than a clean refusal. The Table 1 record is a clean refusal; this
  divergence is one of the items reported by `replay_grade.py
  --verify-jsons`.
- **Single-LLM hallucinated confirmations on s01, s04, s05, s08, s09,
  s12** are converted from rubric-success to strict-grader-failure at
  aggregation time (the `strict_grader_hallucination_override` flag).
- **Wall-clock times are network-bound.** Per-scenario times depend on
  GPT-4o latency; ±20 % variance is expected on a different network.

## Citation

Please cite the corresponding [IEEE Access paper](https://ieeexplore.ieee.org/abstract/document/11559625) if you use this code.
Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

**BibTeX:**

```bibtex
@article{samarathunga2026agenticnav,
  title={AgenticNav: A Hierarchical Multi-Agentic System for LLM-Driven Autonomous Problem-Solving in Robotics},
  author={Samarathunga, Kaveesha and Gurusinghe, Ranuri and Sivasothynathan, Kugesan and Mars, Jason and Logeeshan, V and Wanigasekara, Chathura},
  journal={IEEE Access},
  year={2026},
  publisher={IEEE}
}

```


## License

MIT: see [`LICENSE`](LICENSE).
