"""CPU-only smoke for the registered live-wake shape."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from adaptive_wake import CommandUserGenerator, ExperimentConfig, LiveWakeHarness, MultiSleepCoordinator, WakePlan


def generator() -> None:
    request = json.load(sys.stdin)
    session = request.get("session_id")
    if session is None:
        session = f"fake-{request['scenario']}-{request['turn']}"
        status = "started"
    else:
        status = "resumed"
    print(json.dumps({"message": str(request["goal"]), "session_id": session, "resume_status": status}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    parser.add_argument("--generator", action="store_true")
    args = parser.parse_args()
    if args.generator:
        generator()
        return
    if not args.out:
        raise SystemExit("--out is required for the smoke run")
    config = ExperimentConfig.from_dict({"arms": ["replay", "nosleep", "sft-ref"],
                                         "wakes": 6, "seeds": [1, 2, 3]})
    plan = WakePlan.from_dict({"turn_count": 4, "injection_turns": [1, 2, 3, 4]})
    harness = LiveWakeHarness(plan, CommandUserGenerator([sys.executable, __file__, "--generator"]), args.out)
    def wake(arm, wake, state, facts, scenario):
        holder = [state]
        artifact = harness.run(
            arm, wake, facts,
            lambda user: holder.__setitem__(0, holder[0] + 1) or f"fake local reply {arm}/{wake}: {user}",
            scenario, None, state_metadata=lambda: {"backbone": "fake", "state": holder[0]},
        )
        return artifact, holder[0]

    MultiSleepCoordinator(config.arms, wake, lambda state: state, 0, harness.store).run(
        facts_for_wake=lambda wake: [
            (f"entity_{wake}_{index}", f"{wake} {index} 0 0 0") for index in range(1, 5)
        ],
        scenario_for_wake=lambda wake: f"scenario-{wake}",
        sleep=lambda arm, wake, state, artifact: state + 1,
        probe=lambda arm, wake, state, artifact: None,
    )


if __name__ == "__main__":
    main()
