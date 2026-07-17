import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import yaml

from .discretizer import Q_SHAPE
from .transition_model import EmpiricalTransitionModel, State


def value_iteration(
    model: EmpiricalTransitionModel,
    allowed_actions: Iterable[int],
    gamma: float = 0.99,
    tolerance: float = 1e-8,
    max_iterations: int = 10000,
) -> Tuple[np.ndarray, dict]:
    allowed = tuple(int(action) for action in allowed_actions)
    states = model.source_states
    values = {state: 0.0 for state in states}
    iterations, residual = 0, float("inf")
    for iterations in range(1, max_iterations + 1):
        updated = {}
        residual = 0.0
        for state in states:
            action_values = []
            for action in allowed:
                outcomes = model.outcomes(state, action)
                if not outcomes:
                    continue
                action_values.append(
                    sum(
                        probability
                        * (reward + (0.0 if terminal else gamma * values.get(next_state, 0.0)))
                        for probability, next_state, reward, terminal in outcomes
                    )
                )
            new_value = max(action_values) if action_values else 0.0
            updated[state] = new_value
            residual = max(residual, abs(new_value - values[state]))
        values = updated
        if residual < tolerance:
            break

    q = np.full(Q_SHAPE, -1.0e9, dtype=np.float32)
    for state in states:
        for action in allowed:
            outcomes = model.outcomes(state, action)
            if outcomes:
                q[state + (action,)] = sum(
                    probability
                    * (reward + (0.0 if terminal else gamma * values.get(next_state, 0.0)))
                    for probability, next_state, reward, terminal in outcomes
                )
    report = {
        **model.coverage(allowed),
        "iterations": iterations,
        "final_residual": residual,
        "gamma": gamma,
        "warning": "Unobserved state-action pairs are excluded from maximization.",
    }
    return q, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Value iteration on an empirical Duckietown MDP model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=10000)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model = EmpiricalTransitionModel.load(args.model)
    q, report = value_iteration(
        model,
        config["q_learning"]["allowed_actions"],
        float(config["q_learning"]["gamma"]),
        args.tolerance,
        args.max_iterations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(args.output), q)
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
