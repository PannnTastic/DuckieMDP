"""Generate the validated M6 cross-solver response-curve experiment."""

import argparse
from hashlib import sha256
import json
import logging
from pathlib import Path
import warnings

import numpy as np
import yaml

from src.actions import ActionConfig
from src.continuous_env import build_continuous_env
from src.explainability.q_policy_adapter import QPolicyAdapter
from src.explainability.response_curves import (
    SweepSpec,
    run_response_suite,
    save_response_suite,
)
from src.explainability.sac_policy_adapter import SACPolicyAdapter
from src.explainability.schema import SolverKind, to_dict


def _quiet() -> None:
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)


def _load_yaml(path: Path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _capture_anchor(policy, config, seed, predicate, max_steps=800):
    env = build_continuous_env(config, seed)
    try:
        env.reset(seed)
        for step in range(max_steps):
            decision = policy.decide_continuous(env.current_state)
            if predicate(step, decision.state, decision.action):
                return decision.state, step
            action = np.asarray(
                [decision.action.v_cmd, decision.action.omega_cmd],
                dtype=np.float32,
            )
            _, _, done, info = env.step(action)
            if done:
                raise RuntimeError(
                    "episode terminated before anchor: %s"
                    % info.get("termination_reason")
                )
    finally:
        env.close()
    raise RuntimeError("no real rollout anchor matched within %d steps" % max_steps)


def _specs():
    return {
        "lane": (
            SweepSpec("d", (-0.30, -0.25, -0.20, -0.10, 0.0, 0.10, 0.20, 0.25, 0.30)),
            SweepSpec("phi", (-1.70, -1.20, -0.60, -0.30, 0.0, 0.30, 0.60, 1.20, 1.70)),
            SweepSpec("curvature", (-9.0, -8.0, -4.0, -1.0, 0.0, 1.0, 4.0, 8.0, 9.0)),
        ),
        "stop": (
            SweepSpec("stop_distance", (0.0, 0.15, 0.30, 0.45, 0.75, 1.50, 3.0, 3.20)),
            SweepSpec("stop_hold_progress", (0.0, 0.33, 0.66, 1.0)),
        ),
        "duck": (
            SweepSpec("duck_longitudinal", (-2.20, -2.0, -0.2, 0.0, 0.2, 0.4, 0.6, 0.9, 1.2, 1.8, 2.0, 2.20)),
            SweepSpec("duck_lateral", (-2.20, -2.0, -1.2, -0.6, -0.3, 0.0, 0.3, 0.6, 1.2, 2.0, 2.20)),
            SweepSpec("duck_present", (False, True), intervention_name="duck_absent_present"),
            SweepSpec("duck_active", (False, True), intervention_name="duck_inactive_active"),
            SweepSpec("duck_crossing_available", (False, True), intervention_name="duck_unavailable_available"),
        ),
    }


def _curve_summary(curve):
    support_status = (
        "unknown_no_visit_count_artifact"
        if curve.solver == SolverKind.Q_LEARNING
        else "anchor_reachable_synthetic_points_not_reachability_proven"
    )
    return {
        "support_status": support_status,
        "anchor_id": curve.anchor_id,
        "feature": curve.feature,
        "solver": curve.solver.value,
        "anchor_action": to_dict(curve.anchor_decision.action),
        "anchor_primitive": curve.anchor_primitive.primitive.value,
        "valid_points": curve.valid_points,
        "rejected_points": curve.rejected_points,
        "action_flip_points": sum(
            point.action_changed_from_anchor is True for point in curve.points
        ),
        "primitive_flip_points": sum(
            point.primitive_changed_from_anchor is True for point in curve.points
        ),
        "minimal_action_counterfactual": (
            None
            if curve.minimal_action_counterfactual is None
            else to_dict(curve.minimal_action_counterfactual)
        ),
        "minimal_primitive_counterfactual": (
            None
            if curve.minimal_primitive_counterfactual is None
            else to_dict(curve.minimal_primitive_counterfactual)
        ),
        "rejection_codes": sorted(
            {
                code
                for point in curve.points
                for code in point.synthetic.validation.reason_codes
            }
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sac-checkpoint",
        type=Path,
        default=Path("artifacts/sac/full_repeat_duck_5min/sac_best.pt"),
    )
    parser.add_argument(
        "--sac-full-config",
        type=Path,
        default=Path("artifacts/sac/full_repeat_duck_5min/config.yaml"),
    )
    parser.add_argument(
        "--sac-lane-config",
        type=Path,
        default=Path("configs/sac_lane.yaml"),
    )
    parser.add_argument(
        "--q-checkpoint",
        type=Path,
        default=Path("artifacts/ablation/full_task/q_learning_teacher/q_table_best.npy"),
    )
    parser.add_argument(
        "--q-config",
        type=Path,
        default=Path("configs/small_loop_stop_duck_q.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/explanations/m6_response_curves"),
    )
    args = parser.parse_args()
    _quiet()

    lane_config = _load_yaml(args.sac_lane_config)
    full_config = _load_yaml(args.sac_full_config)
    lane_config["environment"]["render_observations"] = False
    full_config["environment"]["render_observations"] = False
    q_config = _load_yaml(args.q_config)

    sac = SACPolicyAdapter.from_checkpoint(
        args.sac_checkpoint,
        allow_observation_expansion=True,
    )
    q_policy = QPolicyAdapter.from_checkpoint(
        args.q_checkpoint,
        allowed_actions=q_config["q_learning"]["allowed_actions"],
        action_config=ActionConfig(**q_config["actions"]),
    )

    lane_anchor, lane_step = _capture_anchor(
        sac,
        lane_config,
        701,
        lambda step, state, action: step >= 5,
    )
    stop_anchor, stop_step = _capture_anchor(
        sac,
        full_config,
        30101,
        lambda step, state, action: (
            state.stop_present
            and not state.stop_satisfied
            and state.stop_distance is not None
            and state.stop_distance <= 0.45
        ),
    )
    duck_anchor, duck_step = _capture_anchor(
        sac,
        full_config,
        30101,
        lambda step, state, action: (
            state.duck_present
            and state.duck_active is True
            and state.duck_longitudinal is not None
            and (state.duck_longitudinal ** 2 + state.duck_lateral ** 2) ** 0.5
            <= 0.40
        ),
    )
    anchors = {
        "lane": (lane_anchor, 701, lane_step),
        "stop": (stop_anchor, 30101, stop_step),
        "duck": (duck_anchor, 30101, duck_step),
    }

    output = args.output_dir
    summaries = []
    written = []
    policies = (
        (SolverKind.Q_LEARNING, q_policy),
        (SolverKind.SAC, sac),
    )
    for scenario, (anchor, seed, step) in anchors.items():
        for solver, policy in policies:
            curves = run_response_suite(
                policy,
                solver,
                anchor,
                _specs()[scenario],
            )
            written.extend(
                str(path)
                for path in save_response_suite(
                    curves,
                    output,
                    prefix=scenario,
                )
            )
            summaries.extend(_curve_summary(curve) for curve in curves)

    payload = {
        "stage": "M6",
        "method": "valid-manifold response curves",
        "policy_mode": {
            "q_learning": "greedy_teacher_free",
            "sac": "deterministic_actor_mean",
        },
        "policy_checkpoints": {
            "q_learning": {
                "path": str(args.q_checkpoint),
                "sha256": q_policy.checkpoint_hash,
            },
            "sac": {
                "path": str(args.sac_checkpoint),
                "sha256": sac.checkpoint_hash,
                "checkpoint_observation_dim": sac.checkpoint_obs_dim,
                "adapter_observation_dim": sac.observation_dim,
                "observation_expanded": sac.observation_expanded,
            },
        },
        "configs": {
            "q_full": {
                "path": str(args.q_config),
                "sha256": _sha256(args.q_config),
            },
            "sac_full": {
                "path": str(args.sac_full_config),
                "sha256": _sha256(args.sac_full_config),
            },
            "sac_lane_anchor_environment": {
                "path": str(args.sac_lane_config),
                "sha256": _sha256(args.sac_lane_config),
            },
        },
        "anchors": {
            name: {
                "seed": seed,
                "decision_step": step,
                "state": to_dict(anchor),
            }
            for name, (anchor, seed, step) in anchors.items()
        },
        "curves": summaries,
        "totals": {
            "curves": len(summaries),
            "valid_points": sum(item["valid_points"] for item in summaries),
            "rejected_points": sum(item["rejected_points"] for item in summaries),
            "action_flip_points": sum(item["action_flip_points"] for item in summaries),
            "primitive_flip_points": sum(item["primitive_flip_points"] for item in summaries),
        },
        "generated_files": written,
    }
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "m6_summary.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    print(json.dumps(payload["totals"], sort_keys=True))
    for item in summaries:
        print(
            "%s/%s valid=%d rejected=%d action_flips=%d primitive_flips=%d"
            % (
                item["solver"],
                item["feature"],
                item["valid_points"],
                item["rejected_points"],
                item["action_flip_points"],
                item["primitive_flip_points"],
            )
        )
    print("summary=%s" % destination)


if __name__ == "__main__":
    main()
