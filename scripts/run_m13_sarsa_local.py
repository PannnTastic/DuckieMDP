"""Generate M13 local paired explanations for the frozen SARSA policy."""

import argparse
import json
import logging
from pathlib import Path
import warnings

import yaml

from src.actions import ActionConfig
from src.env_wrapper import build_env
from src.explainability.action_outcomes import (
    EventHorizonConfig,
    prepare_branch,
    q_action,
    run_paired_outcomes,
)
from src.explainability.primitives import PrimitiveLabeler
from src.explainability.sarsa_policy_adapter import SarsaPolicyAdapter
from src.explainability.schema import to_dict


CASES = (
    {
        "name": "lane_correction",
        "config_role": "lane",
        "seed": 101,
        "target_primitive": "LaneCorrectLeft",
        "foil_action_id": 1,
    },
    {
        "name": "stop_hold",
        "config_role": "full",
        "seed": 101,
        "target_primitive": "StopHold",
        "foil_action_id": 4,
    },
    {
        "name": "pedestrian_yield",
        "config_role": "full",
        "seed": 101,
        "target_primitive": "YieldHold",
        "foil_action_id": 4,
    },
)


def _quiet():
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)


def _load_yaml(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    config["environment"]["render_observations"] = False
    return config


def _policy(checkpoint, config):
    return SarsaPolicyAdapter.from_checkpoint(
        checkpoint,
        allowed_actions=config["sarsa"]["allowed_actions"],
        action_config=ActionConfig(**config["actions"]),
    )


def _find_prefix(config, policy, seed, target_primitive, max_decisions=250):
    """Return actions strictly before the first real target decision."""
    env = build_env(config, int(seed))
    labeler = PrimitiveLabeler()
    prefix = []
    try:
        raw = env.reset(int(seed))
        for step in range(int(max_decisions)):
            decision = policy.decide_raw(raw)
            next_raw, _, done, info = env.step(int(decision.action.action_id))
            primitive = labeler.label(
                decision,
                info.get("events"),
                info.get("termination_reason", "in_progress"),
            )
            if primitive.primitive.value == target_primitive:
                return tuple(prefix), step, decision
            prefix.append(int(decision.action.action_id))
            raw = next_raw
            if done:
                break
    finally:
        env.close()
    raise RuntimeError(
        "SARSA target primitive %s not reached for seed %d"
        % (target_primitive, seed)
    )


def _final_return(branch):
    profiles = branch.reward_profile
    return None if not profiles else float(profiles[-1].discounted_total)


def _write_report(path, report, case, prefix_length):
    selected = report.selected_decision.action
    foil = report.foil_action
    text = """SARSA local explanation: {name}

Policy contract:
  trained with teacher guidance: yes
  teacher active during explanation: no
  policy mode: greedy, deterministic lowest-id tie break

Branch:
  manifest: {manifest}
  seed: {seed}
  prefix decisions: {prefix}
  selected: {selected_name} -> {selected_primitive}
  foil: {foil_name} -> {foil_primitive}
  selected discounted return: {selected_return:.6f}
  foil discounted return: {foil_return:.6f}

Interpretation:
{explanation}

This is a simulator-based interventional outcome, not a probability.
""".format(
        name=case["name"],
        manifest=report.manifest_id,
        seed=case["seed"],
        prefix=prefix_length,
        selected_name=selected.action_name,
        selected_primitive=report.factual.first_primitive,
        foil_name=foil.action_name,
        foil_primitive=report.counterfactual.first_primitive,
        selected_return=_final_return(report.factual),
        foil_return=_final_return(report.counterfactual),
        explanation=report.explanation,
    )
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "artifacts/ablation/full_task/sarsa_teacher/q_table_best.npy"
        ),
    )
    parser.add_argument(
        "--lane-config",
        type=Path,
        default=Path("configs/small_loop_lane_sarsa.yaml"),
    )
    parser.add_argument(
        "--full-config",
        type=Path,
        default=Path("configs/small_loop_stop_duck_sarsa.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/explanations/m13_sarsa/local"),
    )
    args = parser.parse_args()
    _quiet()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    configs = {
        "lane": (args.lane_config, _load_yaml(args.lane_config)),
        "full": (args.full_config, _load_yaml(args.full_config)),
    }
    reports = {}

    for case in CASES:
        config_path, config = configs[case["config_role"]]
        policy = _policy(args.checkpoint, config)
        prefix, branch_step, discovered = _find_prefix(
            config,
            policy,
            case["seed"],
            case["target_primitive"],
        )
        factory = lambda seed, cfg=config: build_env(cfg, int(seed))
        prepared = prepare_branch(
            env_factory=factory,
            reset_seed=case["seed"],
            action_prefix=prefix,
            policy=policy,
            config_path=config_path,
            checkpoint_path=args.checkpoint,
        )
        if prepared.selected_decision.action.action_id != discovered.action.action_id:
            raise AssertionError("reconstructed SARSA branch changed selected action")
        foil = q_action(policy, case["foil_action_id"])
        report = run_paired_outcomes(
            env_factory=factory,
            prepared=prepared,
            policy=policy,
            foil_action=foil,
            max_horizon=30,
            gamma=float(config["sarsa"]["gamma"]),
            event_horizon=EventHorizonConfig(enabled=False),
        )
        if report.factual.first_primitive != case["target_primitive"]:
            raise AssertionError(
                "%s produced %s, expected %s"
                % (
                    case["name"],
                    report.factual.first_primitive,
                    case["target_primitive"],
                )
            )
        json_path = output / ("sarsa_" + case["name"] + ".json")
        txt_path = output / ("sarsa_" + case["name"] + ".txt")
        json_path.write_text(
            json.dumps(to_dict(report), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        _write_report(txt_path, report, case, len(prefix))
        reports[case["name"]] = {
            "manifest_id": report.manifest_id,
            "seed": case["seed"],
            "branch_step": branch_step,
            "prefix_decisions": len(prefix),
            "selected_action": report.selected_decision.action.action_name,
            "selected_primitive": report.factual.first_primitive,
            "foil_action": report.foil_action.action_name,
            "foil_primitive": report.counterfactual.first_primitive,
            "selected_discounted_return_h30": _final_return(report.factual),
            "foil_discounted_return_h30": _final_return(report.counterfactual),
            "branch_invariants": dict(report.branch_invariants),
            "json": str(json_path),
            "text": str(txt_path),
        }

    invariants_valid = all(
        all(
            bool(values["branch_invariants"][name])
            for name in (
                "same_manifest",
                "same_policy_selected_action_at_branch",
                "only_first_action_forced",
                "selected_and_foil_differ",
            )
        )
        and values["branch_invariants"]["teacher_active"] is False
        for values in reports.values()
    )
    summary = {
        "stage": "M13-SARSA-local",
        "method": "COViz-inspired paired simulator action outcomes",
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": _policy(args.checkpoint, configs["full"][1]).checkpoint_hash,
        },
        "configs": {
            role: str(value[0]) for role, value in configs.items()
        },
        "policy_contract": {
            "training_algorithm": "sarsa",
            "trained_with_teacher_guidance": True,
            "teacher_active": False,
            "policy_mode": "greedy_teacher_free_lowest_id_tie_break",
        },
        "reports": reports,
        "acceptance": {
            "three_scenarios_generated": len(reports) == 3,
            "all_branch_invariants_valid": invariants_valid,
            "all_reports_use_sarsa_identity": all(
                json.loads(Path(values["json"]).read_text(encoding="utf-8"))[
                    "selected_decision"
                ]["solver"]
                == "sarsa"
                for values in reports.values()
            ),
            "passed": len(reports) == 3 and invariants_valid,
        },
    }
    summary_path = output.parent / "m13_local_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "accepted": summary["acceptance"]["passed"],
        "checkpoint": summary["checkpoint"],
        "reports": {
            name: {
                "selected": values["selected_primitive"],
                "foil": values["foil_primitive"],
            }
            for name, values in reports.items()
        },
        "summary": str(summary_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

