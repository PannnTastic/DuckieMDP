"""Run EDP3--EDP4 explanation atoms from real label-free anchors."""

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import warnings

from src.explainability.action_outcomes import (
    EventHorizonConfig, prepare_branch, run_paired_outcomes,
)
from src.explainability.eddp.counterfactual_profile import (
    choose_foil, counterfactual_profile,
)
from src.explainability.eddp.provenance import atomic_json
from src.explainability.eddp.runtime import environment_factory, load_runtime
from src.explainability.eddp.schema import (
    AnchorRecord, ExplanationAtom, read_jsonl, stable_id, write_jsonl,
)
from src.explainability.eddp.signature import (
    physical_profile_from_report, reward_profile_from_report,
)
from src.explainability.eddp.verification import verification_profile
from src.explainability.schema import to_dict


def _quiet():
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)


def _load_atom(path):
    return ExplanationAtom.from_dict(json.loads(path.read_text(encoding="utf-8")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/explainability/eddp_v1.yaml"))
    parser.add_argument("--max-atoms", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--skip-anchor", action="append", default=[])
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    _quiet()
    cfg, shared_path, shared, policies, gammas = load_runtime(args.config)
    output = Path(cfg["experiment"]["output_dir"])
    anchors = [
        AnchorRecord.from_dict(row)
        for row in read_jsonl(output / "anchors_label_free.jsonl")
    ]
    all_anchor_count = len(anchors)
    anchors = anchors[int(args.start_index):]
    if args.max_atoms:
        anchors = anchors[: int(args.max_atoms)]
    skipped_ids = set(args.skip_anchor)
    anchors = [anchor for anchor in anchors if anchor.anchor_id not in skipped_ids]
    counter_cfg = cfg["counterfactual"]
    pair_dir = output / "paired_reports"
    atom_dir = output / "atoms"
    pair_dir.mkdir(parents=True, exist_ok=True)
    atom_dir.mkdir(parents=True, exist_ok=True)
    atoms = []
    failures = []
    for index, anchor in enumerate(anchors, start=1):
        atom_path = atom_dir / (anchor.anchor_id + ".json")
        if atom_path.is_file() and not args.no_resume:
            atoms.append(_load_atom(atom_path))
            continue
        solver = anchor.solver.value
        policy = policies[solver]
        factory = environment_factory(shared, solver)
        try:
            prepared = prepare_branch(
                env_factory=factory,
                reset_seed=anchor.seed,
                action_prefix=anchor.action_prefix,
                policy=policy,
                config_path=shared_path,
                checkpoint_path=Path(anchor.checkpoint_path),
            )
            foil, foil_protocol = choose_foil(policy, prepared.selected_decision)
            report = run_paired_outcomes(
                env_factory=factory,
                prepared=prepared,
                policy=policy,
                foil_action=foil,
                max_horizon=int(counter_cfg["max_horizon"]),
                gamma=gammas[solver],
                fixed_horizons=tuple(counter_cfg["fixed_horizons"]),
                event_horizon=EventHorizonConfig(
                    enabled=bool(counter_cfg.get("event_horizon", False))
                ),
            )
            report_payload = to_dict(report)
            pair_path = pair_dir / (anchor.anchor_id + ".json")
            atomic_json(pair_path, report_payload)
            cf_profile = counterfactual_profile(policy, anchor.state)
            verify = verification_profile(policy, anchor.state)
            invariants = dict(report.branch_invariants)
            invariant_pass = all(
                bool(value) for name, value in invariants.items()
                if name != "teacher_active"
            ) and not bool(invariants.get("teacher_active", True))
            validity = {
                "counterfactual_valid_fraction": (
                    float(cf_profile["valid_attempts"])
                    / max(1, int(cf_profile["attempts"]))
                ),
                "branch_invariants_pass": invariant_pass,
                "paired_outcome_valid": invariant_pass,
                "foil_protocol": foil_protocol,
            }
            atom = ExplanationAtom(
                atom_id=stable_id(
                    {"anchor_id": anchor.anchor_id, "foil_protocol": foil_protocol},
                    "atom",
                ),
                anchor_id=anchor.anchor_id,
                solver=anchor.solver,
                seed=anchor.seed,
                episode_id=anchor.episode_id,
                decision_step=anchor.decision_step,
                block_id=anchor.block_id,
                block_offset=anchor.block_offset,
                selection_context=anchor.selection_context,
                observed_context=anchor.observed_context,
                counterfactual_profile=cf_profile,
                physical_profile=physical_profile_from_report(report_payload),
                reward_profile=reward_profile_from_report(report_payload),
                verification_profile=verify,
                validity=validity,
                paired_report_path=str(pair_path),
            )
            atomic_json(atom_path, atom.as_dict())
            atoms.append(atom)
        except Exception as error:
            failures.append({
                "anchor_id": anchor.anchor_id,
                "solver": solver,
                "seed": anchor.seed,
                "decision_step": anchor.decision_step,
                "error_type": type(error).__name__,
                "error": str(error),
            })
        if index % 10 == 0:
            print("processed=%d/%d success=%d failures=%d" % (
                index, len(anchors), len(atoms), len(failures)
            ), flush=True)

    # Rebuild from every atomically completed shard after process restarts.
    atoms = [_load_atom(path) for path in sorted(atom_dir.glob("*.json"))]
    write_jsonl(output / "explanation_atoms_label_free.jsonl", atoms)
    atomic_json(output / "counterfactual_failures.json", {"failures": failures})
    summary = {
        "stage": "EDP3-EDP4",
        "method": "state counterfactual + paired action-outcome + numeric verification",
        "requested_anchors": all_anchor_count,
        "selected_in_this_process": len(anchors),
        "explicitly_quarantined_anchor_ids": sorted(skipped_ids),
        "successful_atoms": len(atoms),
        "failures": len(failures),
        "solver_counts": dict(sorted(Counter(
            atom.solver.value for atom in atoms
        ).items())),
        "context_counts": dict(sorted(Counter(
            atom.selection_context for atom in atoms
        ).items())),
        "files": {
            "atoms": str(output / "explanation_atoms_label_free.jsonl"),
            "paired_reports": str(pair_dir),
            "failures": str(output / "counterfactual_failures.json"),
        },
        "acceptance": {
            "at_least_95_percent_success": len(atoms) >= 0.95 * max(1, all_anchor_count),
            "all_branch_invariants_pass": all(
                atom.validity["branch_invariants_pass"] for atom in atoms
            ),
            "no_m2_primitive_in_atom_schema": all(
                "primitive" not in json.dumps(atom.as_dict()).lower()
                for atom in atoms
            ),
            "teacher_inactive": True,
        },
    }
    summary["acceptance"]["passed"] = all(summary["acceptance"].values())
    atomic_json(output / "counterfactual_summary.json", summary)
    print(json.dumps({
        "successful_atoms": len(atoms), "failures": len(failures),
        "passed": summary["acceptance"]["passed"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
