"""Summarize support-aware verification and primitive certification gates."""

import argparse
from collections import Counter
import json
from pathlib import Path

from src.explainability.certified_primitives.reporting import atomic_json
from src.explainability.certified_primitives.schema import read_jsonl
from src.explainability.eddp.verification import RELATIONS


def _relation_summary(records, relation):
    profiles = [
        row["verification_evidence"]["verification_profile"]
        for row in records
    ]
    applicable = [
        profile for profile in profiles
        if profile.get("%s_applicable" % relation, False)
    ]
    status = Counter(
        str(profile.get("%s_status" % relation, "UNKNOWN"))
        for profile in profiles
    )
    strata = Counter(
        str(profile.get("%s_pair_stratum" % relation, "unknown"))
        for profile in applicable
    )
    return {
        "total_states": len(profiles),
        "applicable": len(applicable),
        "eligible": sum(
            bool(profile.get("%s_eligible" % relation, False))
            for profile in profiles
        ),
        "status_counts": dict(sorted(status.items())),
        "pair_strata": dict(sorted(strata.items())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Support-aware C-EDDP output directory",
    )
    args = parser.parse_args()
    output = args.output_dir
    instances = tuple(
        read_jsonl(output / "certified_explanation_instances.jsonl")
    )
    primitives = tuple(
        read_jsonl(output / "primitive_certificates.jsonl")
    )
    solvers = sorted({str(row["solver"]) for row in instances})
    by_solver = {}
    for solver in solvers:
        records = [
            row for row in instances if str(row["solver"]) == solver
        ]
        by_solver[solver] = {
            relation: _relation_summary(records, relation)
            for relation in RELATIONS
        }

    failed_gates = Counter()
    for primitive in primitives:
        for name, passed in primitive["gate_results"].items():
            if not passed:
                failed_gates[str(name)] += 1
    summary = {
        "stage": "CEDP-support-aware-audit",
        "instances": len(instances),
        "solvers": solvers,
        "relations_by_solver": by_solver,
        "primitive_status_counts": dict(sorted(Counter(
            str(row["status"]) for row in primitives
        ).items())),
        "primitive_failed_gate_counts": dict(sorted(failed_gates.items())),
        "claim_semantics": {
            "eligible_relation_pair": "source and target are both supported",
            "unsupported_pair": "ABSTAIN; neither PASS nor FAIL",
            "tabular_support": (
                "evaluation count in frozen C-EDDP anchors, not training visits"
            ),
        },
    }
    atomic_json(output / "support_aware_audit.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
