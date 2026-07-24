"""Create a deterministic hash manifest for the explanation release bundle."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("configs/explainability/four_policy_reproducible.yaml")
DEFAULT_OUTPUT = Path(
    "artifacts/explainability/four_policy/reproducibility_manifest.json"
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict:
    absolute = ROOT / path
    if not absolute.is_file():
        raise FileNotFoundError(path)
    return {
        "path": path.as_posix(),
        "bytes": absolute.stat().st_size,
        "sha256": _sha256(absolute),
    }


def _critical_source_files() -> list[Path]:
    paths = [
        Path("requirements-explainability-lock.txt"),
        Path(".python-version"),
        Path("docs/primitive_lexicon_v1.freeze.json"),
        Path("configs/explainability/shared_gated_duck_config.yaml"),
        Path("src/actions.py"),
        Path("src/continuous_env.py"),
        Path("src/continuous_state.py"),
        Path("src/discretizer.py"),
        Path("src/duck_controller.py"),
        Path("src/env_wrapper.py"),
        Path("src/reward.py"),
        Path("src/state.py"),
    ]
    paths.extend(
        path.relative_to(ROOT)
        for path in sorted((ROOT / "src/agents").glob("*.py"))
    )
    paths.extend(
        path.relative_to(ROOT)
        for path in sorted((ROOT / "src/explainability").glob("**/*.py"))
    )
    script_names = {
        "freeze_explanation_release.py",
        "generate_primitive_real_evidence.py",
        "verify_explanation_reproducibility.py",
    }
    for path in sorted((ROOT / "scripts").glob("*")):
        if not path.is_file():
            continue
        if (
            path.name in script_names
            or path.name == "reproduce_explanation_pipeline.sh"
            or path.name.startswith("run_cedp_")
            or path.name.startswith("run_eddp_")
            or path.name.startswith("run_m")
        ):
            paths.append(path.relative_to(ROOT))
    return sorted(set(paths))

def build_manifest(config_path: Path) -> dict:
    config = yaml.safe_load((ROOT / config_path).read_text(encoding="utf-8"))
    policies = {}
    for solver, values in sorted(config["policies"].items()):
        policies[solver] = {
            "checkpoint": _record(Path(values["checkpoint"])),
            "training_config": _record(Path(values["training_config"])),
            "evaluation_mode": values["evaluation_mode"],
            "teacher_active_during_explanation": False,
        }

    evidence_root = ROOT / "artifacts/explainability/four_policy"
    evidence = []
    for path in sorted(evidence_root.glob("**/*")):
        if not path.is_file() or path.name in {
            "reproducibility_manifest.json",
            "SHA256SUMS",
        }:
            continue
        evidence.append(_record(path.relative_to(ROOT)))

    payload = {
        "schema_version": "1.0.0",
        "bundle": "four-policy-explanation-derived-driving-primitives",
        "canonical_config": _record(config_path),
        "python": "3.9.15",
        "q_table_shape": [5, 5, 3, 3, 4, 2, 5, 7],
        "policy_modes": {
            "q_learning": "greedy_teacher_free",
            "sarsa": "greedy_teacher_free",
            "sac": "deterministic_actor_mean",
            "td3": "deterministic_actor_mean",
        },
        "policies": policies,
        "evidence": evidence,
        "critical_source": [
            _record(path) for path in _critical_source_files()
        ],
        "expected_results": {
            "solvers": ["q_learning", "sac", "sarsa", "td3"],
            "primitive_families": [
                "CurveNegotiation",
                "LaneKeeping",
                "PedestrianYield",
                "StopCompliance",
            ],
            "temporal_instances": 238,
            "explained_decisions": 4000,
            "paired_local_outcomes": 6,
        },
        "scope": {
            "main_result": (
                "explanation-derived primitive families carrying Why, "
                "What-if, Verification, and Temporal evidence"
            ),
            "support_aware_cluster_certification": "optional audit",
            "video": "regenerated locally and intentionally excluded from Git",
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    payload["manifest_sha256"] = sha256(encoded).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_manifest(args.config)
    destination = ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    records = [payload["canonical_config"]]
    for policy in payload["policies"].values():
        records.extend((policy["checkpoint"], policy["training_config"]))
    records.extend(payload["evidence"])
    records.extend(payload["critical_source"])
    unique = {record["path"]: record["sha256"] for record in records}
    sums_path = ROOT / "artifacts/explainability/four_policy/SHA256SUMS"
    sums_path.write_text(
        "".join(
            "%s  %s\n" % (digest, path)
            for path, digest in sorted(unique.items())
        ),
        encoding="utf-8",
    )
    print(
        "explanation release frozen: files=%d manifest_sha256=%s"
        % (
            len(payload["evidence"]) + len(payload["critical_source"]),
            payload["manifest_sha256"],
        )
    )


if __name__ == "__main__":
    main()
