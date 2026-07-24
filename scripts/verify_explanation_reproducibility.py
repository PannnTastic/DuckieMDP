"""Verify that a fresh clone contains a complete four-policy explanation bundle."""

from __future__ import annotations

import argparse
import contextlib
from hashlib import sha256
import importlib.metadata
import io
import json
import logging
from pathlib import Path
import platform
import sys
import warnings as warning_control

import numpy as np
import yaml



ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(
    "artifacts/explainability/four_policy/reproducibility_manifest.json"
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_record(record: dict, errors: list[str]) -> None:
    path = ROOT / record["path"]
    if not path.is_file():
        errors.append("missing file: %s" % record["path"])
        return
    if path.stat().st_size != int(record["bytes"]):
        errors.append("size mismatch: %s" % record["path"])
    actual = _sha256(path)
    if actual != record["sha256"]:
        errors.append("sha256 mismatch: %s" % record["path"])


def _check_evidence(manifest: dict, errors: list[str]) -> dict:
    evidence_path = (
        ROOT / "artifacts/explainability/four_policy/primitive_real_evidence.json"
    )
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    expected = manifest["expected_results"]
    actual_names = sorted(payload["primitives"])
    if actual_names != sorted(expected["primitive_families"]):
        errors.append("primitive family mismatch: %r" % actual_names)
    if int(payload["total_instances"]) != int(expected["temporal_instances"]):
        errors.append("temporal instance count mismatch")
    if int(payload["total_decisions"]) != int(expected["explained_decisions"]):
        errors.append("explained decision count mismatch")
    solver_names = sorted({
        solver
        for card in payload["primitives"].values()
        for solver in card["solver_instances"]
    })
    if solver_names != sorted(expected["solvers"]):
        errors.append("solver coverage mismatch: %r" % solver_names)

    paired = list(
        (ROOT / "artifacts/explainability/four_policy/paired_outcomes").glob(
            "*.json"
        )
    )
    if len(paired) != int(expected["paired_local_outcomes"]):
        errors.append("paired local outcome count mismatch")
    for path in paired:
        report = json.loads(path.read_text(encoding="utf-8"))
        invariants = report["branch_invariants"]
        required = (
            "same_manifest",
            "same_policy_selected_action_at_branch",
            "only_first_action_forced",
            "selected_and_foil_differ",
        )
        if not all(bool(invariants[name]) for name in required):
            errors.append("invalid paired branch invariants: %s" % path.name)
        if invariants.get("teacher_active") is not False:
            errors.append("teacher active in paired outcome: %s" % path.name)
    return {
        "primitive_families": actual_names,
        "temporal_instances": payload["total_instances"],
        "explained_decisions": payload["total_decisions"],
        "paired_local_outcomes": len(paired),
    }


def _package_versions() -> dict:
    names = (
        "duckietown-gym-daffy",
        "gym",
        "numpy",
        "torch",
        "scikit-learn",
        "PyYAML",
    )
    return {name: importlib.metadata.version(name) for name in names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--strict-python",
        action="store_true",
        help="Require Python 3.9.15 exactly instead of reporting a warning.",
    )
    args = parser.parse_args()
    manifest_path = ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []

    for section in ("canonical_config",):
        _check_record(manifest[section], errors)
    for policy in manifest["policies"].values():
        _check_record(policy["checkpoint"], errors)
        _check_record(policy["training_config"], errors)
        if policy["teacher_active_during_explanation"]:
            errors.append("teacher must be inactive during explanation")
    for record in manifest["evidence"] + manifest["critical_source"]:
        _check_record(record, errors)

    required_python = manifest["python"]
    actual_python = platform.python_version()
    if actual_python != required_python:
        message = "Python %s active; frozen run used %s" % (
            actual_python,
            required_python,
        )
        (errors if args.strict_python else warnings).append(message)

    q_shape = tuple(manifest["q_table_shape"])
    for solver in ("q_learning", "sarsa"):
        checkpoint = ROOT / manifest["policies"][solver]["checkpoint"]["path"]
        if tuple(np.load(checkpoint, mmap_mode="r").shape) != q_shape:
            errors.append("%s Q-table shape changed" % solver)

    config_path = ROOT / manifest["canonical_config"]["path"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if Path(config["experiment"]["output_dir"]).parts[0] != "runs":
        errors.append("generated output must remain under ignored runs/")
    import_output = io.StringIO()
    with contextlib.ExitStack() as stack:
        stack.enter_context(warning_control.catch_warnings())
        stack.enter_context(contextlib.redirect_stdout(import_output))
        stack.enter_context(contextlib.redirect_stderr(import_output))
        warning_control.simplefilter("ignore")
        logging.disable(logging.CRITICAL)
        from src.explainability.eddp.runtime import load_runtime
        runtime = load_runtime(config_path)
    policies = runtime[3]
    expected_solvers = set(manifest["expected_results"]["solvers"])
    if set(policies) != expected_solvers:
        errors.append("runtime policy set mismatch: %r" % sorted(policies))
    for solver, policy in policies.items():
        expected_hash = manifest["policies"][solver]["checkpoint"]["sha256"]
        if policy.checkpoint_hash != expected_hash:
            errors.append("%s adapter checkpoint hash mismatch" % solver)

    evidence = _check_evidence(manifest, errors)
    result = {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "python": {
            "actual": actual_python,
            "required": required_python,
            "executable": sys.executable,
        },
        "packages": _package_versions(),
        "manifest_sha256": manifest["manifest_sha256"],
        "runtime_solvers": sorted(policies),
        "evidence": evidence,
    }
    if args.output:
        destination = ROOT / args.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
