"""C-EDDP provenance freeze over policies and M1--M13 evidence artefacts."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from .schema import CEDP_SCHEMA_VERSION, file_sha256


def _hash_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def freeze_manifest(
    *,
    policies: Mapping[str, Mapping[str, Any]],
    explanation_artifacts: Mapping[str, Path],
    baselines: Mapping[str, Path],
    m2_lexicon: Path,
) -> Mapping[str, Any]:
    def freeze_files(records):
        result = {}
        for name, path_value in records.items():
            path = Path(path_value)
            if not path.is_file():
                raise FileNotFoundError(path)
            result[str(name)] = {"path": str(path), "sha256": file_sha256(path)}
        return result

    frozen_policies = {}
    for name, record in policies.items():
        checkpoint = Path(record["checkpoint"])
        config = Path(record["training_config"])
        if not checkpoint.is_file() or not config.is_file():
            raise FileNotFoundError("missing frozen policy artefact for %s" % name)
        frozen_policies[name] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "training_config": str(config),
            "training_config_sha256": file_sha256(config),
            "evaluation_mode": str(record["evaluation_mode"]),
            "teacher_active_during_explanation": False,
        }
    lexicon = Path(m2_lexicon)
    if not lexicon.is_file():
        raise FileNotFoundError(lexicon)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        commit = "unavailable"
    payload = {
        "stage": "CEDP0",
        "schema_version": CEDP_SCHEMA_VERSION,
        "git_commit": commit,
        "policies": frozen_policies,
        "m1_m13_explanation_artifacts": freeze_files(explanation_artifacts),
        "baselines_not_main_input": freeze_files(baselines),
        "m2_lexicon": {
            "path": str(lexicon),
            "sha256": file_sha256(lexicon),
            "role": "sealed external reconciliation only",
        },
        "method_contract": {
            "main_input": "M1--M13 explanation instances",
            "eddp_v1_role": "baseline and migration smoke only",
            "teacher_inactive": True,
            "policies_not_retrained": True,
            "primitive_result_scope": (
                "descriptive families carrying Why, What-if, Verification, "
                "and Temporal evidence"
            ),
            "cluster_certification_role": "optional audit",
        },
    }
    payload["manifest_sha256"] = _hash_payload(payload)
    return payload
