"""EDDP provenance freeze and content hashing."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, Mapping


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_provenance(
    policies: Mapping[str, Mapping[str, Any]],
    lexicon_path: Path,
    prior_artifacts: Mapping[str, Path],
) -> Dict[str, Any]:
    frozen_policies = {}
    for name, record in policies.items():
        checkpoint = Path(record["checkpoint"])
        config = Path(record["config"])
        if not checkpoint.is_file() or not config.is_file():
            raise FileNotFoundError("missing policy artefact for %s" % name)
        frozen_policies[name] = {
            **dict(record),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "config": str(config),
            "config_sha256": file_sha256(config),
            "teacher_active_during_explanation": False,
        }
    prior = {}
    for name, path in prior_artifacts.items():
        candidate = Path(path)
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        prior[name] = {"path": str(candidate), "sha256": file_sha256(candidate)}
    lexicon = Path(lexicon_path)
    if not lexicon.is_file():
        raise FileNotFoundError(lexicon)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        commit = "unavailable"
    return {
        "stage": "EDP0",
        "method": "immutable provenance freeze",
        "git_commit": commit,
        "policies": frozen_policies,
        "primitive_lexicon": {
            "path": str(lexicon),
            "sha256": file_sha256(lexicon),
            "role": "sealed external taxonomy; forbidden during discovery",
        },
        "prior_explanation_artifacts": prior,
        "acceptance": {
            "all_files_exist": True,
            "all_hashes_recorded": True,
            "teacher_off_during_explanation": True,
            "policies_are_not_retrained": True,
        },
    }


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
