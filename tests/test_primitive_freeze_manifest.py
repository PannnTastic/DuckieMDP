import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from src.explainability.primitives import (
    PRIMITIVE_SCHEMA_VERSION,
    PrimitiveThresholds,
)


MANIFEST = Path("docs/primitive_lexicon_v1.freeze.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_frozen_primitive_lexicon_content_has_not_changed():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["primitive_schema_version"] == PRIMITIVE_SCHEMA_VERSION
    for path, expected in payload["files"].items():
        assert _sha256(Path(path)) == expected, (
            "%s changed after primitive freeze; create a new schema version "
            "and rerun all dependent evaluation" % path
        )


def test_frozen_threshold_hash_matches_runtime_defaults():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    thresholds = asdict(PrimitiveThresholds())
    assert payload["threshold_config"] == thresholds
    canonical = json.dumps(
        thresholds,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == payload[
        "threshold_config_sha256"
    ]
    assert payload["independence_contract"][
        "clustering_executed_before_freeze"
    ] is False
