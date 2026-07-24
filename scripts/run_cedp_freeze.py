"""CEDP0: freeze policy, explanation, baseline, and lexicon provenance."""

import argparse
from pathlib import Path

import yaml

from src.explainability.certified_primitives.provenance import freeze_manifest
from src.explainability.certified_primitives.reporting import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/explainability/cedp_v2.yaml"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output_dir or Path(config["experiment"]["output_dir"])
    manifest = freeze_manifest(
        policies=config["policies"],
        explanation_artifacts={
            key: Path(value) for key, value in config["explanation_artifacts"].items()
        },
        baselines={
            key: Path(value) for key, value in config["baselines_not_main_input"].items()
        },
        m2_lexicon=Path(config["experiment"]["m2_lexicon"]),
    )
    atomic_json(output / "provenance_manifest.json", manifest)
    print("CEDP0 PASS manifest_sha256=%s" % manifest["manifest_sha256"])


if __name__ == "__main__":
    main()
