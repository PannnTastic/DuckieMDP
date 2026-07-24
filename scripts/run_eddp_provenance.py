"""Run EDP0 provenance freeze."""

import argparse
from pathlib import Path

import yaml

from src.explainability.eddp.provenance import atomic_json, freeze_provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/explainability/eddp_v1.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    policies = {
        name: {
            "checkpoint": value["checkpoint"],
            "config": value["training_config"],
            "evaluation_mode": value["evaluation_mode"],
        }
        for name, value in config["policies"].items()
    }
    prior = {
        "m4_acceptance": Path("docs/m4_deterministic_replay_acceptance.json"),
        "m11_summary": Path("runs/explanations/m11_bottom_up_clustering/m11_summary.json"),
        "m12_summary": Path("runs/explanations/m12_policy_comparison/comparison_summary.json"),
        "m13_summary": Path("runs/explanations/m13_sarsa/m13_sarsa_explanation_report.json"),
    }
    # Older repositories may name the M4 acceptance artefact differently.
    if not prior["m4_acceptance"].is_file():
        candidates = sorted(Path("docs").glob("*m4*acceptance*.json"))
        if candidates:
            prior["m4_acceptance"] = candidates[0]
        else:
            prior.pop("m4_acceptance")
    payload = freeze_provenance(
        policies, Path(config["experiment"]["primitive_lexicon"]), prior
    )
    payload["eddp_config"] = str(args.config)
    payload["eddp_config_sha256"] = __import__(
        "src.explainability.eddp.provenance", fromlist=["file_sha256"]
    ).file_sha256(args.config)
    output = Path(config["experiment"]["output_dir"])
    path = output / "provenance_manifest.json"
    atomic_json(path, payload)
    print("provenance=%s policies=%d" % (path, len(policies)))


if __name__ == "__main__":
    main()
