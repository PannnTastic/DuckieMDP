"""Select the best SAC checkpoint using development seeds only."""

import argparse
import json
import shutil
from pathlib import Path

import yaml

from .evaluate_sac import evaluate_policy


def checkpoint_score(report):
    """Lexicographic criterion frozen in the experimental plan."""
    failure = report["total_failure_rate"]
    if report["stage"] == "lane":
        return (
            report["task_success_rate"],
            -failure,
            report["mean_return"],
            -report["p95_abs_d"],
        )
    return (
        report["task_success_rate"],
        -failure,
        report["stop_compliance_rate"],
        -report["false_stop_rate"],
        # Untuk ablation steering, pilih policy paling halus hanya setelah
        # seluruh kriteria keselamatan dan kepatuhan tetap terjaga.
        -report.get("mean_abs_omega_on_straight", float("inf")),
        report["mean_return"],
        -report["p95_abs_d"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    evaluation = config["evaluation"]
    episodes = int(evaluation["development_episodes"])
    seeds = [int(value) for value in evaluation["development_seeds"]]
    candidates = sorted(args.checkpoint_dir.glob("sac_step_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint in {args.checkpoint_dir}")

    records = []
    best = None
    best_score = None
    for checkpoint in candidates:
        report = evaluate_policy(args.config, checkpoint, episodes, seeds)
        score = checkpoint_score(report)
        records.append(
            {"checkpoint": str(checkpoint), "score": list(score), "report": report}
        )
        # Candidates are sorted by step. Strict > preserves the earlier
        # checkpoint when two candidates have exactly the same score.
        if best_score is None or score > best_score:
            best, best_score = checkpoint, score
        print(f"checkpoint={checkpoint.name} score={score}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.output)
    selection = args.output.with_name("checkpoint_selection.json")
    selection.write_text(
        json.dumps(
            {"selected": str(best), "output": str(args.output), "records": records},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"best={best}")
    print(f"copied_to={args.output}")


if __name__ == "__main__":
    main()
