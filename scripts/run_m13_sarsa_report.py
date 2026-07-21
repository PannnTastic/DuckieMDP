"""Generate the fail-closed M13 SARSA explanation report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.explainability.sarsa_explanation_report import (
    build_sarsa_extension_report,
    sarsa_explanation_index_rows,
)


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "runs/explanations/m13_sarsa"
    output.mkdir(parents=True, exist_ok=True)

    report = build_sarsa_extension_report(root)
    json_path = output / "m13_sarsa_explanation_report.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows = sarsa_explanation_index_rows(report)
    csv_path = output / "m13_sarsa_local_explanation_index.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "accepted": report["acceptance"]["passed"],
        "failed_checks": report["acceptance"]["failed_checks"],
        "local_cases": len(rows),
        "json": str(json_path.relative_to(root)),
        "csv": str(csv_path.relative_to(root)),
    }
    print(json.dumps(result, sort_keys=True))
    if not report["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

