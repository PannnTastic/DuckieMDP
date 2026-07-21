"""Generate the M12 unified explanation JSON and CSV index."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.explainability.explanation_report import (
    build_unified_report,
    explanation_index_rows,
)


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "runs/explanations/m12_unified_report"
    output.mkdir(parents=True, exist_ok=True)

    report = build_unified_report(root)
    json_path = output / "unified_explanation_report.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rows = explanation_index_rows(report)
    csv_path = output / "local_explanation_index.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "accepted": report["acceptance"]["passed"],
        "local_cases": len(rows),
        "json": str(json_path.relative_to(root)),
        "csv": str(csv_path.relative_to(root)),
    }))


if __name__ == "__main__":
    main()
