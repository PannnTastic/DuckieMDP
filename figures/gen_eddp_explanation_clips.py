#!/usr/bin/env python3
"""Create short data-only explanation GIFs for frozen EDDP cluster cards."""

import json
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt


ROOT = Path("runs/explanations/eddp_v1")
OUT = ROOT / "explanation_clips"
COLORS = {"factual": "#0072B2", "foil": "#D55E00"}


def _atoms():
    return [
        json.loads(line)
        for line in (ROOT / "explanation_atoms_label_free.jsonl").read_text().splitlines()
        if line.strip()
    ]


def _value(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return ("%%.%df" % digits) % value
    return str(value)


def _flip_text(atom):
    flips = [
        name[:-5].replace("_", " ")
        for name, value in atom["counterfactual_profile"].items()
        if name.endswith("_flip") and value
    ]
    return ", ".join(flips) if flips else "no tested one-feature action flip"


def _draw(frame, ax, card, atom, report):
    ax.clear()
    ax.axis("off")
    factual = report["factual"]["physical"]
    foil = report["counterfactual"]["physical"]
    selected = report["selected_decision"]["action"]
    contrast = report["foil_action"]
    ax.text(0.02, 0.96, card["candidate_name"], fontsize=16,
            fontweight="bold", va="top", color="#1F2937")
    ax.text(0.02, 0.89,
            "%s | context=%s | temporal atom %d/%d" % (
                atom["solver"], atom["selection_context"], frame + 1,
                len(card["_atoms"]),
            ), fontsize=10, color="#4B5563", va="top")
    ax.text(0.02, 0.81, "Decision explanation", fontsize=11,
            fontweight="bold", color="#111827")
    ax.text(0.04, 0.75, "State counterfactual flips: " + _flip_text(atom),
            fontsize=9, va="top", wrap=True)
    ax.text(0.04, 0.69,
            "Selected: %s   |   Pre-registered foil: %s" % (
                selected.get("action_name", selected),
                contrast.get("action_name", contrast),
            ), fontsize=9, va="top")

    ax.text(0.02, 0.60, "Paired action-outcome explanation (H=15)",
            fontsize=11, fontweight="bold", color="#111827")
    labels = ["progress (m)", "max |d| (m)", "max |phi| (rad)",
              "brake ratio", "steer reversals", "stop violations"]
    keys = ["forward_progress_m", "max_abs_lateral_error_m",
            "max_abs_heading_error_rad", "brake_ratio",
            "steering_reversals", "stop_violations"]
    y = 0.53
    ax.text(0.43, y + 0.045, "selected", color=COLORS["factual"],
            fontsize=9, fontweight="bold", ha="center")
    ax.text(0.68, y + 0.045, "foil", color=COLORS["foil"],
            fontsize=9, fontweight="bold", ha="center")
    for label, key in zip(labels, keys):
        ax.text(0.05, y, label, fontsize=9, va="center")
        ax.text(0.43, y, _value(factual.get(key)), fontsize=9,
                ha="center", color=COLORS["factual"])
        ax.text(0.68, y, _value(foil.get(key)), fontsize=9,
                ha="center", color=COLORS["foil"])
        y -= 0.055
    factual_seq = " -> ".join(factual.get("primitive_sequence", [])[:5])
    foil_seq = " -> ".join(foil.get("primitive_sequence", [])[:5])
    ax.text(0.02, 0.17, "Post-freeze M2 sequence (presentation only)",
            fontsize=9, fontweight="bold")
    ax.text(0.04, 0.12, "selected: " + factual_seq, fontsize=8,
            color=COLORS["factual"], wrap=True)
    ax.text(0.04, 0.07, "foil: " + foil_seq, fontsize=8,
            color=COLORS["foil"], wrap=True)
    ax.text(0.98, 0.02,
            "Cluster frozen before these M2 labels were opened",
            fontsize=7, color="#6B7280", ha="right")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((ROOT / "eddp_discovery_summary.json").read_text())
    atoms = _atoms()
    outputs = []
    for card in summary["cluster_cards"]:
        block = card["representative_block_id"]
        selected_atoms = sorted(
            (atom for atom in atoms if atom["block_id"] == block),
            key=lambda atom: atom["block_offset"],
        )
        if not selected_atoms:
            continue
        card = dict(card)
        card["_atoms"] = selected_atoms
        reports = [
            json.loads(Path(atom["paired_report_path"]).read_text())
            for atom in selected_atoms
        ]
        fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=120)
        fig.patch.set_facecolor("#FAFAF7")

        def update(frame):
            _draw(frame, ax, card, selected_atoms[frame], reports[frame])
            return (ax,)

        clip = animation.FuncAnimation(
            fig, update, frames=len(selected_atoms), interval=1300,
            repeat=True, blit=False,
        )
        path = OUT / ("C%02d_%s.gif" % (
            card["cluster_id"], card["candidate_name"].rsplit("_C", 1)[0]
        ))
        clip.save(path, writer=animation.PillowWriter(fps=1))
        plt.close(fig)
        outputs.append(str(path))
    manifest = {"kind": "data_only_explanation_clip", "outputs": outputs}
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
