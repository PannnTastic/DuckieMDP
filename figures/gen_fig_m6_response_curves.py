"""Publication figure for M6 valid-manifold response curves."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runs" / "explanations" / "m6_response_curves"
OUTPUT_PDF = ROOT / "figures" / "fig_m6_response_curves.pdf"
OUTPUT_PNG = ROOT / "figures" / "fig_m6_response_curves.png"

BLUE = "#0072B2"
ORANGE = "#E69F00"
RED = "#D55E00"
GRAY = "#777777"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
        "lines.linewidth": 1.8,
        "lines.markersize": 4,
    }
)


def _load(scenario, solver, feature):
    path = DATA / f"{scenario}_{solver}_{feature}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _series(report, component):
    valid_x, valid_y, rejected_x = [], [], []
    for point in report["points"]:
        value = point["requested_value"]
        if not point["synthetic"]["validation"]["valid"]:
            rejected_x.append(float(value))
            continue
        valid_x.append(float(value))
        valid_y.append(float(point["decision"]["action"][component]))
    order = np.argsort(valid_x)
    return (
        np.asarray(valid_x)[order],
        np.asarray(valid_y)[order],
        np.asarray(rejected_x),
    )


def main():
    panels = (
        ("lane", "d", "Lateral offset $d$ (m)", "Lane-offset response"),
        ("stop", "stop_distance", "$d_{stop}$ (m)", "Stop-distance response"),
        (
            "duck",
            "duck_longitudinal",
            "Duckie longitudinal position (m)",
            "Pedestrian-position response",
        ),
    )
    components = (
        ("v_cmd", "$v_{cmd}$", (0.0, 0.43)),
        ("omega_cmd", "$\\omega_{cmd}$", (-1.6, 1.6)),
    )
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 6.5), constrained_layout=True)
    for row, (scenario, feature, xlabel, row_title) in enumerate(panels):
        q = _load(scenario, "q_learning", feature)
        sac = _load(scenario, "sac", feature)
        for col, (component, ylabel, ylim) in enumerate(components):
            ax = axes[row, col]
            qx, qy, qr = _series(q, component)
            sx, sy, sr = _series(sac, component)
            ax.step(
                qx,
                qy,
                where="mid",
                color=BLUE,
                marker="s",
                label="Q-learning",
                zorder=3,
            )
            ax.plot(
                sx,
                sy,
                color=ORANGE,
                marker="o",
                label="SAC",
                zorder=4,
            )
            rejected = np.unique(np.concatenate((qr, sr)))
            if rejected.size:
                y_marker = ylim[0] + 0.04 * (ylim[1] - ylim[0])
                ax.scatter(
                    rejected,
                    np.full_like(rejected, y_marker),
                    marker="x",
                    color=RED,
                    linewidth=1.2,
                    label="Rejected off-manifold" if row == 0 and col == 0 else None,
                    zorder=5,
                )
            ax.set_ylim(*ylim)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{row_title}: {ylabel}")
            if row == 0 and col == 0:
                ax.legend(loc="best")
    fig.suptitle(
        "Valid-manifold policy response curves",
        fontsize=11,
        fontweight="bold",
    )
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF)
    fig.savefig(OUTPUT_PNG, dpi=300)
    plt.close(fig)
    print(OUTPUT_PDF)
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
