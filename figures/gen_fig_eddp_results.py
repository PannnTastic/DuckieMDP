#!/usr/bin/env python3
"""Generate publication-quality EDDP result figures from frozen artifacts."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path("runs/explanations/eddp_v1")
OUT = ROOT / "figures"
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
UNKNOWN = "#B0BEC5"


def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
    })


def _read_csv(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / (stem + ".pdf"))
    fig.savefig(OUT / (stem + ".png"), dpi=300)
    plt.close(fig)


def main():
    _style()
    summary = json.loads((ROOT / "eddp_discovery_summary.json").read_text())
    assignments = _read_csv("cluster_assignments_unlabeled.csv")
    signatures = _read_csv("signatures_unlabeled.csv")
    metadata = {
        "segment_index", "block_id", "solver", "seed", "episode_id",
        "selection_context", "start_step", "end_step", "length",
        "expected_length", "missing_offsets", "complete_window", "segment_index",
    }
    feature_names = [name for name in signatures[0] if name not in metadata]
    features = np.asarray([
        [float(row[name]) for name in feature_names] for row in signatures
    ])
    embedding = PCA(n_components=2, random_state=0).fit_transform(
        StandardScaler().fit_transform(features)
    )
    labels = np.asarray([int(row["hdbscan_cluster"]) for row in assignments])
    splits = np.asarray([row["split"] for row in assignments])
    solvers = np.asarray([row["solver"] for row in assignments])

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6))
    ax = axes[0, 0]
    for cluster in sorted(set(labels)):
        mask = labels == cluster
        color = UNKNOWN if cluster < 0 else COLORS[cluster % len(COLORS)]
        label = "Unknown" if cluster < 0 else "C%02d" % cluster
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=24,
                   color=color, alpha=0.82, label=label, edgecolor="white", linewidth=0.3)
    heldout = splits == "heldout"
    ax.scatter(embedding[heldout, 0], embedding[heldout, 1], s=48,
               facecolors="none", edgecolors="#222222", linewidth=0.7,
               label="held-out")
    ax.set_title("(a) Label-free signature embedding")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(ncol=2, loc="best")

    ax = axes[0, 1]
    contexts = ["duck", "stop", "lane", "nominal"]
    cluster_ids = [card["cluster_id"] for card in summary["cluster_cards"]]
    bottom = np.zeros(len(cluster_ids))
    context_colors = ["#E69F00", "#D55E00", "#0072B2", "#009E73"]
    for context, color in zip(contexts, context_colors):
        values = np.asarray([
            card["context_counts"].get(context, 0)
            for card in summary["cluster_cards"]
        ])
        ax.bar(cluster_ids, values, bottom=bottom, color=color,
               edgecolor="white", linewidth=0.4, label=context)
        bottom += values
    ax.set_title("(b) Physical context composition")
    ax.set_xlabel("Candidate cluster")
    ax.set_ylabel("Temporal segments")
    ax.set_xticks(cluster_ids, ["C%02d" % value for value in cluster_ids])
    ax.legend(ncol=2)

    ax = axes[1, 0]
    confusion = summary["hdbscan"]["reconciliation_after_freeze"]["overall"]["confusion"]
    primitive_names = sorted({row["primitive"] for row in confusion})
    matrix = np.zeros((len(cluster_ids), len(primitive_names)), dtype=int)
    for row in confusion:
        matrix[cluster_ids.index(row["cluster_id"]), primitive_names.index(row["primitive"])] = row["count"]
    image = ax.imshow(matrix, cmap="YlOrBr", aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j]:
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7)
    ax.set_title("(c) Post-freeze reconciliation with M2")
    ax.set_yticks(range(len(cluster_ids)), ["C%02d" % value for value in cluster_ids])
    ax.set_xticks(range(len(primitive_names)), primitive_names, rotation=50, ha="right")
    ax.set_ylabel("EDDP candidate")
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)

    ax = axes[1, 1]
    variants = {"main": summary["hdbscan"]["diagnostics"]["all"]}
    for name, result in summary["ablations"].items():
        if "diagnostics" in result:
            variants[name.replace("without_", "-")] = result["diagnostics"]["all"]
    labels_ablation = list(variants)
    coverage = [variants[name]["coverage"] for name in labels_ablation]
    silhouette = [variants[name]["silhouette"] or 0.0 for name in labels_ablation]
    x = np.arange(len(labels_ablation))
    width = 0.36
    ax.bar(x - width / 2, coverage, width, label="coverage", color="#0072B2")
    ax.bar(x + width / 2, silhouette, width, label="silhouette", color="#E69F00")
    ax.set_title("(d) Explanation-component ablation")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x, labels_ablation, rotation=28, ha="right")
    ax.legend()

    fig.tight_layout()
    _save(fig, "fig_eddp_main_results")

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    marker = {"q_learning": "o", "sarsa": "s", "sac": "^"}
    for ax, solver in zip(axes, ("q_learning", "sarsa", "sac")):
        candidates = [row for row in assignments if row["solver"] == solver]
        selected_seed = max(Counter(row["seed"] for row in candidates),
                            key=lambda seed: sum(row["seed"] == seed for row in candidates))
        candidates = sorted(
            (row for row in candidates if row["seed"] == selected_seed),
            key=lambda row: int(row["start_step"]),
        )
        for row in candidates:
            cluster = int(row["hdbscan_cluster"])
            color = UNKNOWN if cluster < 0 else COLORS[cluster % len(COLORS)]
            ax.scatter(int(row["start_step"]), cluster, s=42, color=color,
                       marker=marker[solver], edgecolor="white", linewidth=0.4)
        ax.set_title(solver.replace("_", " ").upper())
        ax.set_xlabel("Decision step")
        ax.set_yticks([-1] + cluster_ids,
                      ["Unknown"] + ["C%02d" % value for value in cluster_ids])
    axes[0].set_ylabel("Explanation-derived candidate")
    fig.suptitle("Temporal candidate timeline on representative held-out rollouts",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig_eddp_candidate_timeline")

    print(json.dumps({
        "main_figure": str(OUT / "fig_eddp_main_results.pdf"),
        "timeline": str(OUT / "fig_eddp_candidate_timeline.pdf"),
        "segments": len(assignments),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
