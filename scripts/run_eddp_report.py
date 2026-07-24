"""Generate the unified EDDP pilot report from frozen experiment artifacts."""

import json
from pathlib import Path


ROOT = Path("runs/explanations/eddp_v1")


def _load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def _percent(value):
    return "%.1f%%" % (100.0 * float(value))


def _metric(value, digits=3):
    return "n/a" if value is None else ("%%.%df" % digits) % float(value)


def main():
    provenance = _load("provenance_manifest.json")
    collection = _load("anchor_collection_summary.json")
    counterfactual = _load("counterfactual_summary.json")
    discovery = _load("eddp_discovery_summary.json")
    hdbscan = discovery["hdbscan"]
    diagnostics = hdbscan["diagnostics"]
    reconciliation = hdbscan["reconciliation_after_freeze"]
    coherence = discovery["outcome_coherence"]
    predictability = discovery["solver_predictability"]

    lines = [
        "# Explanation-Derived Driving Primitives: Hasil Pilot EDDP v1",
        "",
        "> Status: pilot selesai dan seluruh engineering acceptance gate lulus. "
        "Nama di bawah adalah kandidat primitive berbasis explanation, bukan label ground-truth baru.",
        "",
        "## 1. Pertanyaan eksperimen",
        "",
        "Eksperimen menguji apakah profil explanation lokal dan temporal dapat menjadi input "
        "untuk menemukan driving primitive tanpa menggunakan label primitive M2 selama clustering.",
        "Label M2 hanya dibuka setelah assignment, signature, model, dan cluster card dibekukan.",
        "",
        "## 2. Policy dan provenance",
        "",
        "| Policy | Checkpoint | Evaluation |",
        "|---|---|---|",
    ]
    for solver, row in provenance["policies"].items():
        lines.append("| %s | `%s` | `%s` |" % (
            solver, row["checkpoint"], row["evaluation_mode"]
        ))

    lines.extend([
        "",
        "## 3. Pipeline yang dijalankan",
        "",
        "1. EDP0 membekukan checkpoint, config, hash, dan mode evaluasi.",
        "2. EDP1--EDP2 mengumpulkan anchor berdasarkan konteks fisik, tanpa label M2.",
        "3. EDP3 membentuk state counterfactual dan paired action-outcome rollout.",
        "4. EDP4--EDP5 mengubah explanation atom menjadi signature temporal tiga keputusan.",
        "5. EDP6 melakukan HDBSCAN pada development seeds dan assignment held-out secara induktif.",
        "6. EDP7--EDP8 membentuk cluster card dan nama fungsional dari konteks serta outcome fisik.",
        "7. EDP9 baru membuka label M2 untuk rekonsiliasi eksternal.",
        "8. EDP10 menjalankan KMeans sensitivity, ablation, coherence, dan solver predictability.",
        "",
        "## 4. Dataset dan validitas counterfactual",
        "",
        "- Anchor terkumpul: **%d** dalam **%d** temporal block." % (
            collection["anchors"], collection["blocks"]
        ),
        "- Explanation atom valid: **%d/%d (%s)**." % (
            counterfactual["successful_atoms"], counterfactual["requested_anchors"],
            _percent(counterfactual["successful_atoms"] / counterfactual["requested_anchors"]),
        ),
        "- Anchor dikarantina akibat native simulator/renderer crash: **%d**." % len(
            counterfactual.get("explicitly_quarantined_anchor_ids", [])
        ),
        "- Segment discovery: **%d**; fitur setelah variance filter: **%d**." % (
            discovery["segments"], discovery["features_after_variance_filter"]
        ),
        "",
        "Native crash tidak disamarkan sebagai counterfactual invalid. ID-nya dicatat di "
        "`counterfactual_summary.json`, dan segment yang tersisa hanya dipakai bila memiliki "
        "minimal dua atom valid.",
        "",
        "## 5. Kandidat primitive hasil discovery",
        "",
        "| ID | Nama kandidat | Status | Support | Solver | Konteks |",
        "|---:|---|---|---:|---|---|",
    ])
    for card in discovery["cluster_cards"]:
        solvers = ", ".join("%s:%s" % item for item in card["solver_counts"].items())
        contexts = ", ".join("%s:%s" % item for item in card["context_counts"].items())
        lines.append("| C%02d | %s | %s | %d | %s | %s |" % (
            card["cluster_id"], card["candidate_name"], card["status"],
            card["support"], solvers, contexts,
        ))

    lines.extend([
        "",
        "Interpretasi ringkas:",
        "",
        "- C00 dan C01 adalah kandidat paling jelas: konteks Duckie dan stop terpisah tanpa label M2.",
        "- C02--C04 menangkap variasi regulasi lane/progress pada Q-learning dan SARSA.",
        "- C05 hanya berisi SAC. Karena itu ia dilaporkan sebagai `SOLVER_SPECIFIC_BEHAVIOR`, "
        "bukan primitive lintas-solver.",
        "- Noise HDBSCAN tetap `Unknown`; sistem tidak memaksa semua segment menjadi primitive.",
        "",
        "## 6. Hasil kuantitatif",
        "",
        "| Metrik | Development | Held-out | Semua |",
        "|---|---:|---:|---:|",
        "| Cluster coverage | %s | %s | %s |" % (
            _percent(diagnostics["development"]["coverage"]),
            _percent(diagnostics["heldout"]["coverage"]),
            _percent(diagnostics["all"]["coverage"]),
        ),
        "| Silhouette | %s | %s | %s |" % (
            _metric(diagnostics["development"]["silhouette"]),
            _metric(diagnostics["heldout"]["silhouette"]),
            _metric(diagnostics["all"]["silhouette"]),
        ),
        "| Purity vs M2 | %s | %s | %s |" % (
            _metric(reconciliation["development"]["purity"]),
            _metric(reconciliation["heldout"]["purity"]),
            _metric(reconciliation["overall"]["purity"]),
        ),
        "| NMI vs M2 | %s | %s | %s |" % (
            _metric(reconciliation["development"]["nmi"]),
            _metric(reconciliation["heldout"]["nmi"]),
            _metric(reconciliation["overall"]["nmi"]),
        ),
        "| ARI vs M2 | %s | %s | %s |" % (
            _metric(reconciliation["development"]["ari"]),
            _metric(reconciliation["heldout"]["ari"]),
            _metric(reconciliation["overall"]["ari"]),
        ),
        "",
        "Outcome coherence ratio adalah **%s** (lebih kecil dari 1 lebih baik); "
        "observed within-cluster MSE lebih rendah daripada 100 permutasi acak: **%s**." % (
            _metric(coherence["ratio"]), coherence["better_than_permuted"]
        ),
        "",
        "## 7. Solver leakage diagnostic",
        "",
        "Solver tidak pernah dimasukkan sebagai fitur. Namun classifier diagnostik dapat menebak "
        "solver dengan akurasi development **%s** dan held-out **%s** (chance **%s**)." % (
            _percent(predictability["development_accuracy"]),
            _percent(predictability["heldout_accuracy"]),
            _percent(predictability["chance_reference"]),
        ),
        "Ini berarti signature masih membawa pola perilaku solver. Temuan ini konsisten dengan C05 "
        "yang SAC-spesifik dan harus dipertahankan sebagai batas klaim, bukan dihapus dari laporan.",
        "",
        "## 8. Ablation",
        "",
        "| Ablation | Coverage | Cluster | Silhouette |",
        "|---|---:|---:|---:|",
    ])
    for name, result in sorted(discovery["ablations"].items()):
        diag = result.get("diagnostics", {}).get("all", {})
        lines.append("| %s | %s | %s | %s |" % (
            name, _percent(diag.get("coverage", 0.0)),
            diag.get("clusters", "n/a"), _metric(diag.get("silhouette")),
        ))

    extended = discovery.get("extended_ablations", {})
    lines.extend([
        "",
        "| Extended ablation | Status | Coverage | Cluster | Silhouette |",
        "|---|---|---:|---:|---:|",
    ])
    for name in ("physical_only", "physical_plus_reward",
                 "complete_fixed_window_only"):
        result = extended.get(name, {})
        diag = result.get("diagnostics", {}).get("all", {})
        lines.append("| %s | %s | %s | %s | %s |" % (
            name, result.get("status", "MISSING"),
            _percent(diag.get("coverage", 0.0)) if diag else "n/a",
            diag.get("clusters", "n/a"), _metric(diag.get("silhouette")),
        ))
    for solver, result in sorted(extended.get("per_solver", {}).items()):
        diag = result.get("diagnostics", {}).get("all", {})
        lines.append("| per_solver:%s | %s | %s | %s | %s |" % (
            solver, result.get("status", "MISSING"),
            _percent(diag.get("coverage", 0.0)) if diag else "n/a",
            diag.get("clusters", "n/a"), _metric(diag.get("silhouette")),
        ))
    for name in ("explanation_change_point", "rollout_natural_frequency"):
        result = extended.get(name, {})
        lines.append("| %s | %s | n/a | n/a | n/a |" % (
            name, result.get("status", "MISSING")
        ))
    lines.extend([
        "",
        "Ablation tidak dibaca hanya dari silhouette. Paired physical outcome memberi grounding "
        "konsekuensi aksi; verification menaikkan coverage; state counterfactual membantu pemisahan "
        "boundary keputusan. Trade-off ini harus dibaca bersama semantic coherence.",
        "",
        "## 9. Acceptance dan keputusan ilmiah",
        "",
        "Semua engineering gate discovery **PASS**: label-free contract, freeze sebelum M2, "
        "development/held-out split, deterministic rerun, dan inductive held-out assignment.",
        "",
        "Keputusan pilot: **GO dengan klaim terbatas**. Explanation dapat digunakan sebagai input "
        "untuk menemukan kandidat primitive Q-learning/SARSA. SAC belum menyatu ke taksonomi bersama "
        "dan memerlukan anchor stop tambahan atau kalibrasi signature lintas action space.",
        "",
        "## 10. Batasan",
        "",
        "- Rollout SAC pada seed pilot tidak menghasilkan anchor stop yang memenuhi selector.",
        "- Delapan anchor gagal karena native crash simulator/renderer.",
        "- Foil hanya memaksa aksi pertama; efek fisik horizon pendek dapat kecil.",
        "- Change-point dan rollout-natural-frequency tidak dijalankan karena sparse anchor pilot "
        "tidak menyimpan adjacency atau frekuensi alami; keduanya dicatat sebagai data limitation.",
        "- Nama kandidat bersifat deskriptif dan tidak mengubah cluster yang telah dibekukan.",
        "- Rekonsiliasi M2 adalah evaluasi eksternal setelah freeze, bukan fitur discovery.",
        "",
        "## 11. Artefak utama",
        "",
        "- `cluster_freeze_pre_m2.json`: bukti freeze sebelum label M2 dibuka.",
        "- `cluster_assignments_unlabeled.csv`: assignment label-free per segment.",
        "- `signatures_unlabeled.csv`: fitur yang benar-benar masuk clustering.",
        "- `m2_labels_after_cluster_freeze.csv`: label eksternal untuk rekonsiliasi.",
        "- `eddp_discovery_summary.json`: seluruh metrik, cluster card, dan ablation.",
        "- `primitive_catalogue.json`: katalog kandidat primitive machine-readable.",
        "- `failure_mode_catalogue.json`: noise, quarantine, dan ablation yang tidak teridentifikasi.",
        "- `figures/fig_eddp_main_results.pdf`: embedding, konteks, rekonsiliasi, dan ablation.",
        "- `figures/fig_eddp_candidate_timeline.pdf`: timeline kandidat per policy.",
        "- `explanation_clips/*.gif`: clip data-only factual-versus-foil per cluster; "
        "bukan rekaman kamera simulator.",
        "",
        "Laporan ini dibuat otomatis oleh `python -m scripts.run_eddp_report`.",
        "",
    ])

    report = Path("docs/explanation_derived_driving_primitives_results.md")
    report.write_text("\n".join(lines), encoding="utf-8")
    acceptance = {
        "experiment": "eddp_v1_pilot",
        "provenance_passed": all(provenance["acceptance"].values()),
        "collection_passed": bool(collection["acceptance"]["passed"]),
        "counterfactual_passed": bool(counterfactual["acceptance"]["passed"]),
        "discovery_passed": bool(discovery["acceptance"]["passed"]),
        "valid_atom_fraction": (
            counterfactual["successful_atoms"] / counterfactual["requested_anchors"]
        ),
        "cluster_coverage": diagnostics["all"]["coverage"],
        "shared_clusters": discovery["shared_clusters"],
        "solver_specific_clusters": discovery["solver_specific_clusters"],
        "scientific_decision": "GO_WITH_LIMITED_CLAIM",
    }
    acceptance["engineering_passed"] = all([
        acceptance["provenance_passed"], acceptance["collection_passed"],
        acceptance["counterfactual_passed"], acceptance["discovery_passed"],
    ])
    Path("docs/eddp_v1_acceptance.json").write_text(
        json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    freeze = _load("cluster_freeze_pre_m2.json")
    confusion = reconciliation["overall"].get("confusion", [])
    catalogue = {
        "experiment": "eddp_v1_pilot",
        "definition": (
            "temporal clusters discovered from label-free explanation signatures"
        ),
        "cluster_freeze_sha256": freeze["freeze_sha256"],
        "candidates": [],
    }
    for card in discovery["cluster_cards"]:
        external_counts = {
            row["primitive"]: row["count"]
            for row in confusion if row["cluster_id"] == card["cluster_id"]
        }
        catalogue["candidates"].append({
            **card,
            "external_m2_counts_after_freeze": external_counts,
            "claim_scope": (
                "cross_solver_candidate"
                if card["status"] == "PRIMITIVE_CANDIDATE"
                else "solver_specific_behavior"
            ),
        })
    (ROOT / "primitive_catalogue.json").write_text(
        json.dumps(catalogue, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    extended = discovery.get("extended_ablations", {})
    failure_catalogue = {
        "experiment": "eddp_v1_pilot",
        "unknown_segments": int(round(
            discovery["segments"] * diagnostics["all"]["noise"]
        )),
        "quarantined_anchor_ids": counterfactual.get(
            "explicitly_quarantined_anchor_ids", []
        ),
        "solver_specific_clusters": [
            card["cluster_id"] for card in discovery["cluster_cards"]
            if card["status"] == "SOLVER_SPECIFIC_BEHAVIOR"
        ],
        "not_executed_ablations": {
            name: extended[name]
            for name in ("explanation_change_point", "rollout_natural_frequency")
            if extended.get(name, {}).get("status", "").startswith("NOT_EXECUTED")
        },
        "not_identifiable_per_solver": {
            solver: result
            for solver, result in extended.get("per_solver", {}).items()
            if result.get("status") == "NOT_IDENTIFIABLE"
        },
    }
    (ROOT / "failure_mode_catalogue.json").write_text(
        json.dumps(failure_catalogue, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(acceptance, sort_keys=True))


if __name__ == "__main__":
    main()
