# Explanation-Derived Driving Primitives: Hasil Pilot EDDP v1

> Status: pilot selesai dan seluruh engineering acceptance gate lulus. Nama di bawah adalah kandidat primitive berbasis explanation, bukan label ground-truth baru.

## 1. Pertanyaan eksperimen

Eksperimen menguji apakah profil explanation lokal dan temporal dapat menjadi input untuk menemukan driving primitive tanpa menggunakan label primitive M2 selama clustering.
Label M2 hanya dibuka setelah assignment, signature, model, dan cluster card dibekukan.

## 2. Policy dan provenance

| Policy | Checkpoint | Evaluation |
|---|---|---|
| q_learning | `artifacts/ablation/full_task/q_learning_teacher/q_table_best.npy` | `greedy_teacher_free` |
| sac | `artifacts/sac/full_repeat_duck_5min/sac_best.pt` | `deterministic_actor_mean` |
| sarsa | `artifacts/ablation/full_task/sarsa_teacher/q_table_best.npy` | `greedy_teacher_free` |

## 3. Pipeline yang dijalankan

1. EDP0 membekukan checkpoint, config, hash, dan mode evaluasi.
2. EDP1--EDP2 mengumpulkan anchor berdasarkan konteks fisik, tanpa label M2.
3. EDP3 membentuk state counterfactual dan paired action-outcome rollout.
4. EDP4--EDP5 mengubah explanation atom menjadi signature temporal tiga keputusan.
5. EDP6 melakukan HDBSCAN pada development seeds dan assignment held-out secara induktif.
6. EDP7--EDP8 membentuk cluster card dan nama fungsional dari konteks serta outcome fisik.
7. EDP9 baru membuka label M2 untuk rekonsiliasi eksternal.
8. EDP10 menjalankan KMeans sensitivity, ablation, coherence, dan solver predictability.

## 4. Dataset dan validitas counterfactual

- Anchor terkumpul: **303** dalam **101** temporal block.
- Explanation atom valid: **295/303 (97.4%)**.
- Anchor dikarantina akibat native simulator/renderer crash: **8**.
- Segment discovery: **101**; fitur setelah variance filter: **188**.

Native crash tidak disamarkan sebagai counterfactual invalid. ID-nya dicatat di `counterfactual_summary.json`, dan segment yang tersisa hanya dipakai bila memiliki minimal dua atom valid.

## 5. Kandidat primitive hasil discovery

| ID | Nama kandidat | Status | Support | Solver | Konteks |
|---:|---|---|---:|---|---|
| C00 | PedestrianRiskModulatedControl_C00 | PRIMITIVE_CANDIDATE | 9 | q_learning:5, sarsa:4 | duck:9 |
| C01 | StopDistanceModulatedControl_C01 | PRIMITIVE_CANDIDATE | 12 | q_learning:6, sarsa:6 | stop:12 |
| C02 | ConservativeLaneRegulation_C02 | PRIMITIVE_CANDIDATE | 9 | q_learning:4, sarsa:5 | lane:4, nominal:5 |
| C03 | ProgressPreservingRegulation_C03 | PRIMITIVE_CANDIDATE | 8 | q_learning:4, sarsa:4 | lane:4, nominal:4 |
| C04 | ContextAwareProceed_C04 | PRIMITIVE_CANDIDATE | 19 | q_learning:10, sarsa:9 | duck:10, lane:7, nominal:2 |
| C05 | ContinuousLowSpeedRegulation_C05 | SOLVER_SPECIFIC_BEHAVIOR | 20 | sac:20 | duck:5, lane:10, nominal:5 |

Interpretasi ringkas:

- C00 dan C01 adalah kandidat paling jelas: konteks Duckie dan stop terpisah tanpa label M2.
- C02--C04 menangkap variasi regulasi lane/progress pada Q-learning dan SARSA.
- C05 hanya berisi SAC. Karena itu ia dilaporkan sebagai `SOLVER_SPECIFIC_BEHAVIOR`, bukan primitive lintas-solver.
- Noise HDBSCAN tetap `Unknown`; sistem tidak memaksa semua segment menjadi primitive.

## 6. Hasil kuantitatif

| Metrik | Development | Held-out | Semua |
|---|---:|---:|---:|
| Cluster coverage | 76.3% | 76.2% | 76.2% |
| Silhouette | 0.297 | 0.314 | 0.288 |
| Purity vs M2 | 0.644 | 0.719 | 0.662 |
| NMI vs M2 | 0.688 | 0.780 | 0.707 |
| ARI vs M2 | 0.421 | 0.544 | 0.490 |

Outcome coherence ratio adalah **0.519** (lebih kecil dari 1 lebih baik); observed within-cluster MSE lebih rendah daripada 100 permutasi acak: **True**.

## 7. Solver leakage diagnostic

Solver tidak pernah dimasukkan sebagai fitur. Namun classifier diagnostik dapat menebak solver dengan akurasi development **96.6%** dan held-out **59.5%** (chance **33.3%**).
Ini berarti signature masih membawa pola perilaku solver. Temuan ini konsisten dengan C05 yang SAC-spesifik dan harus dipertahankan sebagai batas klaim, bukan dihapus dari laporan.

## 8. Ablation

| Ablation | Coverage | Cluster | Silhouette |
|---|---:|---:|---:|
| without_paired_physical_outcome | 66.3% | 8 | 0.492 |
| without_state_counterfactual | 72.3% | 6 | 0.256 |
| without_verification | 53.5% | 5 | 0.342 |

| Extended ablation | Status | Coverage | Cluster | Silhouette |
|---|---|---:|---:|---:|
| physical_only | COMPLETED | 29.7% | 2 | 0.366 |
| physical_plus_reward | COMPLETED | 24.8% | 2 | 0.407 |
| complete_fixed_window_only | COMPLETED | 75.5% | 6 | 0.281 |
| per_solver:q_learning | COMPLETED | 57.9% | 2 | 0.180 |
| per_solver:sac | NOT_IDENTIFIABLE | n/a | n/a | n/a |
| per_solver:sarsa | NOT_IDENTIFIABLE | n/a | n/a | n/a |
| explanation_change_point | NOT_EXECUTED_DATASET_LIMITATION | n/a | n/a | n/a |
| rollout_natural_frequency | NOT_EXECUTED_DATASET_LIMITATION | n/a | n/a | n/a |

Ablation tidak dibaca hanya dari silhouette. Paired physical outcome memberi grounding konsekuensi aksi; verification menaikkan coverage; state counterfactual membantu pemisahan boundary keputusan. Trade-off ini harus dibaca bersama semantic coherence.

## 9. Acceptance dan keputusan ilmiah

Semua engineering gate discovery **PASS**: label-free contract, freeze sebelum M2, development/held-out split, deterministic rerun, dan inductive held-out assignment.

Keputusan pilot: **GO dengan klaim terbatas**. Explanation dapat digunakan sebagai input untuk menemukan kandidat primitive Q-learning/SARSA. SAC belum menyatu ke taksonomi bersama dan memerlukan anchor stop tambahan atau kalibrasi signature lintas action space.

## 10. Batasan

- Rollout SAC pada seed pilot tidak menghasilkan anchor stop yang memenuhi selector.
- Delapan anchor gagal karena native crash simulator/renderer.
- Foil hanya memaksa aksi pertama; efek fisik horizon pendek dapat kecil.
- Change-point dan rollout-natural-frequency tidak dijalankan karena sparse anchor pilot tidak menyimpan adjacency atau frekuensi alami; keduanya dicatat sebagai data limitation.
- Nama kandidat bersifat deskriptif dan tidak mengubah cluster yang telah dibekukan.
- Rekonsiliasi M2 adalah evaluasi eksternal setelah freeze, bukan fitur discovery.

## 11. Artefak utama

- `cluster_freeze_pre_m2.json`: bukti freeze sebelum label M2 dibuka.
- `cluster_assignments_unlabeled.csv`: assignment label-free per segment.
- `signatures_unlabeled.csv`: fitur yang benar-benar masuk clustering.
- `m2_labels_after_cluster_freeze.csv`: label eksternal untuk rekonsiliasi.
- `eddp_discovery_summary.json`: seluruh metrik, cluster card, dan ablation.
- `primitive_catalogue.json`: katalog kandidat primitive machine-readable.
- `failure_mode_catalogue.json`: noise, quarantine, dan ablation yang tidak teridentifikasi.
- `figures/fig_eddp_main_results.pdf`: embedding, konteks, rekonsiliasi, dan ablation.
- `figures/fig_eddp_candidate_timeline.pdf`: timeline kandidat per policy.
- `explanation_clips/*.gif`: clip data-only factual-versus-foil per cluster; bukan rekaman kamera simulator.

Laporan ini dibuat otomatis oleh `python -m scripts.run_eddp_report`.
