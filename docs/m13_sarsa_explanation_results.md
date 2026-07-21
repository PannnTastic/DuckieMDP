# M13 — Hasil Explanation SARSA

## Status

M13 selesai dan **diterima**. SARSA sekarang dijelaskan dengan framework yang
sama seperti Q-learning:

1. decision explanation berbasis Q-table dan driving primitive;
2. counterfactual state/response curve;
3. COViz-inspired paired action-outcome rollout;
4. LEGIBLE-inspired metamorphic policy testing;
5. exact finite-state characterization dan safety-property checking;
6. rule extraction sebagai surrogate post-hoc;
7. matched behavioral comparison dengan Q-learning dan SAC.

Integrated Gradients dan critic probes tidak dipaksakan ke SARSA karena SARSA
adalah tabel, bukan neural actor/critic. Untuk policy tabular, pemeriksaan
langsung seluruh tabel memberi bukti yang lebih tepat.

## Policy yang benar-benar dijelaskan

| Field | Nilai |
|---|---|
| Solver | SARSA |
| Checkpoint | `artifacts/ablation/full_task/sarsa_teacher/q_table_best.npy` |
| SHA-256 | `0266ad6f6fdae71bf2dfb7c7121f66e038d16f75e214a931f8c9d50bc6ad3313` |
| Shape | `(5,5,3,3,4,2,5,7)` |
| Config full-task | `configs/small_loop_stop_duck_sarsa.yaml` |
| Training | teacher-guided SARSA |
| Explanation/evaluation | greedy, deterministic, teacher-free |
| Tie breaking | action-id terkecil |

Teacher hanya membantu memilih sebagian action saat training. Semua action pada
explanation dan evaluation berasal dari argmax tabel SARSA; teacher tidak aktif.

## Local decision dan outcome explanation

Tiga state kritis diambil dari rollout nyata. Untuk setiap state, simulator
direplay sampai branch point yang sama. Cabang factual mengeksekusi action
SARSA, cabang foil mengeksekusi satu action pembanding, lalu kedua cabang
kembali menggunakan policy SARSA yang sama.

| Skenario | Action SARSA → primitive | Foil → primitive | Q-margin | Return factual | Return foil |
|---|---|---|---:|---:|---:|
| Lane correction | `slow_left` → `LaneCorrectLeft` | `fast_straight` → `CruiseCurveLeft` | 11.425 | 4.014 | 3.899 |
| Stop sign | `brake` → `StopHold` | `slow_straight` → `DecelerateStop` | 22.143 | 18.054 | 18.007 |
| Pedestrian | `brake` → `YieldHold` | `slow_straight` → `YieldDecelerate` | 7.308 | 3.117 | 3.191 |

Semua branch invariants lulus: manifest sama, prefix sama, hanya action pertama
yang dipaksa, selected dan foil berbeda, dan teacher mati.

Q-margin adalah selisih nilai action terbaik terhadap action terbaik kedua.
Nilai ini bukan probabilitas dan bukan confidence yang terkalibrasi.

Pada contoh pedestrian, foil mempunyai sampled 30-step return sedikit lebih
tinggi. Ini bukan kontradiksi: Q SARSA memperkirakan expected discounted return
dari pengalaman training, sedangkan paired rollout adalah satu konsekuensi
simulator pada satu manifest. Temuan ini harus dilaporkan, bukan disembunyikan.

Artefak yang mudah dibaca:

- `runs/explanations/m13_sarsa/local/sarsa_lane_correction.txt`;
- `runs/explanations/m13_sarsa/local/sarsa_stop_hold.txt`;
- `runs/explanations/m13_sarsa/local/sarsa_pedestrian_yield.txt`.

Versi JSON menyimpan state, seluruh Q-values, branch provenance, reward profile,
physical outcomes, primitive sequence, dan invariants.

## Response curves dan state counterfactual

Enam sweep dibuat dari anchor rollout nyata:

- lateral offset `d`;
- heading error `phi`;
- curvature;
- stop distance;
- stop-satisfied flag;
- Duckie threat.

Hasilnya 32 query valid, 0 query ditolak, 16 action flip, dan 15 primitive
flip. Bentuk SARSA berupa fungsi tangga karena state masuk ke bin diskret.

Folder hasil:
`runs/explanations/m13_sarsa/analysis/response_curves/`.

## Metamorphic policy testing

Sebanyak 24 pasangan intervensi diuji:

- 21 PASS;
- 3 FAIL;
- seluruh source/target state lolos manifold validator.

Tiga FAIL berada pada relation lane-symmetry. Ini merupakan finding policy:
policy tidak sepenuhnya simetris saat tanda lane error dibalik. FAIL tersebut
bukan kegagalan pipeline.

Folder hasil:
`runs/explanations/m13_sarsa/analysis/metamorphic/`.

## Exact finite-state explanation

Seluruh 9.000 state index dievaluasi satu kali.

| Stratum | Jumlah state |
|---|---:|
| Representable | 9.000 |
| Semantically valid | 7.875 |
| Evaluation-reachable | 200 |
| Supported proxy | 141 |

Karena visit-count training historis tidak tersedia, `supported` memakai
evaluation reach count minimal tiga kunjungan. Itu adalah proxy dan dilabeli
demikian.

Safety property pada state supported:

| Property | Applicable | Violation |
|---|---:|---:|
| Duckie crossing-near tidak boleh fast | 10 | 0 |
| Stop-near dan belum satisfied tidak boleh fast | 4 | 0 |

Pada reachable stratum, pedestrian property memiliki 4 violation dari 19
applicable states. State ini tetap finding penting. Angka besar pada
unknown/unvisited cells tidak boleh disebut sebagai perilaku yang telah
dipelajari; banyak cell masih mempunyai nilai awal/tie.

Artefak policy map, one-bin flips, reach counts, dan violation table berada di:
`runs/explanations/m13_sarsa/analysis/exact/`.

## Rule extraction

Decision tree post-hoc mencapai fidelity 1,0 pada action dan primitive untuk
domain 9.000 state. Action tree mempunyai depth 15, 258 leaves, dan 515 nodes.

Tree tetap **bukan policy asli**. Policy utama, local explanation, dan
perbandingan selalu memakai Q-table SARSA asli. Fidelity 1,0 hanya menyatakan
tree berhasil merangkum mapping pada sampel finite yang diberikan.

Artefak:
`runs/explanations/m13_sarsa/analysis/rules/`.

## Perbandingan Q-learning, SARSA, dan SAC

Kontrak bersama:

- map `small_loop`;
- seed 20101, 20202, 20303, 20404, 20505;
- pose awal identik sampai `atol=1e-7`;
- `frame_skip=6`;
- 250 decision step/episode;
- teacher mati;
- SAC memakai deterministic actor mean;
- tidak ada surrogate yang mengontrol kendaraan.

| Metric | Q-learning | SARSA | SAC |
|---|---:|---:|---:|
| Mean return | 44.634 | 44.634 | 17.822 |
| Stop compliance | 100% | 100% | 100% |
| Undesirable primitive rate | 6,96% | 6,96% | 0% |
| Timeout | 5/5 | 5/5 | 5/5 |

Q-learning dan SARSA menghasilkan action yang sama pada seluruh 1.250 matched
decision steps. Namun checkpoint-nya tidak sama:

- maximum absolute Q-value difference: 20,196;
- greedy action sama pada 8.929/9.000 state (99,211%);
- greedy action berbeda pada 71 state.

Jadi kesamaan metrik rollout disebabkan lima trajectory bersama tidak memasuki
71 state yang membedakan action greedy kedua policy. Ini tidak membuktikan kedua
algoritma identik pada semua kondisi.

Perbandingan ini deskriptif, bukan solver-isolation experiment: histori
training dan target update Q-learning/SARSA berbeda, sementara SAC juga
mempunyai state/action representation berbeda.

## Laporan terpadu dan acceptance

Laporan primer:

- `runs/explanations/m13_sarsa/m13_sarsa_explanation_report.json`;
- `runs/explanations/m13_sarsa/m13_sarsa_local_explanation_index.csv`.

Report builder bersifat fail-closed dan lulus 18/18 gate. Acceptance yang
tracked ada di `docs/m13_sarsa_acceptance.json`.


Regresi penuh setelah integrasi:

```text
152 passed in 51.83s
```

## Reproduce


```bash
PYTHONWARNINGS=ignore PYTHONPATH=. .venv-sac/bin/python \
  scripts/run_m13_sarsa_local.py

PYTHONWARNINGS=ignore PYTHONPATH=. .venv-sac/bin/python \
  scripts/run_m13_sarsa_analysis.py

PYTHONWARNINGS=ignore PYTHONPATH=. .venv-sac/bin/python \
  scripts/run_m13_sarsa_comparison.py

PYTHONPATH=. .venv-sac/bin/python scripts/run_m13_sarsa_report.py
```

## Batas klaim

- Explanation menargetkan greedy evaluation policy, bukan epsilon/teacher
  behavior saat training.
- Paired rollout adalah simulator-based interventional outcome, bukan bukti
  kausal dunia nyata atau probabilitas.
- Q-margin bukan probabilitas.
- Supported state memakai evaluation support proxy.
- Metamorphic FAIL adalah finding policy.
- Lima seed cukup untuk integration validation, belum cukup untuk confidence
  interval thesis.
- Surrogate tree hanya ringkasan post-hoc.

