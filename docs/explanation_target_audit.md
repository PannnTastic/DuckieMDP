# Audit Target Policy dan Config Explanation

Dokumen ini membekukan identitas policy dan config yang benar-benar dijelaskan.
Nama file saja tidak cukup: checkpoint diperiksa dengan SHA-256 agar file lain
yang kebetulan bernama sama tidak dapat masuk diam-diam ke laporan.

## Target policy kanonis

| Solver | Checkpoint | SHA-256 | Mode explanation/evaluation |
|---|---|---|---|
| Q-learning | `artifacts/ablation/full_task/q_learning_teacher/q_table_best.npy` | `59929f4e2c1968d274e9d3aa6c83ccb2cd7c915c632093cfaba6b544340906dd` | greedy, teacher-free, tie dipilih ke action-id terkecil |
| SARSA | `artifacts/ablation/full_task/sarsa_teacher/q_table_best.npy` | `0266ad6f6fdae71bf2dfb7c7121f66e038d16f75e214a931f8c9d50bc6ad3313` | greedy, teacher-free, tie dipilih ke action-id terkecil |
| SAC | `artifacts/sac/full_repeat_duck_5min/sac_best.pt` | `0b01447edd85e539de57f9a304fc287d26d5d7c4e73a5a51cd493fea2f4c4f2b` | deterministic actor mean, teacher-free, tanpa sampling |

Kontrak tambahan:

- shape Q-table wajib `(5,5,3,3,4,2,5,7)` dan action yang diizinkan 0--6;
- checkpoint SAC wajib menerima observation 15-D tanpa ekspansi;
- surrogate tree M10 hanya merangkum policy dan tidak pernah menghasilkan
  action untuk explanation utama atau perbandingan M12;
- teacher Q-learning/SARSA digunakan saat sebagian training, tetapi selalu
  mati saat explanation dan evaluation.

## Nama dan peran config

| Config | Peran yang benar | Bukan untuk |
|---|---|---|
| `configs/small_loop_stop_duck_q.yaml` | task full stop-sign + satu crossing Duckie untuk Q explanation/evaluation | bukan bukti config training identik byte-per-byte |
| `configs/small_loop_lane_q_no_teacher.yaml` | lingkungan lane-only untuk anchor/case lane Q | bukan checkpoint lane-only; policy tetap Q full-task kanonis |
| `configs/small_loop_stop_duck_sarsa.yaml` | task full stop-sign + Duckie untuk SARSA explanation/evaluation | bukan policy Q-learning walaupun shape tabel sama |
| `configs/small_loop_lane_sarsa.yaml` | lingkungan lane-only untuk anchor SARSA | bukan checkpoint lane-only |
| `artifacts/sac/full_repeat_duck_5min/config.yaml` | config kanonis SAC yang menghasilkan checkpoint dan dipakai untuk explanation full-task | bukan repeated-crossing deployment |
| `configs/sac_lane.yaml` | lingkungan lane-only untuk anchor/case lane SAC | bukan checkpoint SAC lane-only; actor tetap SAC full-task kanonis |
| `artifacts/sac/full_repeat_duck_5min/config_repeat_duck.yaml` | deployment/audit video dengan Duckie menyeberang berulang | bukan config training dan bukan task utama explanation |
| `runs/explanations/m12_policy_comparison/shared_comparison_config.yaml` | manifest evaluasi bersama yang dibuat M12 | bukan config training solver |

Hash config yang dibekukan:

| Config | SHA-256 |
|---|---|
| `configs/small_loop_stop_duck_q.yaml` | `5f92d9e66c1ce3e353725add2a58c95b77ca5e2832dd1771b1063afbc3604dab` |
| `configs/small_loop_lane_q_no_teacher.yaml` | `cd83a466ca3fdab25a5e2a368e929fd2fbe30f7bb66248c40d2a397afdb08b0d` |
| `configs/small_loop_stop_duck_sarsa.yaml` | `594fe045f144b79c9806140801e9f27813b82c69711778900dc5720f15384c1c` |
| `configs/small_loop_lane_sarsa.yaml` | `14991a3dacd5706e6b28dda66a7822035ba8759ebb1ca94e01027b3908f5e8e3` |
| `artifacts/sac/full_repeat_duck_5min/config.yaml` | `3cc91129f2793fd624e7cd1600dc6636a57cb70e13f97e1ce0514f559fdb62b2` |
| `configs/sac_lane.yaml` | `832f2b026dbaa1b3fe943093e0624c6a3c2cdf6e531d7c570a5b158d106037a0` |
| `artifacts/sac/full_repeat_duck_5min/config_repeat_duck.yaml` | `d7fb73d0b05e48ec71063d790c7c3730f459779587cea6bae6dfbf2a1621dc75` |

## Matriks M1--M13

| Tahap | Policy yang digunakan | Config/sumber state |
|---|---|---|
| M1 | adapter Q dan SAC kanonis | kontrak schema; tidak menjalankan task |
| M2 | tidak membutuhkan checkpoint untuk membekukan leksikon | aturan primitive kanonis |
| M3 | kedua adapter kanonis saat merekam rollout | config sesuai skenario |
| M4 | policy tidak menjadi objek klaim utama | lane SAC, stop SAC, full SAC satu crossing, serta repeated deployment hanya untuk test replay |
| M5 lane Q | Q full-task kanonis | `small_loop_lane_q_no_teacher.yaml` |
| M5 stop/pedestrian Q | Q full-task kanonis | `small_loop_stop_duck_q.yaml` |
| M5 lane SAC | SAC full-task kanonis | `sac_lane.yaml` |
| M5 stop/pedestrian SAC | SAC full-task kanonis | config SAC kanonis |
| M6 | Q dan SAC kanonis | Q full, SAC full, dan SAC lane sebagai anchor environment |
| M7 | Q dan SAC kanonis | anchor valid M6 + action config Q full |
| M8 | Q kanonis | Q full; seluruh 9.000 state direpresentasikan |
| M9 | SAC kanonis | config SAC kanonis dan development rollout nyata |
| M10 | Q dan SAC kanonis | Q full + SAC full; tree hanya surrogate post-hoc |
| M11 | deferred/opsional | tidak ada policy baru |
| M12 | Q dan SAC kanonis | shared comparison manifest dengan seed, pose, frame-skip, dan horizon sama |
| M13 | SARSA full-task kanonis; Q dan SAC hanya untuk matched comparison | config SARSA lane/full + shared comparison manifest M12 |

## Perbedaan config yang disengaja

Checkpoint Q dilatih dengan artifact training miliknya, sedangkan explanation
full-task memakai `small_loop_stop_duck_q.yaml` yang membatasi kewajiban menjadi
satu crossing per episode dan menambahkan field evaluasi/repackaging. Karena itu
hasil ini dilabeli sebagai explanation terhadap policy terlatih pada task
evaluasi kanonis, bukan klaim bahwa kedua YAML identik.

M12 membuat config bersama dari config SAC, lalu mengubah:

- horizon dari 9.000 menjadi 1.500 physics tick;
- crossing menjadi satu per episode;
- `straight_steer_penalty` menjadi 0 karena wrapper Q lama tidak menyediakan
  `action_omega` kontinu.

Perubahan tersebut hanya untuk evaluasi matched. Checkpoint, action policy, dan
bobot kedua solver tidak diubah. Karena state/action/training regime kedua solver
memang berbeda, M12 adalah perbandingan perilaku deskriptif, bukan eksperimen
isolasi solver murni.

## Temuan audit dan koreksi

Audit menemukan `q_lane_correction.json` lama memakai checkpoint Q lane-only,
sementara explanation stop, pedestrian, dan M6--M12 memakai Q full-task.
Kasus lane kemudian dibuat ulang memakai checkpoint Q full-task kanonis; config
lane-only kini hanya menentukan lingkungan skenario.

`src/explainability/explanation_report.py` sekarang fail-closed bila:

- salah satu dari enam kasus lokal mempunyai path atau hash checkpoint berbeda;
- M6, M7, M8, M9, atau M10 memakai checkpoint non-kanonis;
- teacher aktif, branch tidak valid, atau surrogate dipakai sebagai policy M12.

Artefak audit utama:

- `runs/explanations/m12_unified_report/unified_explanation_report.json`;
- `runs/explanations/m12_unified_report/local_explanation_index.csv`.

M13 menambahkan audit fail-closed SARSA:

- tiga local cases wajib beridentitas `sarsa`;
- path, hash, dan shape checkpoint harus cocok pada local, analysis, dan
  comparison;
- teacher harus mati dan greedy table asli harus menghasilkan action;
- surrogate tidak boleh menggantikan tabel;
- seluruh 9.000 state harus terenumerasi.

Status terakhir: seluruh gate identitas checkpoint dan seluruh gate metodologis
bernilai `true`.

Artefak audit M13:
`runs/explanations/m13_sarsa/m13_sarsa_explanation_report.json`.

