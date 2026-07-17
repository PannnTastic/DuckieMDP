# Ablation Artifact Manifest

Folder ini menyimpan artefak minimum yang diperlukan untuk memeriksa hasil
tanpa mempertahankan seluruh checkpoint dan video mentah.

## Policy terbaik

```text
lane/
├── q_learning_teacher/
├── sarsa_teacher/
└── q_learning_no_teacher/

full_task/
├── q_learning_teacher/
├── sarsa_teacher/
└── q_learning_no_teacher/
```

Setiap folder policy berisi:

- `config.yaml`: exact historical training config;
- `q_table_best.npy`: Q-table `(5,5,3,3,4,2,5,7)`;
- `training.csv`: log episode;
- `evaluation.json`: evaluasi historis saat eksperimen dibuat.

Folder full-task juga memiliki `evaluation_fair.json`, yaitu evaluasi ulang
dengan satu crossing, 30 episode, progress minimum 5 m, dan brake ratio maksimum
25%. File ini digunakan untuk tabel perbandingan README.

## Progression

`progression/` menyimpan config, training CSV, dan seluruh evaluation JSON untuk
tahapan Q-learning tanpa teacher. Q-table intermediate sengaja tidak disimpan
karena tidak diperlukan untuk tabel ablation dan dapat dibuat ulang dari config.

## Failure ablation

`failures/value_iteration_coverage/` menyimpan transition model, Q-table hasil
Value Iteration, coverage report, dan evaluation. Hasil ini penting sebagai
bukti bahwa konvergensi Bellman pada empirical model tidak menjamin policy baik
bila coverage transition rendah.

SARSA tanpa teacher belum dijalankan; tidak ada artifact atau angka sintetis
untuk sel tersebut.
