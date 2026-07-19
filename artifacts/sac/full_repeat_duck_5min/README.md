# SAC Full Task — 5 Menit dengan Repeated Duckie

Folder ini adalah paket checkpoint terkurasi untuk versi repository yang siap
dipush. Video tidak disimpan di Git. Video dapat dirender ulang secara lokal
dari checkpoint dan config di folder ini.

## Isi artifact

| File | Isi |
|---|---|
| `sac_best.pt` | Checkpoint SAC terpilih, 3.14 MB |
| `config.yaml` | Config training/evaluation long-horizon yang menghasilkan checkpoint |
| `config_repeat_duck.yaml` | Deployment config dengan repeated crossing + hysteresis 1 meter |
| `evaluation_30.json` | Evaluasi deterministic 30 episode dengan task kanonis |
| `repeated_duck_audit_3.json` | Audit runtime repeated Duckie dengan hysteresis 1 meter |
| `SHA256SUMS` | Hash integritas seluruh artifact |

Checkpoint SHA-256:

```text
0b01447edd85e539de57f9a304fc287d26d5d7c4e73a5a51cd493fea2f4c4f2b
```

## Catatan repeated Duckie

Checkpoint dilatih memakai task long-horizon kanonis dengan satu crossing per
episode. `config_repeat_duck.yaml` mengaktifkan deployment repeated-Duckie:

```yaml
max_crossings_per_episode: 0
repeat_rearm_distance: 1.0
```

Nol berarti crossing tidak dibatasi. Hysteresis satu meter mencegah Duckie
langsung berbalik ketika ego masih berhenti di zebra crossing.

Audit tiga episode lima menit menghasilkan:

- 3/3 episode mencapai timeout;
- rata-rata 6 crossing per episode;
- tidak ada stop violation;
- mean forward progress sekitar 28.75 meter.

## Evaluasi task kanonis

`evaluation_30.json` mencatat:

| Metrik | Hasil |
|---|---:|
| Timeout rate | 100% |
| Total failure rate | 0% |
| Offroad rate | 0% |
| Duck collision rate | 0% |
| Other collision rate | 0% |
| Task success rate | 83.33% |
| Stop compliance | 97.60% |
| Mean forward progress | 28.05 m |
| Mean absolute lateral error | 0.0627 m |
| p95 absolute lateral error | 0.1048 m |
| Resume after Duckie clear | 100% |

## Setup

Jalankan dari root repository pada WSL/Linux:

```bash
python -m venv .venv-sac
.venv-sac/bin/python -m pip install -r requirements-sac.txt
```

Config meminta CUDA. Ubah `training.device` menjadi `cpu` bila hanya melakukan
inference pada mesin tanpa CUDA.

## Verifikasi repeated crossing

```bash
.venv-sac/bin/python scripts/check_repeated_duck_crossings.py \
  --config artifacts/sac/full_repeat_duck_5min/config_repeat_duck.yaml \
  --checkpoint artifacts/sac/full_repeat_duck_5min/sac_best.pt \
  --episodes 3 \
  --output runs/release_repeat_duck_audit.json
```

## Render ulang video 5 menit

```bash
.venv-sac/bin/python -m src.render_sac_multiview_video \
  --config artifacts/sac/full_repeat_duck_5min/config_repeat_duck.yaml \
  --checkpoint artifacts/sac/full_repeat_duck_5min/sac_best.pt \
  --output runs/sac_best_repeat_duck_multiview_5min_20fps.mp4 \
  --seed 30101 \
  --fps 20 \
  --max-steps 9000
```

Output berada di `runs/`, yang sengaja diabaikan Git.

## Evaluasi ulang task kanonis

```bash
.venv-sac/bin/python -m src.evaluate_sac \
  --config artifacts/sac/full_repeat_duck_5min/config.yaml \
  --checkpoint artifacts/sac/full_repeat_duck_5min/sac_best.pt \
  --episodes 30 \
  --output runs/release_evaluation_30.json
```

Evaluation bersifat deterministic dan teacher-free. Checkpoint tidak diubah.
