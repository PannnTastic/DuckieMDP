# Design Decisions

- Action design: Option A, tujuh discrete semantic actions.
- State design: Option S2, discretized state dengan Q-table berukuran 9,000 x 7.
- `tile` adalah tile lookahead 0.30 m sepanjang lane tangent, bukan tile ego saat ini.
- Stop sign difilter berdasarkan jarak lateral dan orientasi sign terhadap arah lane.
- Stop tracker memakai identitas object sign internal agar pergantian dua sign tidak
  membuat `sigma_stop` tersangkut. Identitas ini tidak masuk state MDP.
- Duckie pada map `udem1` dikonversi saat environment dibuat dari static map object
  menjadi `DuckieObj` dinamis. Map package yang terpasang tidak dimodifikasi di disk.
- `Simulator.step` dipanggil langsung dengan normalized wheel commands. Karena itu
  `v_fast` dan `v_slow` adalah command magnitudes, bukan kecepatan fisik dalam m/s,
  dan transform gain/trim milik `DuckietownEnv` tidak digunakan.
- Episode timeout adalah truncation: loop episode berhenti, tetapi TD target tetap
  bootstrap. Duck collision, other collision, offroad, dan goal adalah termination.
- Termination reason dicatat eksplisit untuk training dan evaluation.
- Tahap lanjutan continuous-control mengikuti
  `docs/continuous_sac_plan.md`. Baseline tabular dan artefak ablation dibekukan;
  SAC memakai wrapper baru agar kontrak `DuckieMDPEnv` tidak berubah.
- Curvature ego-relative kategorikal sudah tersedia. M7 menambahkan signed
  continuous curvature tanpa mengganti fitur kategorikal baseline.
- Reward utama SAC dipertahankan sama dengan tabular. Smoothness penalty hanya
  ablation dan mewajibkan previous action menjadi bagian state.
- "Reward sama" untuk track utama berarti config full-task Q-learning
  **tanpa teacher**, termasuk `duck_unsafe=-5` dan `unnecessary_stop=-2`.
  Teacher-guided tabular adalah track assisted terpisah.
- M9 utama memakai SAC teacher-free. Demonstration-assisted SAC hanya ablation;
  evaluasi semua regime selalu deterministic dan tanpa live teacher.
