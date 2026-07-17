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

