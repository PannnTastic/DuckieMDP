# Walkthrough Codebase Explainable MDP Policy

Dokumen ini menjelaskan pipeline explainability DuckieMDP dari M1 sampai M12:
metode yang digunakan, letaknya di kode, proses runtime, hasil eksperimen, dan
cara menafsirkan hasilnya. Spesifikasi formal tetap berada di
[`explainable_mdp_policy_plan.md`](explainable_mdp_policy_plan.md).
Identitas checkpoint, hash, serta peran setiap config dibekukan di
[`explanation_target_audit.md`](explanation_target_audit.md).

---

# 1. Gambaran besar

Dua policy yang dijelaskan adalah tabular Q-learning dan continuous SAC.
Representasi internalnya berbeda, tetapi keduanya diterjemahkan ke state/action
semantik dan bahasa driving primitive yang sama.

```text
Q checkpoint -> QPolicyAdapter ----+
                                   +-> CanonicalState/Action
SAC checkpoint -> SACPolicyAdapter-+           |
                                               v
                                      Primitive labeler
                                               |
                         +---------------------+--------------------+
                         |                     |                    |
                  decision reason       paired outcomes       verification
                    "mengapa?"             "what if?"       "konsisten/aman?"
                         |                     |                    |
                         +---------------------+--------------------+
                                               |
                                      JSON / CSV / video
```

Nama ringkas framework:

> **Primitive-grounded decision explanation, COViz-inspired paired
> action-outcome counterfactual, dan LEGIBLE-inspired metamorphic/safety
> verification.**

### Tiga pilar

1. **Decision explanation** menjelaskan primitive, trigger, Q-margin atau
   diagnostic actor, response curve, dan state counterfactual.
2. **Outcome explanation** membandingkan selected action dengan foil action
   melalui dua branch simulator.
3. **Verification** memeriksa safety property dan metamorphic relation.

Tidak satu pun pilar menggantikan yang lain. Primitive mengatakan *apa* yang
dilakukan; trigger dan state counterfactual membantu menjawab *mengapa*;
paired rollout menunjukkan *apa akibat alternatifnya*; verification memeriksa
*konsistensi dan keselamatannya*.

### Explanation lokal dan global

Explanation lokal dibuat untuk satu state (s_t):

- state semantik;
- selected action (a^*=pi(s_t));
- primitive dan trigger;
- foil action;
- minimal perubahan state;
- paired outcome;
- Q-margin atau SAC diagnostic;
- validity dan support.

Explanation global meliputi policy map, response curves, rule/tree summary,
primitive frequency, transition/duration, dan violation rate.

---

# 2. Policy dan reward yang dijelaskan

### Q-learning

State diskrit:

$$
\bar{s}=(\operatorname{bin}(d),\operatorname{bin}(\phi+d),\operatorname{bin}(v),
\kappa_{cat},\operatorname{bin}(d_{stop}),\sigma_{stop},h_{duck}).
$$

Shape state `(5,5,3,3,4,2,5)` menghasilkan 9.000 state. Dengan tujuh action,
Q-table berbentuk `(5,5,3,3,4,2,5,7)`. Action-nya adalah
`fast_left/straight/right`, `slow_left/straight/right`, dan `brake`.
Implementasi ada di [`src/discretizer.py`](../src/discretizer.py).

Dimensi kedua memakai (e=phi+d), bukan heading murni. Karena itu ia disebut
`lane_heading_entangled` saat dibandingkan dengan SAC.

### SAC

Observation kontinu 15-D:

```text
[d, phi, v, kappa,
 stop_present, d_stop, sigma_stop,
 duck_present, duck_longitudinal, duck_lateral,
 duck_v_longitudinal_relative, duck_v_lateral_relative,
 duck_active, duck_crossing_available, stop_hold_progress]
```

Action SAC adalah (a=(v_{cmd},omega_{cmd})). Implementasi ada di
[`src/continuous_state.py`](../src/continuous_state.py). Explanation
menargetkan deterministic mean actor: sampling dan teacher dimatikan.

### Reward

Explanation tidak mengubah reward. Ia membaca decomposition dari
[`src/reward.py`](../src/reward.py):

$$
r=r_{progress}+r_{lateral}+r_{heading}+r_{time}
+r_{pedestrian}+r_{stagnation}+r_{steering}+r_{events}.
$$

Reward profile dan physical outcome disimpan terpisah. Reward adalah objective
optimisasi; outcome fisik seperti maximum (|d|), minimum Duckie clearance,
stop violation, steering reversal, dan lane departure lebih mudah dipahami.

---

# 3. Peta file

| File | Fungsi |
|---|---|
| [`schema.py`](../src/explainability/schema.py) | Schema state, action, decision, explanation |
| [`semantic_state.py`](../src/explainability/semantic_state.py) | Konversi state Q/SAC |
| [`q_policy_adapter.py`](../src/explainability/q_policy_adapter.py) | Greedy Q query teacher-free |
| [`sac_policy_adapter.py`](../src/explainability/sac_policy_adapter.py) | Deterministic SAC actor query |
| [`primitives.py`](../src/explainability/primitives.py) | Primitive, trigger, threshold, precedence |
| [`trajectory.py`](../src/explainability/trajectory.py) | Step recorder dan segmenter |
| [`scenario_manifest.py`](../src/explainability/scenario_manifest.py) | Branch provenance |
| [`simulator_branching.py`](../src/explainability/simulator_branching.py) | Deterministic replay |
| [`action_outcomes.py`](../src/explainability/action_outcomes.py) | Factual/foil branching |
| [`temporal_outcomes.py`](../src/explainability/temporal_outcomes.py) | Reward/physical profiles |
| [`counterfactual.py`](../src/explainability/counterfactual.py) | Manifold validator |
| [`response_curves.py`](../src/explainability/response_curves.py) | Feature sweeps |
| [`metamorphic.py`](../src/explainability/metamorphic.py) | MR testing |
| [`explain_q.py`](../src/explainability/explain_q.py) | Exact Q audit |
| [`explain_sac.py`](../src/explainability/explain_sac.py) | IG/critic/boundary diagnostics |
| [`rule_extraction.py`](../src/explainability/rule_extraction.py) | Tree/rule extraction |
| [`compare_policies.py`](../src/explainability/compare_policies.py) | Q-SAC behavioral comparison |
| [`explanation_report.py`](../src/explainability/explanation_report.py) | Unified report |
| [`video_overlay.py`](../src/explainability/video_overlay.py) | Overlay primitive/foil/evidence |

---

# 4. Alur satu keputusan

1. Environment menghasilkan `RawState` atau `ContinuousState`.
2. `semantic_state.py` mengubahnya menjadi `CanonicalState`.
3. Adapter men-query policy asli dan menghasilkan `PolicyDecision`.
4. `label_primitive()` membaca state, action, event, dan history.
5. `TrajectoryRecorder.append()` menyimpan state, action, primitive, reward
   terms, event, posisi, waktu, dan termination.
6. Modul M5-M10 menambahkan counterfactual, verification, atau diagnostic.
7. JSON/CSV menjadi data primer; Markdown/video hanya presentasi.

---

# 5. Milestone M1-M12

## M1 - Schema dan policy adapters

**Tujuan:** kedua checkpoint dapat ditanya melalui interface sama tanpa
training ulang atau mutasi checkpoint.

Kode:

- [`schema.py`](../src/explainability/schema.py):
  `CanonicalState`, `CanonicalAction`, `PolicyDecision`,
  `ExplanationRecord`, `SolverKind`, `PolicyMode`;
- [`semantic_state.py`](../src/explainability/semantic_state.py):
  fungsi konversi Q/SAC;
- [`q_policy_adapter.py`](../src/explainability/q_policy_adapter.py):
  `QPolicyAdapter`;
- [`sac_policy_adapter.py`](../src/explainability/sac_policy_adapter.py):
  `SACPolicyAdapter`.

`QPolicyAdapter` memvalidasi shape, membuat snapshot read-only, memilih
greedy argmax, merekam seluruh tie, lalu memilih action id terkecil agar
reproducible. Q-margin dihitung sebagai:

$$
\Delta_Q=Q(s,a_{best})-Q(s,a_{second}).
$$

Q-margin bukan probability atau confidence terkalibrasi.

`SACPolicyAdapter` hanya memuat actor, tidak mengalokasikan replay buffer,
menggunakan actor mean, menyimpan observation names, dan mendukung migrasi
14-D ke 15-D secara eksplisit dengan weight fitur baru nol.

Test:
[`tests/test_explainability_m1.py`](../tests/test_explainability_m1.py).

Hasil utama: Q adapter cocok dengan discretizer tanpa memutasi table; tie-break
reproducible; SAC adapter sama dengan deterministic agent sampai toleransi
(10^{-7}); ekspansi observation tidak terjadi diam-diam.

## M2 - Driving primitive labeler

**Tujuan:** menerjemahkan action mentah menjadi bahasa berkendara bersama.

Kode:
[`primitives.py`](../src/explainability/primitives.py), terutama
`DrivingPrimitive`, `PrimitiveThresholds`, `label_primitive()`, dan
`PrimitiveLabeler`.

Primitive lane mencakup `CruiseStraight`, curve, correction, deceleration,
dan recovery. Primitive stop mencakup approach, decelerate, hold, satisfied,
dan resume. Primitive pedestrian mencakup approach, yield, wait, dan resume.
Perilaku buruk mencakup `UnsafeProceed`, `UnnecessaryBrake`,
`OscillatorySteering`, `PrematureResume`, dan `LaneDeparture`.

Action bukan primitive. Brake dapat berarti `StopHold`, `YieldHold`,
`EmergencyLaneRecovery`, atau `UnnecessaryBrake` tergantung konteks.

Precedence utama: event aktual -> temporal transition -> active pedestrian
risk -> unsatisfied stop -> inactive/relevant pedestrian -> lane behavior.
Dengan demikian Duckie tidak aktif tidak menutupi kewajiban stop, tetapi
Duckie aktif di koridor mendapat prioritas keselamatan.

Freeze:
[`primitive_lexicon_v1.freeze.json`](primitive_lexicon_v1.freeze.json).

Hasil: schema 1.0.1; 15 unit tests passed; rollout primitive coverage Q 100%
pada 250 steps dan SAC 100% pada 300 steps. Threshold seperti hold speed 0,03,
stop hold distance 0,45 m, corridor half-width 0,40 m, dan oscillation omega
0,50 dibekukan sebelum clustering.

## M3 - Trajectory recorder dan segmenter

**Tujuan:** mengubah label per-step menjadi cerita temporal.

Kode:
[`trajectory.py`](../src/explainability/trajectory.py), terutama
`TrajectoryStep`, `PrimitiveSegment`, `TrajectoryRecord`,
`TrajectoryRecorder`, dan `segment_primitives()`.

Contoh segmentasi:

```text
step 0-51   CruiseStraight
step 52-65  DecelerateStop
step 66-72  StopHold
step 73-91  ResumeAfterStop
```

Setiap segment menyimpan duration, cumulative reward, dan event counts. Ini
membedakan brake singkat, full stop, yield hold, diam tidak perlu, dan
oscillatory steering.

Test:
[`tests/test_explainability_trajectory.py`](../tests/test_explainability_trajectory.py).

Validasi meliputi boundaries, reward segment, JSON/JSONL, atomic output,
provenance hash, dan penolakan non-contiguous steps.

## M4 - Deterministic replay dan branching

**Tujuan:** memastikan dua branch counterfactual benar-benar mulai dari
kondisi sama.

Kode:

- [`scenario_manifest.py`](../src/explainability/scenario_manifest.py):
  `ScenarioManifest`, RNG/controller capture, manifest hash;
- [`simulator_branching.py`](../src/explainability/simulator_branching.py):
  `run_action_replay()` dan `assert_replays_identical()`.

Yang dibekukan adalah reset seed, initial conditions, controller parameters,
RNG streams, clock, dan action prefix. Reaksi Duckie terhadap trajectory ego
bersifat endogen: perbedaannya adalah bagian sah dari outcome, bukan leakage.

Acceptance:
[`m4_deterministic_replay_acceptance.json`](m4_deterministic_replay_acceptance.json).

| Skenario | Steps | Crossing | Identik |
|---|---:|---:|---|
| Lane | 120 | 0 | Ya |
| Stop | 250 | 0 | Ya |
| One crossing | 500 | 1 | Ya |
| Repeated crossing | 1.500 | 7 | Ya |

State, action, reward per-term, event, Duckie phase, dan termination identik
pada `atol=1e-7, rtol=0`. M4 adalah blocking gate sebelum M5.
---

## M5 - COViz-inspired paired action outcomes

**Pertanyaan:** apa akibat selected action dibandingkan foil action pada state
awal yang sama?

Kode:

- [`action_outcomes.py`](../src/explainability/action_outcomes.py):
  `prepare_branch()`, `run_paired_outcomes()`;
- [`temporal_outcomes.py`](../src/explainability/temporal_outcomes.py):
  `compute_reward_profile()`, `compute_physical_outcome()`,
  `build_explanation_text()`.

Prosesnya: replay sampai branch point, verifikasi selected action policy,
jalankan factual branch dengan selected action dan counterfactual branch dengan
foil. Hanya action pertama dipaksakan; sesudah itu kedua branch kembali memakai
policy asli yang sama.

Outcome meliputi discounted/undiscounted reward per-term, progress, mean/max
lane error, minimum Duckie clearance, stop/collision, brake ratio, steering
reversal/jerk, primitive sequence, dan termination. Satu rollout disebut
simulator intervention, bukan probability.

Hasil:
[`m5_explanation_results.md`](m5_explanation_results.md) dan
`runs/explanations/{q,sac}_*.json`.

| Solver | Skenario | Selected | Foil | Return H30 selected/foil |
|---|---|---|---|---:|
| Q | Lane | LaneCorrectLeft | CruiseCurveRight | 1,913 / 1,837 |
| Q | Stop | StopHold | DecelerateStop | 18,054 / 18,007 |
| Q | Duckie | YieldHold | YieldDecelerate | 3,117 / 3,191 |
| SAC | Lane | LaneCorrectLeft | CruiseStraight | -33,688 / -33,246 |
| SAC | Stop | StopHold | DecelerateStop | 12,358 / 12,093 |
| SAC | Duckie | YieldHold | PrematureResume | -0,583 / -10,409 |

Hasil negatif dipertahankan. Contohnya, foil Q pedestrian sedikit lebih tinggi
return-nya pada H30. Reward total bukan satu-satunya definisi keselamatan.
Sebaliknya, SAC `PrematureResume` jauh lebih buruk dan memberi contrast yang
jelas.

M4/M5 saat ini memiliki core API, tests, frozen acceptance, dan artifacts,
tetapi belum mempunyai satu CLI runner seperti M6-M12.

## M6 - State counterfactual dan response curves

**Pertanyaan:** perubahan state apa yang membuat policy mengganti action atau
primitive?

M5 mengubah action dan melihat outcome. M6 mengubah state input dan melihat
respons policy.

Kode:

- [`counterfactual.py`](../src/explainability/counterfactual.py):
  `validate_state()`, `make_counterfactual()`, solver projection;
- [`response_curves.py`](../src/explainability/response_curves.py):
  `SweepSpec`, `run_response_curve()`, minimal counterfactual;
- [`run_m6_response_curves.py`](../scripts/run_m6_response_curves.py).

Semua sweep dimulai dari real rollout anchor: lane seed 701 step 5, stop seed
30101 step 21, dan Duckie seed 30101 step 82. Synthetic state divalidasi
sebelum policy dipanggil.

Contoh kontrak manifold:

- stop absent berarti distance absent, sigma false, hold progress zero;
- Duckie absent memakai sentinel geometry dan inactive flags;
- sweep Duckie geometry mewajibkan `duck_present=True`;
- active/crossing flags harus konsisten;
- semua nilai finite dan berada dalam bounds.

Hasil:
[`m6_response_curve_results.md`](m6_response_curve_results.md) dan
[`fig_m6_response_curves.png`](../figures/fig_m6_response_curves.png).

- 20 curves;
- 114 valid-manifold queries;
- 22 rejected points disimpan untuk audit tetapi tidak di-query;
- 76 action flips dan 60 primitive flips;
- Q curvature dan stop-distance sweep menghasilkan nol action flip pada anchor;
- SAC stop-distance menghasilkan tujuh action flips;
- minimal Duckie-longitudinal primitive flip sekitar 0,242 m untuk Q dan
  0,542 m untuk SAC.

Interpretasi: Q response berbentuk tangga dan dibatasi bin; SAC memberi profil
kontinu. Nol flip bukan otomatis bagus: bisa berarti fitur tidak digunakan pada
region tersebut.

## M7 - LEGIBLE-inspired metamorphic testing

**Pertanyaan:** ketika risiko bertambah dan faktor lain tetap, apakah action
berubah ke arah yang masuk akal?

Kode:

- [`metamorphic.py`](../src/explainability/metamorphic.py);
- [`run_m7_metamorphic.py`](../scripts/run_m7_metamorphic.py).

Empat relation:

1. `MR-STOP`: stop makin dekat tidak boleh meningkatkan speed;
2. `MR-PEDESTRIAN`: risiko Duckie meningkat tidak boleh membuat proceed lebih
   agresif;
3. `MR-CURVATURE`: curvature meningkat tidak boleh meningkatkan speed;
4. `MR-LANE-SYMMETRY`: mirror (d,phi) diharapkan mirror steering.

Q memakai ordinal speed `brake=0, slow=1, fast=2`; SAC memakai tolerance
kontinu. Setiap relation memiliki precondition agar tidak diterapkan pada
konteks yang salah.

Hasil:
[`m7_metamorphic_results.md`](m7_metamorphic_results.md).

| Solver | Relation | PASS | FAIL |
|---|---|---:|---:|
| Q | STOP | 6 | 0 |
| Q | PEDESTRIAN | 6 | 0 |
| Q | CURVATURE | 6 | 0 |
| Q | LANE-SYMMETRY | 3 | 3 |
| SAC | STOP | 6 | 0 |
| SAC | PEDESTRIAN | 5 | 1 |
| SAC | CURVATURE | 3 | 3 |
| SAC | LANE-SYMMETRY | 1 | 5 |

Seluruh 48 pairs valid dan applicable. FAIL adalah policy finding, bukan
otomatis bug. Symmetry dapat gagal karena map/support tidak simetris atau
policy memang tidak mempelajari hubungan tersebut. Nama yang tepat adalah
LEGIBLE-inspired, bukan implementasi penuh LEGIBLE.

## M8 - Exact Q-table characterization

**Pertanyaan:** karena Q-table finite, bagaimana membaca dan memverifikasi
seluruh mapping-nya?

Kode:

- [`explain_q.py`](../src/explainability/explain_q.py):
  enumeration, Q-margin, one-bin flips, safety checker, support;
- [`run_m8_exact_q.py`](../scripts/run_m8_exact_q.py).

Empat strata:

1. `representable`: semua indeks table;
2. `valid_manifold`: kombinasi semantik konsisten;
3. `reachable`: muncul minimal sekali dalam rollout;
4. `supported`: evaluation reach count minimal tiga.

Checkpoint historis tidak memiliki training visit count. Jadi supported adalah
evaluation proxy, bukan bukti training visitation.

Hasil:
[`m8_exact_q_results.md`](m8_exact_q_results.md).

| Stratum | State |
|---|---:|
| Representable | 9.000 |
| Valid manifold | 7.875 |
| Reachable | 201 |
| Supported | 138 |

Supported safety properties:

- near unsatisfied stop: 0/4 violations;
- near crossing pedestrian: 0/10 violations.

Reachable pedestrian mempunyai 4/20 violations. Whole-table violation rate
sangat tinggi karena sebagian besar cell tidak dipelajari dan tetap tie/nilai
inisialisasi. Ini terlihat dari Q-margin:

| Stratum | Median margin | Tie rate |
|---|---:|---:|
| Representable | 0,000 | 96,54% |
| Reachable | 5,378 | 4,48% |
| Supported | 5,883 | 0% |

One-bin supported flips:

- lateral 1,92%;
- tracking error entangled 77,14%;
- speed 0%;
- curvature 48,84%;
- stop distance 0%;
- stop satisfied 100%;
- Duckie threat 10,87%.

Kesimpulan: behavior kuat pada wilayah supported tidak berarti seluruh 9.000
cell telah dipelajari. Unsupported violations tetap disimpan untuk audit dan
tidak dinarasikan seolah-olah learned behavior.

## M9 - SAC internal diagnostics

**Pertanyaan:** fitur apa yang mendorong output actor, di mana boundary-nya,
dan seberapa stabil diagnosis tersebut?

Kode:

- [`explain_sac.py`](../src/explainability/explain_sac.py):
  `SACInternalDiagnostics`, `integrated_gradients()`,
  `critic_probes()`, `local_boundary_search()`,
  `attribution_stability()`;
- [`run_m9_sac_diagnostics.py`](../scripts/run_m9_sac_diagnostics.py).

Integrated Gradients:

$$
IG_i(x)=(x_i-x'_i)\int_0^1
\frac{\partial F(x'+\alpha(x-x'))}{\partial x_i}\,d\alpha.
$$

IG dihitung terpisah untuk `v_cmd` dan `omega_cmd`, memakai neutral
canonical baseline dan empirical cruise-straight centroid. Pemilihan baseline
dapat mengubah attribution, sehingga sensitivity wajib dilaporkan.

Hasil:
[`m9_sac_internal_results.md`](m9_sac_internal_results.md).

- 2.000 real development states;
- 57 real cruise-straight states untuk empirical baseline;
- 1.024 integration steps;
- maximum completeness residual 0,002054, limit 0,005;
- stability median 0,000756 dan p95 0,004655, limit 0,10.

Dominant concept yang stabil pada kedua baseline:

- stop `v_cmd`: pedestrian;
- Duckie `v_cmd`: pedestrian;
- Duckie `omega_cmd`: pedestrian.

Lane anchor dan beberapa steering claims tidak stabil, sehingga tidak diklaim
robust.

Critic probes membandingkan actor action dengan action canonical. Karena probe
dapat berada di luar actor distribution dan replay snapshot tidak tersedia,
hasil diberi label `LOW_ACTOR_SUPPORT_NO_REPLAY_SNAPSHOT`. Ia disebut
critic-probe comparison, bukan advantage atau confidence, dan bukan outcome
evidence utama.

## M10 - Solver-aware rule extraction

**Pertanyaan:** bisakah policy diringkas sebagai rules atau decision tree?

Kode:

- [`rule_extraction.py`](../src/explainability/rule_extraction.py);
- [`run_m10_rule_extraction.py`](../scripts/run_m10_rule_extraction.py).

scikit-learn 1.6.1 dan joblib 1.4.2 hanya digunakan post-hoc. Checkpoint tidak
dilatih ulang.

Untuk Q, seluruh 9.000 mapping digunakan. Hasilnya:

- action tree fidelity 100%, depth 15, 257 leaves;
- primitive tree fidelity 100%, depth 15, 138 leaves;
- 1.250 closed-loop decisions dan nol mismatch;
- lima episode timeout tanpa failure.

Tree Q exact pada **domain indeks tabel**, bukan state kontinu di antara bins.

Untuk SAC, development dan held-out masing-masing 2.000 states. Primitive tree
mencapai global fidelity 94,90%, lane 95,54%, pedestrian 94,29%, tetapi stop
hanya 74,14%. Continuous-action tree memiliki MAE (v=0,01029), MAE
(omega=0,07967), dan primitive fidelity 97,45%.

Closed-loop memberi temuan terpenting:

| Metric | Actor asli | Tree surrogate |
|---|---:|---:|
| Failure | 0% | 0% |
| Mean return | 106,74 | 20,73 |
| Stop compliance | 100% | 85,71% |
| Duckie yield-step | 100% | 37,14% |

Kesimpulan: Q tree dapat menjadi exact table summary. SAC tree hanya global
summary dan **tidak boleh menggantikan actor**. High open-loop fidelity tidak
menjamin closed-loop safety equivalence.

Hasil lengkap:
[`m10_rule_extraction_results.md`](m10_rule_extraction_results.md).

## M11 - Bottom-up primitive discovery

**Tujuan:** memeriksa apakah cluster behavior yang ditemukan tanpa label cocok
dengan primitive top-down.

Kode:

- `src/explainability/signatures.py`;
- `src/explainability/cluster_primitives.py`;
- `src/explainability/reconcile_clusters.py`;
- `scripts/run_m11_bottom_up_clustering.py`.

**Status: selesai dan diterima sebagai validasi deskriptif.** M11 memakai 500
fixed windows, bukan boundary `segments.csv` yang berasal dari label M2. Dari
37 fitur tanpa primitive/trigger/event/solver, HDBSCAN menemukan 11 cluster,
coverage 69,40%, silhouette 0,4483, purity 59,94%, NMI 0,4299, dan ARI 0,1967.

Alignment Q-learning lebih kuat (purity 70,27%; NMI 0,6120; ARI 0,3737) daripada
SAC (purity 48,15%; NMI 0,2056; ARI 0,0914). Cluster behavior seluruhnya
solver-specific walaupun nama solver bukan fitur. Safety primitive yang jarang
(`StopHold`, `YieldHold`) masuk noise sehingga belum mendapat konfirmasi
bottom-up dari ensemble lima seed ini. Hasil ini berarti dukungan leksikon
bersifat parsial dan solver-dependent, bukan kegagalan implementasi.

Hasil lengkap:
[`m11_bottom_up_clustering_results.md`](m11_bottom_up_clustering_results.md).

## M12 - Unified comparison, report, dan video

### Matched behavioral comparison

Kode:

- [`compare_policies.py`](../src/explainability/compare_policies.py);
- [`run_m12_policy_comparison.py`](../scripts/run_m12_policy_comparison.py).

Kontrak fairness: `small_loop`, lima seed held-out, initial pose sama pada
(10^{-7}), frame skip 6, horizon 1.500 physics ticks, teacher-free,
Q greedy deterministic dan SAC deterministic mean, bukan
M10 surrogate.

Hasil:

| Metric | Q-learning | SAC |
|---|---:|---:|
| Timeout | 5/5 | 5/5 |
| Mean return | 44,63 | 17,82 |
| Stop compliance | 100% | 100% |
| Pedestrian yield command | 100% | 100% |
| Unsafe proceed | 0% | 0% |
| Unnecessary brake | 0% | 0% |
| Undesirable primitive | 6,96% | 0% |
| Mean first-brake distance | 0,326 m | 0,251 m |
| Straight mean abs omega | 0,480 | 0,540 |

Q mempunyai 87 `OscillatorySteering` steps. SAC tidak memiliki oscillatory
label pada ensemble ini, tetapi mean steering magnitude SAC sedikit lebih
besar. Jadi klaim yang benar: Q lebih sering membalik steering diskrit;
magnitude steering dan reversal frequency adalah dua metrik berbeda.

Q menemukan sepuluh stop opportunities dan SAC lima karena trajectory/progress
mereka berbeda, bukan environment mismatch.

### Influence signatures

Q menggunakan supported one-bin flip rate; SAC menggunakan absolute IG pada
tiga anchors. Nilai dinormalisasi per solver dan hanya dibandingkan
kualitatif.

| Concept comparable | Q | SAC |
|---|---:|---:|
| Speed | 0,000 | 0,155 |
| Road | 0,445 | 0,031 |
| Stop | 0,456 | 0,093 |
| Pedestrian | 0,099 | 0,721 |

Lane/heading dikeluarkan karena Q memakai (phi+d), sedangkan SAC memisahkan
(d) dan (phi). Dua signature ini bukan estimator yang identik.

### Unified report

Kode:

- [`explanation_report.py`](../src/explainability/explanation_report.py);
- [`run_m12_unified_report.py`](../scripts/run_m12_unified_report.py).

Output primer:

- `runs/explanations/m12_unified_report/unified_explanation_report.json`;
- `runs/explanations/m12_unified_report/local_explanation_index.csv`.

Report menggabungkan enam local explanations, M5 paired outcomes, M6 curves,
M7 verification, M8 Q audit, M9 SAC diagnostics, M10 rules, M12 comparison,
hashes, acceptance gates, dan epistemic limitations.

### Video overlay

Kode:

- [`video_overlay.py`](../src/explainability/video_overlay.py);
- [`render_multiview_video.py`](../src/render_multiview_video.py);
- [`render_sac_multiview_video.py`](../src/render_sac_multiview_video.py).

Q overlay menampilkan primitive, trigger, second-best foil, dan exact Q-margin.
SAC overlay menampilkan primitive, trigger, semantic foil, dan critic-probe
delta. Probe tetap diberi caveat dan tidak disebut Q-margin.

Smoke test: 1920x1080, 20 FPS, 20 frames, 30 physics ticks untuk masing-masing
policy. Regression akhir: **147 tests passed**.

Hasil lengkap:
[`m12_policy_comparison_results.md`](m12_policy_comparison_results.md).

---

# 6. Cara membaca satu explanation JSON

Contoh: `runs/explanations/q_lane_correction.json`.

| Field | Makna |
|---|---|
| `selected_decision.state` | Canonical state branch point |
| `selected_decision.action` | Action policy asli |
| `selected_decision.diagnostics` | Q-values/Q-margin atau actor diagnostics |
| `foil_action` | Action pembanding |
| `factual` | Selected-action rollout |
| `counterfactual` | Foil-action rollout |
| `reward_profile` | Return per horizon dan per reward term |
| `physical` | Outcome fisik |
| `primitive_sequence` | Perubahan behavior temporal |
| `branch_invariants` | Bukti branch fair |
| `world_mode` | Reactive/scripted semantics |
| `explanation` | Teks deterministic dari evidence |

`physical_delta_counterfactual_minus_factual` berarti:

$$
\Delta=outcome_{foil}-outcome_{selected}.
$$

Tanda positif tidak selalu lebih baik. Delta maximum lateral error positif
berarti foil lebih buruk; delta progress positif berarti foil lebih maju.

---

# 7. Mengapa metode Q dan SAC berbeda?

| Pertanyaan | Q-learning | SAC |
|---|---|---|
| Policy action | Argmax table | Mean actor |
| Value evidence | Exact Q-values | Learned double critic |
| Separation | Exact Q-margin | Heuristic critic-probe delta |
| Global inspection | Enumerasi 9.000 states | Sampling/probing |
| State response | Step function | Continuous curve |
| Attribution | One-bin flips | Integrated Gradients |
| Rules | Bisa exact pada table domain | Surrogate approximation |
| Verification | Exhaustive finite check | Sampled/interventional |
| Outcome | Paired simulator rollout | Paired simulator rollout |
| Bahasa | Driving primitives | Driving primitives |

Metode internal harus berbeda karena representasi policy berbeda. Metode
behavioral dibuat sama sehingga primitive, counterfactual outcome, dan safety
specification tetap dapat dibandingkan.

---

# 8. Status kekuatan bukti

### Bukti paling langsung

- exact Q lookup;
- deterministic selected action;
- frozen primitive rule dan trigger;
- actual simulator event;
- branch invariants;
- paired simulator physical outcome;
- exhaustive Q enumeration.

### Bukti diagnosis

- Q-margin;
- response curve;
- one-bin influence;
- IG;
- critic probes;
- surrogate tree.

### Yang bukan real-world causal proof

- IG/feature importance;
- learned critic value;
- one simulator branch pair;
- surrogate rules.

Istilah ilmiah yang dipakai adalah **simulator-based interventional
counterfactual**, bukan real-world causal explanation.

---

# 9. Cara menjalankan ulang

Dari root `duckie-mdp`.

### Semua tests

```bash
PYTHONWARNINGS=ignore .venv-sac/bin/python -m pytest -q -p no:warnings
```

### M6-M10

```bash
PYTHONPATH=. .venv-sac/bin/python scripts/run_m6_response_curves.py
PYTHONPATH=. .venv-sac/bin/python scripts/run_m7_metamorphic.py
PYTHONPATH=. .venv-sac/bin/python scripts/run_m8_exact_q.py
PYTHONPATH=. .venv-sac/bin/python scripts/run_m9_sac_diagnostics.py
PYTHONPATH=. .venv-sac/bin/python scripts/run_m10_rule_extraction.py
```

### M12

```bash
PYTHONPATH=. .venv-sac/bin/python scripts/run_m12_policy_comparison.py
PYTHONPATH=. .venv-sac/bin/python scripts/run_m12_unified_report.py
```

### M13 SARSA

```bash
PYTHONPATH=. .venv-sac/bin/python scripts/run_m13_sarsa_local.py
PYTHONPATH=. .venv-sac/bin/python scripts/run_m13_sarsa_analysis.py
PYTHONPATH=. .venv-sac/bin/python scripts/run_m13_sarsa_comparison.py
PYTHONPATH=. .venv-sac/bin/python scripts/run_m13_sarsa_report.py
```

### Lokasi hasil per milestone

| M | Hasil |
|---|---|
| M1 | `tests/test_explainability_m1.py` |
| M2 | `docs/primitive_lexicon_v1.freeze.json` |
| M3 | `tests/test_explainability_trajectory.py`, trajectory JSON |
| M4 | `docs/m4_deterministic_replay_acceptance.json` |
| M5 | `docs/m5_explanation_results.md`, `runs/explanations/{q,sac}_*.json` |
| M6 | `docs/m6_response_curve_results.md`, M6 figure/data |
| M7 | `docs/m7_metamorphic_results.md`, M7 JSON/CSV |
| M8 | `docs/m8_exact_q_results.md`, exact map/flips/violations |
| M9 | `docs/m9_sac_internal_results.md`, IG/probes/boundaries |
| M10 | `docs/m10_rule_extraction_results.md`, trees/rules |
| M11 | `docs/m11_bottom_up_clustering_results.md`, clusters/confusion/signatures |
| M12 | `docs/m12_policy_comparison_results.md`, unified report/video |
| M13 | `docs/m13_sarsa_explanation_results.md`, SARSA report/index |

---

# 10. Kesimpulan keseluruhan

### Q-learning

- berhasil stop dan yield pada matched supported evaluation;
- sebagian besar table cells tidak berisi learned policy yang bermakna;
- supported states mempunyai margin kuat dan nol violation untuk dua safety
  properties utama;
- tracking error, curvature, dan stop-satisfied flag paling sering mengubah
  action;
- discrete macro-actions lebih sering memunculkan oscillatory steering;
- dapat diekstrak menjadi exact tree pada finite table domain.

### SAC

- deterministic actor bertahan, stop, dan yield pada matched seeds;
- merespons stop distance secara lebih gradual;
- pedestrian features dominan pada beberapa IG claims yang stabil;
- sebagian IG claims baseline-sensitive dan sengaja tidak diklaim;
- critic probes hanya diagnostic;
- surrogate tree tampak akurat open-loop tetapi tidak mempertahankan
  closed-loop compliance, sehingga actor asli tetap wajib.

### Perbandingan

M12 bukan eksperimen isolasi algoritma Q vs algoritma SAC, karena keduanya
berbeda pada state representation, action space, exploration, training regime,
dan teacher history. Klaim sahnya adalah perbedaan behavior pada manifest yang
sama.

---

# 11. Limitasi yang wajib disebut

1. Q support memakai evaluation reach count karena training visit count tidak
   tersedia.
2. SAC explanation menargetkan deterministic mean actor, bukan stochastic
   training policy.
3. IG bergantung pada baseline.
4. Critic probes dapat out-of-distribution.
5. Satu M5 branch pair bukan probability.
6. Reactive Duckie boleh bereaksi berbeda setelah branch; itu bagian outcome.
7. Primitive adalah vocabulary rule-based yang dibekukan, bukan isi pikiran
   agent.
8. M11 memberi dukungan bottom-up parsial; rare stop/yield windows masih noise
   dan SAC tidak membentuk pemisahan primitive yang kuat.
9. Lima matched seeds cukup untuk integration validation, bukan population
   inference.
10. Video adalah presentasi; JSON/CSV adalah evidence primer.

---

# 12. Referensi metode

- COViz, AAAI 2024:
  <https://ojs.aaai.org/index.php/AAAI/article/view/28863>
- LEGIBLE, IJCAI 2025:
  <https://www.ijcai.org/proceedings/2025/696>
- Integrated Gradients, Sundararajan et al. 2017:
  <https://proceedings.mlr.press/v70/sundararajan17a.html>
- VIPER/policy extraction, NeurIPS 2018:
  <https://proceedings.neurips.cc/paper/2018/hash/e6d8545daa42d5ced125a4bf747b3688-Abstract.html>

Project ini tidak mengklaim implementasi penuh COViz, LEGIBLE, atau VIPER.
Istilah yang dipakai adalah `COViz-inspired`, `LEGIBLE-inspired`, dan
`solver-aware rule extraction`.

---

# 13. Kalimat siap pakai untuk presentasi

> Kami menjelaskan policy tabular Q-learning/SARSA dan continuous SAC melalui
> satu bahasa driving primitive. Alasan keputusan diperiksa melalui direct Q
> inspection atau actor diagnostics dan state counterfactual. Konsekuensi
> action dijelaskan melalui paired simulator rollouts dari branch point
> identik. Konsistensi dan keselamatan diuji dengan metamorphic relations,
> exhaustive Q-table checking, dan closed-loop evaluation. Seluruh klaim
> disimpan dalam JSON/CSV versioned; tree dan video adalah ringkasan, bukan
> pengganti policy asli atau bukti kausal dunia nyata.


---

# 14. Extension M13: explanation SARSA

SARSA memakai adapter terpisah di
`src/explainability/sarsa_policy_adapter.py`, tetapi memakai schema,
primitive labeler, manifold validator, response-curve engine, metamorphic
checker, exact-table checker, dan paired-outcome engine yang sama dengan
Q-learning.

Perbedaan training tidak dihapus:

```text
Q-learning target = r + gamma max_a Q(s', a)
SARSA target      = r + gamma Q(s', a_next)
```

Setelah training selesai, keduanya dievaluasi sebagai greedy table lookup.
Karena itu explanation internal yang tepat untuk keduanya adalah Q-values,
Q-margin, exact enumeration, one-bin flips, response curves, safety checking,
dan rule extraction. IG/critic probes tetap khusus SAC.

Alur M13:

```text
SARSA checkpoint
  -> SarsaPolicyAdapter (identity tetap sarsa)
  -> 3 paired local explanations
  -> 6 response curves
  -> 24 metamorphic pairs
  -> exact enumeration 9.000 state
  -> action/primitive rule trees
  -> matched Q/SARSA/SAC rollout
  -> fail-closed M13 report
```

Implementasi utama:

- `src/explainability/sarsa_policy_adapter.py`: load tabel dan menjaga solver
  identity `sarsa`;
- `src/explainability/sarsa_explanation_report.py`: mengikat semua artefak,
  hash, shape, mode, dan acceptance gate;
- `scripts/run_m13_sarsa_local.py`: tiga local paired explanations;
- `scripts/run_m13_sarsa_analysis.py`: response, metamorphic, exact, rules;
- `scripts/run_m13_sarsa_comparison.py`: rollout tiga policy pada manifest
  sama;
- `scripts/run_m13_sarsa_report.py`: JSON audit dan CSV index;
- `tests/test_explainability_sarsa.py`: kontrak adapter, checkpoint, shape,
  dan perbandingan tabel.

Hasil dan interpretasi lengkap ada di
`docs/m13_sarsa_explanation_results.md`.

