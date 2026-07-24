# Certified Driving Primitives from M1–M13 Explanations

## 1. Tujuan dokumen

Dokumen ini menjelaskan secara lengkap bagaimana codebase:

1. membaca policy Q-learning, SARSA, SAC, dan TD3;
2. menghasilkan explanation melalui pipeline M1–M13;
3. menguji explanation melalui tiga pilar:
   - **Decision — Why:** mengapa policy memilih keputusan tersebut;
   - **Outcome — What-if:** apa yang terjadi jika aksi awal diganti;
   - **Verification:** apakah perilakunya konsisten dengan relasi keselamatan;
4. menggabungkan explanation per-state menjadi segmen temporal;
5. mengubah segmen explanation tersebut menjadi vocabulary driving primitive;
6. menentukan apakah hasilnya `CERTIFIED_PRIMITIVE`, `PRIMITIVE_CANDIDATE`, atau `UNKNOWN`;
7. menerjemahkan angka explanation menjadi temporal arc yang mudah dibaca manusia.

Dokumen ini sengaja membedakan tiga objek yang sering tertukar:

| Objek | Arti |
|---|---|
| Local explanation | Bukti untuk satu state/decision: Why, What-if, dan Verification |
| Primitive family | Nama semantik seperti `StopCompliance` atau `CurveNegotiation` |
| Certified primitive cluster | Kelompok segmen explanation yang lolos seluruh certification gate |

Dengan demikian, nama primitive bukan diberikan hanya karena kendaraan sedang mengerem atau berbelok. Primitive dibentuk dari rangkaian explanation yang memiliki pemicu keputusan, outcome counterfactual, bukti verifikasi, struktur temporal, provenance, dan status sertifikasi.

---

## 2. Ringkasan hasil utama

Pipeline menghasilkan **415 segmen temporal** dari empat policy:

- tabular Q-learning;
- tabular SARSA;
- continuous-action SAC;
- continuous-action TD3.

Hasil semantic family assignment:

| Primitive family | Jumlah segmen | Policy yang terwakili | Human-readable temporal arc |
|---|---:|---|---|
| `LaneKeeping` | 78 | Q-learning, SARSA, SAC, TD3 | Monitor lane → Correct → Converge → Cruise |
| `StopCompliance` | 93 | Q-learning, SARSA, SAC, TD3 | Approach → Decelerate → Stop → Hold → Resume |
| `CurveNegotiation` | 223 | Q-learning, SARSA, SAC, TD3 | Detect curve → Reduce speed → Steer → Stabilize |
| `PedestrianYield` | 21 | SAC, TD3 pada separation explanation-level | Detect pedestrian → Decelerate → Yield hold → Clear → Resume |
| **Total** | **415** | Empat policy | — |

Hasil discovery dan certification:

| Ukuran | Nilai | Makna |
|---|---:|---|
| Segmen temporal | 415 | Seluruh candidate segment sebelum clustering |
| Segmen mendapat cluster | 318 | Segmen yang berada di support cluster |
| Noise/di luar cluster | 97 | Tidak dipaksa masuk primitive |
| Cluster ditemukan | 23 | Kelompok explanation-temporal hasil discovery |
| `CERTIFIED_PRIMITIVE` | 8 cluster | Lolos seluruh certification gate |
| `PRIMITIVE_CANDIDATE` | 15 cluster | Memiliki pola, tetapi belum lolos semua gate |
| Runtime certified assignment | 137/415 | Segmen yang dapat dipetakan ke support certified pada runtime |
| Runtime `UNKNOWN` | 278/415 | Segmen sengaja tidak diberi klaim certified |

Catatan penting:

- **415 bukan jumlah primitive yang certified.** Itu jumlah seluruh segmen yang memperoleh family vocabulary.
- **8 adalah jumlah cluster certified**, bukan jumlah segmen.
- **137/415 adalah runtime assignment ke support certified.**
- `PedestrianYield` berhasil menjadi family yang terpisah pada SAC dan TD3, tetapi discovered cluster-nya masih `PRIMITIVE_CANDIDATE`, bukan certified.
- `UNKNOWN` adalah mekanisme keselamatan epistemik. Sistem memilih tidak membuat klaim ketika bukti atau support tidak cukup.

Artefak utama:

- [`primitive_catalogue.md`](../runs/explanations/cedp_v2_4policy/primitive_catalogue.md)
- [`primitive_certificates.jsonl`](../runs/explanations/cedp_v2_4policy/primitive_certificates.jsonl)
- [`certification_summary.json`](../runs/explanations/cedp_v2_4policy/certification_summary.json)
- [`discovery_summary.json`](../runs/explanations/cedp_v2_4policy/discovery_summary.json)
- [`runtime_assignments.json`](../runs/explanations/cedp_v2_4policy/runtime_assignments.json)
- [`cedp_v2_report.json`](../runs/explanations/cedp_v2_4policy/cedp_v2_report.json)

---

## 3. Arsitektur keseluruhan

```text
Q-learning / SARSA / SAC / TD3 checkpoint
                    │
                    ▼
M1  Canonical state, action, dan policy adapter
                    │
                    ▼
M2  Frozen top-down primitive vocabulary
                    │
                    ▼
M3  Temporal trajectory recording dan segmentation
                    │
                    ▼
M4  Deterministic scenario replay
                    │
                    ▼
M5  Paired factual-versus-foil outcome rollout
                    │
                    ▼
M6  Valid state counterfactual dan response curve
                    │
                    ▼
M7  Metamorphic verification
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
M8 Exact Q/SARSA         M9 SAC diagnostics
          │                   │
          └─────────┬─────────┘
                    ▼
M10 Rule extraction dan fidelity
                    │
                    ▼
M11 Bottom-up explanation signature clustering
                    │
                    ▼
M12 Unified cross-policy report dan rendering
                    │
                    ▼
M13 SARSA local explanation extension
                    │
                    ▼
C-EDP v2:
collect → segment → discover → describe → certify
        → reconcile → frozen runtime assignment
                    │
                    ▼
Why + What-if + Verification + Temporal arc + Status
                    │
                    ▼
LaneKeeping / StopCompliance /
CurveNegotiation / PedestrianYield
```

M1–M13 membangun dan memvalidasi explanation. C-EDP v2 tidak mengganti hasil tersebut. C-EDP v2 **memakai ulang explanation M1–M13**, mengelompokkannya secara temporal, menemukan regularitas explanation, memberi descriptor, dan menjalankan certification gate.

---

## 4. Bentuk satu explanation

Satu explanation lengkap direpresentasikan secara konseptual sebagai:

```text
Explanation E =
(
    decision_summary,
    outcome_summary,
    verification_summary,
    temporal_summary,
    provenance,
    status
)
```

### 4.1 `decision_summary` — Why

Menjelaskan fitur state mana yang mampu mengubah keputusan policy.

Contoh:

```text
lateral_flip = 1.000
heading_flip = 1.000
stop_satisfied_flip = 1.000
```

Interpretasi:

- `lateral_flip=1` berarti intervensi valid pada lateral error mengubah keputusan;
- `heading_flip=1` berarti keputusan sensitif terhadap heading error;
- `stop_satisfied_flip=1` berarti status kewajiban stop merupakan decision lever;
- `duck_risk_flip=1` berarti perubahan ancaman pedestrian dapat membalik aksi/primitive.

Nilai flip bukan probabilitas dan bukan “tingkat keyakinan”. Nilai itu menunjukkan apakah intervensi pada fitur menghasilkan perubahan keputusan sesuai prosedur counterfactual.

### 4.2 `outcome_summary` — What-if

Membandingkan akibat temporal aksi factual dengan aksi foil.

Contoh:

```text
factual_cumulative_steering_jerk = 20.385
foil_cumulative_steering_jerk = 22.113
rollout_steps = 15
elapsed_seconds = 3.0
```

Interpretasi:

- cabang factual menjalankan aksi policy pada decision pertama;
- cabang foil memaksa aksi pembanding pada decision pertama;
- setelah itu keduanya kembali memakai policy yang sama;
- kondisi awal, RNG stream, controller parameters, dan prefix dibuat sama;
- perbedaan trajectory setelah branch merupakan outcome intervensi aksi.

Outcome tidak hanya reward. Pipeline juga mencatat:

- lane departure;
- minimum pedestrian clearance;
- stop violation;
- progress;
- heading/lateral error;
- brake duration;
- steering reversal dan jerk;
- primitive sequence;
- termination reason.

### 4.3 `verification_summary` — Is it consistent and safe?

Menyimpan hasil metamorphic relation dan safety-property checking.

Contoh:

```text
pedestrian_pass = 1.000
curvature_pass = 0.620
lane_symmetry_pass = 0.620
```

Interpretasi:

- nilai mendekati `1` berarti relation applicable lebih sering dipenuhi;
- nilai di bawah `1` menunjukkan pelanggaran atau boundary behavior;
- relation hanya dinilai saat precondition-nya berlaku;
- nilai yang tidak applicable tidak boleh diperlakukan sebagai kegagalan.

### 4.4 `temporal_summary` — bagaimana keputusan berkembang

Descriptor mentah dapat berisi:

```text
previous_distance = 4.764
continuity = 1.000
persistence = 1.000
decision_change = 0.420
```

Angka tersebut kemudian dirender menjadi temporal arc manusia:

```text
Approach → Decelerate → Stop → Hold → Resume
```

Temporal arc tidak mengganti bukti numerik. Arc adalah presentation layer yang merangkum urutan bukti.

### 4.5 `status`

```text
CERTIFIED_PRIMITIVE
PRIMITIVE_CANDIDATE
UNKNOWN
```

- `CERTIFIED_PRIMITIVE`: seluruh certification gate lulus;
- `PRIMITIVE_CANDIDATE`: pola explanation ada, tetapi satu atau lebih gate belum lulus;
- `UNKNOWN`: runtime instance berada di luar support yang aman untuk diklaim.

---

## 5. M1–M13: fungsi setiap tahap dan setiap kode

## M1 — Schema dan policy adapters

Tujuan M1 adalah membuat empat solver terlihat seragam bagi pipeline explanation.

### `src/explainability/schema.py`

Tugas:

- mendefinisikan `CanonicalState`;
- mendefinisikan canonical discrete/continuous action;
- mendefinisikan `PolicyDecision`;
- mendefinisikan trajectory/explanation record;
- mendefinisikan enum solver, primitive, outcome, dan status.

Mengapa diperlukan:

Q-table memakai indeks diskrit, sedangkan SAC/TD3 memakai vector kontinu. Tanpa schema bersama, hasil antar-policy tidak dapat dibandingkan.

### `src/explainability/semantic_state.py`

Tugas:

- mengubah raw simulator state menjadi nama fitur semantik;
- memetakan state diskrit Q/SARSA ke concept space;
- memetakan observation vector SAC/TD3 ke nama seperti lateral error, heading error, speed, curvature, stop distance, dan duck geometry;
- menjaga sentinel untuk object yang absent.

Output:

```text
semantic state dengan nama dan satuan yang konsisten
```

### `src/explainability/q_policy_adapter.py`

Tugas:

- memuat Q-table Q-learning atau SARSA;
- melakukan greedy inference;
- mengembalikan seluruh Q-value tujuh aksi;
- menghitung best action, second-best action, Q-margin, dan tie;
- mengubah action index menjadi nama aksi.

Q-learning dan SARSA sama-sama memakai adapter tabular karena bentuk artefaknya sama: tabel `Q(s,a)`. Perbedaannya berada pada cara tabel dilatih, bukan cara tabel dibaca ketika evaluasi.

### `src/explainability/sac_policy_adapter.py`

Tugas:

- memuat checkpoint actor SAC;
- membentuk observation dengan urutan fitur training yang benar;
- memakai deterministic actor mean saat explanation;
- mengubah output actor menjadi `(v_cmd, omega_cmd)`;
- menyediakan metadata observation names dan checkpoint migration.

Mengapa deterministic:

Explanation harus dapat direproduksi. Sampling actor saat evaluasi akan mencampurkan alasan policy dengan noise eksplorasi.

### `src/explainability/td3_policy_adapter.py`

Tugas:

- memuat deterministic actor TD3;
- memakai canonical continuous observation yang sama;
- mengubah output menjadi `(v_cmd, omega_cmd)`;
- membuat TD3 dapat mengikuti pipeline explanation yang sama dengan SAC.

### `tests/test_explainability_m1.py`

Tugas:

- memeriksa schema;
- memeriksa shape state/action;
- memeriksa adapter mengeluarkan keputusan dalam format yang sama;
- mencegah perubahan lama merusak kontrak explanation.

Kontribusi terhadap primitive:

M1 memastikan semua primitive nanti mempunyai input semantik dan format keputusan yang sama, walaupun solver-nya berbeda.

---

## M2 — Primitive vocabulary dan labeler

### `src/explainability/primitives.py`

Tugas:

- mendefinisikan vocabulary top-down seperti lane correction, stop, yield, resume, cruise, dan undesirable behavior;
- mengubah state-action-event menjadi label perilaku;
- menyediakan aturan temporal untuk membedakan aksi mentah dan primitive.

Contoh:

`brake` tidak otomatis berarti satu primitive. Bergantung konteks, ia dapat berarti:

- `StopHold`;
- `YieldHold`;
- `EmergencyBrake`;
- `UnnecessaryBrake`.

### `docs/primitive_lexicon_v1.freeze.json`

Tugas:

- membekukan aturan vocabulary sebelum bottom-up clustering;
- menyimpan hash/versi leksikon;
- mencegah label diubah setelah melihat cluster;
- membuat reconciliation top-down versus bottom-up sah.

Kontribusi terhadap primitive:

M2 menyediakan bahasa manusia untuk reconciliation dan presentation. Label M2 **tidak digunakan sebagai fitur dalam explanation-only clustering**, sehingga cluster tidak sekadar menyalin label yang sudah diberikan.

---

## M3 — Trajectory recording dan temporal segmentation

### `src/explainability/trajectory.py`

Tugas:

- merekam state, action, reward terms, event, primitive label, dan termination;
- menyatukan keputusan individual menjadi episode;
- membentuk segment saat primitive/context berubah;
- menyimpan provenance solver, seed, checkpoint, config, dan timestep.

### `src/explainability/certified_primitives/trajectory.py`

Tugas:

- membaca local explanation records M1–M13;
- mengurutkannya berdasarkan solver, seed, episode, dan waktu;
- menjaga hubungan instance dengan factual/foil trajectory.

### `src/explainability/certified_primitives/segmentation.py`

Tugas:

- mengelompokkan explanation yang berdekatan menjadi temporal segment;
- memisahkan perubahan decision/outcome/verification yang bermakna;
- mencegah satu timestep dianggap primitive lengkap.

### `src/explainability/certified_primitives/signature.py`

Tugas:

- mengubah satu segment menjadi explanation signature numerik;
- hanya memakai fitur decision, counterfactual outcome, verification, certificate, dan temporal;
- menolak token raw state/action/context yang dilarang sebagai discovery feature.

### `scripts/run_cedp_segment.py`

Tugas:

- menjalankan temporal segmentation;
- menyimpan segment records dan explanation signatures.

Kontribusi terhadap primitive:

M3 adalah titik ketika local explanation per-state berubah menjadi kandidat perilaku temporal. Primitive tidak lagi “aksi pada satu frame”, melainkan rangkaian keputusan yang bertahan dan berubah sepanjang waktu.

---

## M4 — Deterministic replay dan scenario manifest

### `src/explainability/scenario_manifest.py`

Tugas:

- menyimpan map;
- initial ego pose;
- initial duck/controller state;
- simulation clock;
- RNG streams;
- config;
- action prefix;
- world mode;
- manifest hash.

`world_mode`:

- `reactive`: Duckie/controller boleh bereaksi terhadap trajectory ego. Reaksi tersebut adalah bagian outcome;
- `scripted`: jadwal eksternal dibekukan terhadap clock untuk ablation.

### `src/explainability/simulator_branching.py`

Tugas:

- melakukan reset dari manifest;
- memutar ulang action prefix sampai branch state;
- membuat factual dan counterfactual branch;
- membandingkan replay identik;
- menghindari ketergantungan pada `deepcopy(env)` yang tidak aman untuk renderer/OpenGL.

### `docs/m4_deterministic_replay_acceptance.json`

Acceptance contract:

```text
atol = 1e-7
rtol = 0
```

Makna:

- `atol` adalah toleransi absolut maksimum;
- `rtol=0` berarti tidak ada toleransi yang membesar mengikuti skala nilai;
- state kontinu, reward terms, controller phase, dan termination harus identik dalam batas tersebut.

Hasil:

- lane scenario identik;
- stop scenario identik;
- crossing scenario identik;
- repeated crossing/rearm scenario identik.

Kontribusi terhadap primitive:

Tanpa M4, perbedaan factual versus foil dapat berasal dari random event yang berbeda. Dengan M4, outcome explanation dapat diatribusikan pada intervensi aksi dalam simulator di bawah kondisi eksogen terkontrol.

---

## M5 — Counterfactual action-outcome

### `src/explainability/action_outcomes.py`

Tugas:

- memilih factual action dari policy;
- memilih foil action dengan protokol yang dibekukan;
- menjalankan paired branch;
- setelah first action, mengembalikan kedua branch ke policy yang sama;
- menghitung perbedaan outcome.

Foil tabular:

- second-best Q action;
- semantic foil;
- safety foil;
- seluruh enam alternatif untuk audit tambahan.

Foil continuous:

- nearest primitive-changing action;
- canonical brake/straight/left/right;
- critic-supported action jika masih berada pada policy support.

### `src/explainability/temporal_outcomes.py`

Tugas:

- menghitung cumulative reward per term;
- menghitung physical outcomes;
- menghasilkan fixed-step horizon;
- menghasilkan event-aligned horizon;
- membentuk ringkasan factual versus foil.

Kontribusi terhadap primitive:

M5 memberikan bagian **What-if**. Contohnya, `StopCompliance` bukan hanya “policy mengerem”, tetapi “jika fast-straight dipaksakan pada state awal yang sama, branch alternatif melanggar stop/collision sementara factual branch berhenti.”

---

## M6 — State counterfactual dan response curves

### `src/explainability/counterfactual.py`

Tugas:

- memeriksa semantic manifold;
- memastikan kombinasi fitur masuk akal;
- melakukan perubahan minimal pada state;
- mencari perubahan terkecil yang membalik aksi atau primitive.

Contoh kontrak:

- `duck_present=0` harus menggunakan sentinel geometry;
- `stop_present=0` tidak boleh memiliki near stop distance;
- `duck_active`, crossing availability, dan controller phase harus konsisten.

### `src/explainability/response_curves.py`

Tugas:

- menyapu satu fitur dari anchor state rollout nyata;
- menahan fitur lain tetap;
- mencatat perubahan action/primitive;
- menghasilkan kurva mulus SAC/TD3 dan fungsi tangga Q/SARSA.

### `scripts/run_m6_response_curves.py`

Tugas:

- menjalankan sweep valid;
- menyimpan CSV/JSON;
- menghasilkan figure.

Hasil audit M6:

- 20 response curves;
- 114 valid counterfactual states;
- 22 state ditolak validator;
- 76 action flips;
- 60 primitive flips.

Kontribusi terhadap primitive:

M6 memberikan bagian **Why** dalam bentuk trigger dan minimal state change: kondisi apa yang harus berubah agar policy mengganti primitive.

---

## M7 — Metamorphic verification

### `src/explainability/metamorphic.py`

Tugas:

- mendefinisikan relation dengan precondition;
- menghasilkan paired state yang valid;
- memeriksa apakah perubahan action sesuai domain expectation;
- mendukung discrete speed level dan continuous tolerance.

Relation utama:

1. stop-distance monotonicity;
2. pedestrian-risk monotonicity;
3. curvature-speed monotonicity;
4. lane left-right symmetry.

Contoh:

```text
Jika stop belum dipenuhi, jalan lurus, lane error aman,
dan stop line dibuat lebih dekat,
maka velocity command tidak boleh meningkat.
```

Kontribusi terhadap primitive:

M7 memberikan bagian **Verification**. Primitive tidak hanya memiliki nama dan outcome, tetapi diuji apakah response policy konsisten terhadap perubahan keadaan yang seharusnya relevan.

---

## M8 — Exact tabular explanation dan verification

### `src/explainability/explain_q.py`

Tugas:

- enumerate seluruh 9.000 indeks representable Q-table;
- memeriksa semantic validity;
- membaca greedy action dan Q-margin;
- menghitung one-bin counterfactual;
- memeriksa properties;
- memisahkan representable, valid, reachable, dan supported state.

### `scripts/run_m8_exact_q.py`

Tugas:

- menjalankan exact tabular audit;
- menghasilkan CSV setiap state;
- menghasilkan ringkasan property pass/violation.

Hasil:

| Stratum | Jumlah |
|---|---:|
| Representable | 9.000 |
| Semantically valid | 7.875 |
| Evaluation reachable | 201 |
| Sufficiently supported | 138 |

Near-condition supported checks:

- near stop: 0 violation dari 4 applicable supported states;
- near pedestrian: 0 violation dari 10 applicable supported states.

Kontribusi terhadap primitive:

M8 memberikan exact policy evidence untuk Q-learning. SARSA dapat diaudit dengan mekanisme yang sama karena bentuk tabelnya identik.

---

## M9 — SAC internal diagnostics

### `src/explainability/explain_sac.py`

Tugas:

- menjalankan deterministic actor inference;
- menghitung Integrated Gradients terhadap `v_cmd` dan `omega_cmd`;
- memakai canonical neutral baseline;
- menghitung critic probes terhadap canonical alternative actions;
- memberi support caveat untuk action di luar distribusi actor;
- mengukur attribution stability dan completeness.

### `scripts/run_m9_sac_diagnostics.py`

Tugas:

- sampling valid rollout states;
- menjalankan actor/critic diagnostics;
- menyimpan CSV attribution dan critic comparison;
- membuat visualisasi.

Hasil audit:

- 2.000 states dianalisis;
- 1.024 diagnostic steps;
- IG completeness error sekitar `0.002054`;
- p95 stability distance sekitar `0.004655`.

Kontribusi terhadap primitive:

M9 memberi bukti internal tambahan untuk continuous actor. Namun primitive tidak ditentukan hanya dari IG. Bukti utama tetap trigger counterfactual, paired outcome, verification, dan temporal behavior.

---

## M10 — Rule extraction

### `src/explainability/rule_extraction.py`

Tugas:

- melatih decision-tree/rule surrogate;
- untuk Q/SARSA, memprediksi discrete action;
- untuk SAC/TD3, memprediksi primitive dan/atau continuous action approximation;
- menghitung fidelity;
- menguji surrogate secara closed-loop;
- menyimpan rule text dan tree.

### `scripts/run_m10_rule_extraction.py`

Tugas:

- menyiapkan dataset;
- melatih tree menggunakan scikit-learn;
- mengevaluasi held-out fidelity;
- menghasilkan file rules/figures.

Hasil audit:

- Q action-tree fidelity: 100%;
- SAC primitive-tree fidelity: sekitar 94,90%;
- SAC surrogate mengalami closed-loop degradation.

Interpretasi:

Tree adalah ringkasan policy, bukan policy asli. Fidelity tinggi pada sample belum menjamin closed-loop equivalence.

Kontribusi terhadap primitive:

M10 menyediakan aturan manusia yang meringkas kapan primitive dipilih, tetapi tidak mengganti counterfactual proof atau original policy.

---

## M11 — Bottom-up clustering dan reconciliation

### `src/explainability/signatures.py`

Tugas:

- membentuk influence signature untuk pipeline M11 awal;
- menormalisasi representation lintas solver.

### `src/explainability/cluster_primitives.py`

Tugas:

- melakukan dimensionality reduction;
- menjalankan clustering;
- menandai noise;
- menghitung coverage dan silhouette.

### `src/explainability/reconcile_clusters.py`

Tugas:

- membandingkan cluster bottom-up dengan frozen M2 lexicon;
- menghitung confusion matrix;
- menghitung ARI/NMI/purity;
- menjaga independence antara discovery dan top-down labels.

### `scripts/run_m11_bottom_up_clustering.py`

Tugas:

- menjalankan seluruh M11;
- menghasilkan cluster records, figures, dan reconciliation report.

Hasil M11 awal:

- 500 windows;
- 11 clusters;
- coverage 69,4%;
- silhouette sekitar 0,4483;
- hasil partial, sehingga kemudian diperkuat menjadi C-EDP v2.

Kontribusi terhadap primitive:

M11 menguji apakah struktur perilaku yang mirip primitive muncul dari explanation signature, bukan hanya dari label buatan manusia.

---

## M12 — Cross-policy comparison dan unified presentation

### `src/explainability/compare_policies.py`

Tugas:

- mencocokkan scenario/state lintas Q-learning dan SAC;
- membandingkan action, primitive, explanation, dan outcome;
- menghasilkan matched policy records.

### `scripts/run_m12_policy_comparison.py`

Tugas:

- menjalankan perbandingan policy;
- menyimpan disagreement dan agreement cases.

### `src/explainability/explanation_report.py`

Tugas:

- menyatukan Why, What-if, Verification, temporal, dan provenance;
- menghasilkan JSON/Markdown/CSV yang dapat diaudit.

### `scripts/run_m12_unified_report.py`

Tugas:

- membangun laporan akhir M1–M12;
- menggabungkan artefak figure, table, dan local explanation.

### Video overlay/renderers

Tugas:

- menampilkan agent camera/BEV;
- mengganti panel vantage dengan explanation;
- menampilkan state, action, primitive, counterfactual outcome, dan status;
- membuat explanation dapat diinspeksi sepanjang trajectory.

Kontribusi terhadap primitive:

M12 adalah lapisan penyajian bersama. Perbedaan representation solver dipertahankan, tetapi bahasa output diseragamkan menjadi primitive.

---

## M13 — SARSA extension

### `scripts/run_m13_sarsa_local.py`

Tugas:

- memuat checkpoint SARSA;
- menghasilkan local explanation dengan adapter tabular.

### `scripts/run_m13_sarsa_analysis.py`

Tugas:

- menjalankan response/intervention/property analysis pada SARSA.

### `scripts/run_m13_sarsa_comparison.py`

Tugas:

- membandingkan SARSA dengan Q-learning dan continuous policy.

### `scripts/run_m13_sarsa_report.py`

Tugas:

- menghasilkan laporan SARSA terintegrasi.

Kontribusi terhadap primitive:

M13 membuktikan pipeline tidak khusus Q-learning atau SAC. Policy tabular on-policy SARSA dapat dijelaskan menggunakan semantic state, paired outcome, verification, dan vocabulary yang sama.

---

## 6. C-EDP v2: mengubah explanation M1–M13 menjadi primitive

Pipeline C-EDP v2 adalah tahap agregasi dan certification setelah explanation tersedia.

### `src/explainability/certified_primitives/provenance.py`

Tugas:

- mencatat commit/config/checkpoint/hash;
- memastikan explanation dapat ditelusuri ke policy dan eksperimen asal.

### `scripts/run_cedp_freeze.py`

Tugas:

- membekukan protocol;
- menyimpan version/hash;
- mencegah threshold dan rule berubah setelah hasil dilihat.

### `src/explainability/certified_primitives/certificate_adapter.py`

Tugas:

- membaca output certificate M1–M13;
- menormalisasi format certificate antar-solver.

### `src/explainability/certified_primitives/collection.py`

Tugas:

- mengumpulkan local explanation;
- memeriksa completeness;
- menyimpan per-instance shard agar proses panjang dapat dilanjutkan.

### `scripts/run_cedp_collect.py`

Tugas:

- menghasilkan factual/foil explanation instances;
- mengumpulkan empat policy;
- menyimpan progress secara incremental.

Hasil:

```text
4.000 local explanation instances
1.000 per solver
```

### `src/explainability/certified_primitives/segmentation.py`

Tugas:

- mengubah 4.000 local explanation menjadi candidate temporal segments.

Hasil:

```text
415 temporal segments
```

### `src/explainability/certified_primitives/signature.py`

Tugas:

- membangun explanation-only feature vector;
- tidak memakai raw state/action label sebagai shortcut.

Kelompok fitur:

- decision counterfactual;
- physical outcome contrast;
- verification result;
- certificate strength;
- temporal continuity/persistence/change.

### `src/explainability/certified_primitives/discovery.py`

Tugas:

- membagi development dan held-out berdasarkan solver seed;
- fit `StandardScaler` hanya pada development;
- fit PCA hanya pada development;
- fit HDBSCAN hanya pada development;
- melakukan frozen held-out assignment;
- menghitung bootstrap stability;
- menghitung outcome coherence.

Hasil:

```text
23 discovered clusters
318/415 assigned
97 noise
```

### `src/explainability/certified_primitives/descriptor.py`

Tugas:

- memilih family vocabulary berbasis explanation;
- membuat:
  - `decision_summary`;
  - `outcome_summary`;
  - `verification_summary`;
  - `temporal_summary`;
- tidak menjadikan top-five angka sebagai satu-satunya alasan family.

Explanation-derived family cues:

| Family | Explanation cue utama |
|---|---|
| `LaneKeeping` | lane-symmetry applicability dan lane recovery structure |
| `StopCompliance` | `stop_satisfied` counterfactual lever dan stop outcome |
| `PedestrianYield` | pedestrian-clearance outcome serta duck-related lever/verification |
| `CurveNegotiation` | sustained steering/curvature outcome pada segment yang bukan stop/pedestrian/lane-recovery family |

### `src/explainability/certified_primitives/certificate_checker.py`

Tugas:

- memeriksa seluruh anggota cluster;
- memeriksa support;
- memeriksa seed diversity;
- memeriksa held-out reproduction;
- memeriksa bootstrap stability;
- memeriksa outcome coherence;
- memeriksa property/counterexample;
- memeriksa traceability;
- mengeluarkan status.

### `scripts/run_cedp_certify.py`

Tugas:

- menjalankan certification gates;
- menulis certificate JSONL dan summary.

### `src/explainability/certified_primitives/reconciliation.py`

Tugas:

- membandingkan explanation-derived cluster dengan frozen M2 lexicon;
- mengukur apakah vocabulary manusia selaras dengan pola yang ditemukan.

### `scripts/run_cedp_reconcile.py`

Tugas:

- menghasilkan reconciliation tables dan metrics.

### `src/explainability/certified_primitives/runtime.py`

Tugas:

- memuat frozen scaler, PCA, clusterer, dan certificate;
- menerima explanation segment baru;
- mengembalikan certified primitive hanya jika berada dalam certified support;
- mengembalikan `UNKNOWN` bila tidak cukup bukti.

### `scripts/run_cedp_runtime.py`

Tugas:

- mengaudit runtime assignment;
- menghasilkan coverage dan UNKNOWN rate.

### `src/explainability/certified_primitives/reporting.py`

Tugas:

- membentuk catalogue;
- menggabungkan evidence;
- menghasilkan tabel status, solver coverage, dan narrative fields.

### `scripts/run_cedp_ablation.py`

Tugas:

- menguji kontribusi kelompok fitur;
- memastikan hasil bukan akibat satu shortcut feature.

### `scripts/run_cedp_report.py`

Tugas:

- menghasilkan laporan lengkap C-EDP.

---

## 7. Bukti primitive 1 — `LaneKeeping`

## 7.1 Definisi manusia

`LaneKeeping` adalah rangkaian keputusan untuk mengamati error lane, memberi koreksi steering yang sesuai, mengurangi error, menstabilkan heading, lalu kembali cruise.

Temporal arc:

```text
Monitor lane
    → detect lateral/heading deviation
    → corrective steering
    → error converges
    → steering neutralizes
    → Cruise
```

Versi ringkas:

```text
Monitor → Correct → Converge → Cruise
```

## 7.2 Why

Decision evidence yang diharapkan:

- `lateral_flip`;
- `heading_flip`;
- lane-symmetry applicability;
- steering/action change ketika tanda lateral/heading dibalik.

Contoh certified cluster `LaneKeeping_C16`:

```text
support = 17 segments
solver support = Q-learning + SARSA
status = CERTIFIED_PRIMITIVE

decision_summary:
- lateral_flip = 1.000
- heading_flip = 1.000
- speed_abs_delta = 1.000
- stop_distance_flip = 1.000
- stop_distance_abs_delta = 1.000
```

Top numeric feature adalah ringkasan statistik cluster, bukan narasi lengkap. Family `LaneKeeping` terutama didukung oleh structure lane intervention, lane-symmetry applicability, dan temporal recovery.

## 7.3 What-if

Contoh `LaneKeeping_C16`:

```text
factual_cumulative_steering_jerk = 35.841
foil_cumulative_steering_jerk = 39.352
delta steering jerk = +3.511 pada foil
```

Interpretasi:

Foil menghasilkan kontrol steering yang lebih kasar. Factual policy mempertahankan trajectory koreksi yang lebih stabil.

## 7.4 Verification

Contoh `LaneKeeping_C16`:

```text
pedestrian_pass = 1.000
curvature_pass = 0.620
lane_symmetry_pass = 0.620
```

`lane_symmetry_pass` adalah relation paling relevan: membalik tanda lateral dan heading pada konteks simetris seharusnya membalik arah steering secara konsisten.

## 7.5 Temporal evidence

Contoh:

```text
previous_distance = 10.719
verification_transition = 1.079
continuity = 1.000
persistence = 0.956
```

Human-readable rendering:

```text
Lane deviation detected
    → corrective steering persists
    → lateral/heading error decreases
    → steering returns toward neutral
    → lane-centered cruise resumes
```

## 7.6 Presentation overlay

Macro renderer menemukan:

| Macro | Instances | Decisions | Solver | Speed arc |
|---|---:|---:|---|---|
| LaneCorrectionLeft | 55 | 663 | Semua 4 policy | `0.101 → 0.082 → 0.173` |
| LaneCorrectionRight | 30 | 180 | Semua 4 policy | `0.150 → 0.105 → 0.189` |

Macro overlay memakai raw state/action hanya untuk merender arc manusia. Ia bukan input certification.

## 7.7 Mengapa rangkaian ini menjadi `LaneKeeping`

Rangkaian dinyatakan sebagai `LaneKeeping` karena:

1. perubahan lateral/heading mengubah keputusan;
2. action factual menghasilkan koreksi lane yang lebih stabil daripada foil;
3. lane symmetry dapat diperiksa;
4. koreksi bertahan secara temporal sampai error pulih;
5. cluster melewati support, held-out, bootstrap, outcome, property, dan traceability gate.

Human-readable explanation:

> Policy melakukan `LaneKeeping` karena lateral dan heading error memerlukan koreksi. Jika foil dipilih, steering jerk meningkat dan pemulihan lane memburuk. Koreksi dipertahankan sampai error menyusut, kemudian policy kembali cruise. Cluster ini berstatus `CERTIFIED_PRIMITIVE`.

---

## 8. Bukti primitive 2 — `StopCompliance`

## 8.1 Definisi manusia

`StopCompliance` adalah rangkaian keputusan yang merespons kewajiban stop: mendekati stop line, menurunkan kecepatan, berhenti, mempertahankan full stop, lalu melanjutkan setelah kewajiban terpenuhi.

Temporal arc:

```text
Approach
    → Decelerate
    → Full stop
    → Hold
    → Stop obligation satisfied
    → Resume
```

## 8.2 Why

Decision evidence utama:

- stop distance intervention;
- `stop_satisfied_flip`;
- peralihan dari move/decelerate menjadi brake/hold;
- no-resume sebelum stop obligation terpenuhi.

Contoh certified cluster `StopCompliance_C05`:

```text
support = 13 segments
solver support = Q-learning + SARSA
status = CERTIFIED_PRIMITIVE

decision_summary:
- lateral_flip = 1.000
- heading_flip = 1.000
- stop_distance_flip = 1.000
- stop_satisfied_abs_delta = 1.000
- duck_risk_abs_delta = 1.000
```

Contoh certified cluster `StopCompliance_C07`:

```text
support = 30 segments
solver support = SAC + TD3
status = CERTIFIED_PRIMITIVE

decision_summary:
- lateral_flip = 1.000
- heading_flip = 1.000
- speed_flip = 1.000
- curvature_flip = 1.000
- stop_satisfied_abs_delta = 1.000
```

Walaupun beberapa flip bernilai jenuh, `stop_satisfied` tetap menjadi lever pembeda family. Angka jenuh tidak dibaca sendiri; ia dibaca bersama paired outcome, verification, temporal segment, dan certificate.

## 8.3 What-if

Contoh `StopCompliance_C07`:

```text
rollout_steps = 15
elapsed_seconds = 3.0
factual_cumulative_steering_jerk = 14.331
foil_cumulative_steering_jerk = 15.900
```

Untuk local stop-critical instance, outcome profile juga membandingkan:

- apakah ego berhenti sebelum stop line;
- minimum speed dan dwell time;
- apakah stop tracker terpenuhi;
- apakah foil melewati stop line;
- collision/off-road/termination;
- factual versus foil reward components.

Certified curve-lane hazard case menggunakan mekanisme yang sama: forcing `fast_straight` pada state curve yang sama dapat menghasilkan collision pada branch foil di bawah manifest/noise yang sama.

## 8.4 Verification

Relation utama:

```text
Selama stop_present = true dan stop_satisfied = false,
stop line dibuat lebih dekat
    ⇒ commanded speed tidak meningkat.
```

M7 Q-learning audit:

```text
stop relation: 6 pass, 0 violation
```

M8 supported near-stop audit:

```text
0 violation dari 4 applicable supported states
```

Certification cluster tetap memakai full per-member verification records, bukan hanya aggregate M7/M8 lama.

## 8.5 Temporal evidence

Contoh `StopCompliance_C07`:

```text
previous_distance = 4.525
continuity = 1.000
persistence = 0.993
decision_change = 0.420
```

Human-readable rendering:

```text
Stop line becomes relevant
    → velocity is reduced during approach
    → ego reaches near-zero velocity
    → braking is held for the required dwell
    → stop tracker changes to satisfied
    → velocity is restored and driving resumes
```

## 8.6 Presentation overlay

Macro stop result:

```text
30 StopSign macro instances
631 decisions
all four policies represented

v_start = 0.156
v_min   = 0.003
v_end   = 0.195
brake_fraction = 0.213
stop relation pass = 1.000 on 23 applicable decisions
```

Human arc:

```text
Approach at 0.156
    → decelerate
    → near-zero stop at 0.003
    → hold
    → resume to 0.195
```

## 8.7 Mengapa rangkaian ini menjadi `StopCompliance`

Rangkaian dinyatakan sebagai `StopCompliance` karena:

1. stop distance/status merupakan decision lever;
2. factual trajectory memenuhi full-stop behavior;
3. foil trajectory menunjukkan akibat alternatif;
4. stop monotonicity dan no-premature-resume dapat diverifikasi;
5. urutan temporal lengkap terlihat: approach sampai resume;
6. terdapat certified cluster untuk policy tabular dan continuous.

Human-readable explanation:

> Policy menjalankan `StopCompliance` karena stop obligation belum terpenuhi dan stop line berada pada jarak relevan. Policy menurunkan kecepatan, berhenti hingga dwell terpenuhi, lalu melanjutkan. Paired rollout menunjukkan konsekuensi aksi alternatif pada kondisi awal yang sama, sedangkan metamorphic verification memastikan kecepatan tidak meningkat ketika stop line dibuat lebih dekat. Cluster tabular dan continuous yang disebut di atas berstatus `CERTIFIED_PRIMITIVE`.

---

## 9. Bukti primitive 3 — `CurveNegotiation`

## 9.1 Definisi manusia

`CurveNegotiation` adalah rangkaian keputusan untuk mendeteksi kebutuhan steering akibat curvature, mengurangi kecepatan bila perlu, memberi steering sesuai arah tikungan, mempertahankan lane, lalu menstabilkan kendaraan setelah tikungan.

Temporal arc:

```text
Detect curve
    → Reduce speed
    → Apply directional steering
    → Track curved lane
    → Recover heading
    → Stabilize
```

## 9.2 Why

Decision evidence:

- curvature intervention;
- heading/lateral response;
- steering direction;
- speed response terhadap curvature;
- sustained decision change sepanjang curve segment.

Contoh certified clusters:

| Cluster | Solver | Support | Status |
|---|---|---:|---|
| `CurveNegotiation_C13` | Q-learning, SARSA | 13 | `CERTIFIED_PRIMITIVE` |
| `CurveNegotiation_C14` | Q-learning, SARSA | 17 | `CERTIFIED_PRIMITIVE` |
| `CurveNegotiation_C21` | Q-learning, SARSA | 15 | `CERTIFIED_PRIMITIVE` |
| `CurveNegotiation_C22` | Q-learning, SARSA | 14 | `CERTIFIED_PRIMITIVE` |

Contoh `CurveNegotiation_C14`:

```text
decision_summary:
- lateral_flip = 1.000
- heading_flip = 1.000
- speed_abs_delta = 1.000
- stop_distance_flip = 1.000
- stop_distance_abs_delta = 1.000
```

Descriptor top features dapat memuat fitur lain yang juga berubah pada cluster. Family assignment dibaca bersama sustained steering, curvature-related outcome, temporal structure, dan exclusion dari stop/pedestrian/lane-recovery family.

## 9.3 What-if

Contoh `CurveNegotiation_C14`:

```text
factual_cumulative_steering_jerk = 24.377
foil_cumulative_steering_jerk = 26.598
factual_brake_steps = 5
```

Local hazard case:

```text
factual:
policy mengurangi speed dan steering mengikuti curve

foil:
first action dipaksa fast-straight

outcome:
foil mengalami immediate collision/lane failure
di bawah initial state dan exogenous noise yang sama
```

Ini adalah contoh paling langsung bahwa primitive bukan sekadar nama “belok”, tetapi ringkasan keputusan yang memiliki outcome safety contrast.

## 9.4 Verification

Relation utama:

```text
Dengan kondisi lain tetap dan precondition lane valid,
|curvature| meningkat
    ⇒ commanded speed tidak meningkat melebihi toleransi.
```

Audit M7 awal:

- Q-learning curvature: 6 pass, 0 violation;
- SAC curvature: 3 pass, 3 violation pada sample awal.

Perbedaan tersebut tidak disembunyikan. Continuous-policy cluster harus lolos certificate gate sendiri; kandidat yang belum stabil tetap `PRIMITIVE_CANDIDATE`.

## 9.5 Temporal evidence

Contoh certified curve cluster:

```text
continuity ≈ 1.000
persistence tinggi
steering outcome bertahan sepanjang segment
```

Human-readable rendering:

```text
Curvature becomes non-zero
    → policy reduces velocity
    → steering follows curve direction
    → lane/heading error remains controlled
    → steering decreases after curve exit
    → stable lane following resumes
```

## 9.6 Presentation overlay

Macro result:

```text
CurveNegotiationLeft instances = 95
decisions = 2,038
all four policies represented

v_start = 0.182
v_min   = 0.095
v_end   = 0.144
brake_fraction = 0.011
```

Human arc:

```text
Detect curve at 0.182
    → reduce to 0.095
    → steer through curve
    → recover to 0.144
```

## 9.7 Mengapa rangkaian ini menjadi `CurveNegotiation`

Rangkaian dinyatakan sebagai `CurveNegotiation` karena:

1. perubahan lane geometry memengaruhi keputusan;
2. factual action menghasilkan speed-steering profile yang mengikuti curve;
3. foil dapat meningkatkan jerk, lane departure, atau collision;
4. curvature-speed relation dapat diuji;
5. tindakan berlangsung sebagai arc detect–steer–recover, bukan satu steering frame;
6. beberapa cluster tabular lolos certification, sementara cluster continuous yang belum memenuhi semua gate tetap candidate.

Human-readable explanation:

> Policy menjalankan `CurveNegotiation` karena geometri lane membutuhkan steering dan penyesuaian kecepatan. Factual branch mempertahankan tracking curve, sedangkan forcing `fast_straight` dapat menghasilkan lane failure atau collision pada branch dengan kondisi awal dan noise identik. Primitive berakhir ketika heading pulih dan kendaraan kembali stabil.

---

## 10. Bukti primitive 4 — `PedestrianYield`

## 10.1 Definisi manusia

`PedestrianYield` adalah rangkaian keputusan untuk mendeteksi pedestrian yang relevan terhadap koridor ego, menurunkan kecepatan, berhenti atau menahan yield, menunggu corridor clear, lalu melanjutkan.

Temporal arc:

```text
Detect pedestrian
    → Assess corridor threat
    → Decelerate
    → Yield hold
    → Pedestrian clears corridor
    → Resume
```

## 10.2 Why

Decision evidence:

- duck risk intervention;
- pedestrian geometry/presence;
- factual minimum duck clearance;
- action/primitive flip ketika pedestrian dipindahkan;
- resume setelah threat clear.

Discovered cluster:

```text
PedestrianCrossing_C03
presentation name = PedestrianYield
support = 21 segments
solver support = SAC + TD3
status = PRIMITIVE_CANDIDATE
```

Contoh decision summary:

```text
lateral_flip = 1.000
heading_flip = 1.000
speed_flip = 1.000
curvature_flip = 1.000
stop_distance_abs_delta = 1.000
```

Family separation tidak ditentukan oleh daftar top-five saja. Descriptor memberi family pedestrian ketika paired physical outcome menunjukkan pedestrian clearance dalam threshold relevan dan explanation contains pedestrian-related lever/verification.

## 10.3 What-if

Contoh cluster:

```text
factual_cumulative_steering_jerk = 16.068
foil_cumulative_steering_jerk = 17.866
rollout_steps = 15
elapsed_seconds = 3.0
```

Outcome yang lebih relevan untuk pedestrian:

- minimum Duckie clearance;
- collision;
- time spent moving while crossing is active;
- yield hold duration;
- resume timing;
- primitive sequence.

## 10.4 Verification

Relation utama:

```text
Jika pedestrian threat dibuat lebih dekat/lebih berbahaya,
commanded speed tidak boleh meningkat.
```

Audit awal:

- Q-learning pedestrian relation: 6 pass, 0 violation;
- SAC pedestrian relation: 5 pass, 1 violation;
- M8 supported near-pedestrian states: 0 violation dari 10 applicable states.

Hasil C-EDP menunjukkan nuansa representation:

- Q-learning dan SARSA tetap sensitif terhadap duck;
- `duck_risk_flip` dan pedestrian relation hidup;
- tetapi pada explanation signature tabular, lane/heading steering flip sering jenuh dan menutupi duck sebagai dominant separator;
- SAC/TD3 mempunyai continuous clearance outcome sehingga yield moment lebih mudah dipisahkan sebagai cluster sendiri.

## 10.5 Temporal evidence

Contoh `PedestrianCrossing_C03`:

```text
previous_distance = 5.583
continuity = 1.000
persistence = 0.994
decision_change = 0.385
```

Human-readable rendering:

```text
Pedestrian becomes relevant to ego corridor
    → velocity decreases
    → ego reaches near-zero speed
    → yield is held while pedestrian remains active
    → crossing clears
    → policy resumes motion
```

## 10.6 Presentation overlay

Macro result:

```text
PedestrianCrossing instances = 22
decisions = 452
all four policies appear in raw state/action overlay

v_start = 0.148
v_min   = 0.010
v_end   = 0.176
brake_fraction = 0.281
```

Human arc:

```text
Detect at 0.148
    → decelerate
    → yield near 0.010
    → hold until clear
    → resume to 0.176
```

Perbedaan penting:

- raw-state macro overlay dapat menemukan pedestrian behavior pada semua empat policy;
- explanation-only discovery memisahkan `PedestrianYield` sebagai cluster khusus hanya pada SAC/TD3;
- karena itu hasil ini disebut **representation-dependent separability**, bukan kegagalan Q/SARSA memahami pedestrian.

## 10.7 Mengapa rangkaian ini menjadi `PedestrianYield`

Rangkaian dinyatakan sebagai family `PedestrianYield` karena:

1. pedestrian-related state/outcome menjadi trigger;
2. factual branch menurunkan speed dan mempertahankan clearance;
3. foil membandingkan akibat bergerak/merespons berbeda;
4. pedestrian-risk monotonicity dapat diuji;
5. keputusan membentuk arc detect–decelerate–hold–clear–resume.

Namun cluster saat ini berstatus `PRIMITIVE_CANDIDATE`, bukan certified.

Human-readable explanation:

> Policy menjalankan candidate `PedestrianYield` ketika pedestrian memasuki kondisi relevan bagi koridor ego. Factual trajectory menurunkan kecepatan dan mempertahankan yield hingga corridor clear; foil menunjukkan outcome alternatif. Pola temporal terpisah dengan jelas pada SAC dan TD3, tetapi cluster belum melewati seluruh certification gate sehingga tidak boleh disebut `CERTIFIED_PRIMITIVE`.

---

## 11. Trace M1–M13 untuk setiap primitive

Tabel berikut menunjukkan bagaimana setiap tahap berkontribusi pada empat primitive.

| Tahap | LaneKeeping | StopCompliance | CurveNegotiation | PedestrianYield |
|---|---|---|---|---|
| M1 | Canonical lane features/action | Canonical stop features | Canonical curvature features | Canonical duck geometry/risk |
| M2 | Lane correction vocabulary | Stop/decelerate/hold/resume vocabulary | Curve steering vocabulary | Yield/hold/resume vocabulary |
| M3 | Correct-until-recovered segment | Approach-to-resume segment | Enter-to-exit curve segment | Detect-to-clear segment |
| M4 | Same initial lane/RNG | Same stop scenario/RNG | Same curve scenario/RNG | Same controller/RNG; reactive duck allowed |
| M5 | Corrective action vs foil | Stop action vs proceed foil | Curve action vs fast-straight | Yield action vs proceed foil |
| M6 | Lateral/heading minimal flip | Stop distance/status flip | Curvature response sweep | Duck distance/risk flip |
| M7 | Lane symmetry | Stop monotonicity | Curvature-speed monotonicity | Pedestrian-risk monotonicity |
| M8 | Exact tabular lane audit | Exact supported near-stop audit | Exact discrete curve response | Exact supported near-pedestrian audit |
| M9 | SAC steering attribution | SAC velocity/stop attribution | SAC curvature/steering attribution | SAC duck/velocity attribution |
| M10 | Lane rules | Stop rules | Curve rules | Yield rules |
| M11 | Bottom-up lane cluster | Bottom-up stop cluster | Bottom-up curve cluster | Bottom-up pedestrian cluster |
| M12 | Cross-policy lane report | Cross-policy stop report | Cross-policy curve report | Cross-policy yield report |
| M13 | SARSA lane explanation | SARSA stop explanation | SARSA curve explanation | SARSA pedestrian sensitivity |
| C-EDP | Certified lane clusters | Certified tabular + continuous stop clusters | Certified tabular curve clusters | SAC/TD3 candidate cluster |

---

## 12. Daftar delapan certified clusters

| Certified cluster | Family | Solver support | Segment support |
|---|---|---|---:|
| `StopCompliance_C05` | StopCompliance | Q-learning, SARSA | 13 |
| `StopCompliance_C07` | StopCompliance | SAC, TD3 | 30 |
| `CurveNegotiation_C13` | CurveNegotiation | Q-learning, SARSA | 13 |
| `CurveNegotiation_C14` | CurveNegotiation | Q-learning, SARSA | 17 |
| `CurveNegotiation_C21` | CurveNegotiation | Q-learning, SARSA | 15 |
| `CurveNegotiation_C22` | CurveNegotiation | Q-learning, SARSA | 14 |
| `LaneKeeping_C16` | LaneKeeping | Q-learning, SARSA | 17 |
| `LaneKeeping_C17` | LaneKeeping | Q-learning, SARSA | 17 |

Interpretasi:

- Stop mempunyai certified cluster pada policy tabular dan continuous.
- LaneKeeping dan CurveNegotiation mempunyai certified cluster tabular pada run ini.
- Continuous lane/curve regularities tetap ditemukan, tetapi cluster terkait yang belum lolos seluruh gate dipertahankan sebagai candidate.
- PedestrianYield terpisah pada SAC/TD3, tetapi masih candidate.

---

## 13. Human-readable temporal arc renderer

Angka temporal tidak langsung disajikan kepada end user. Presentation layer mengubah evidence menjadi phase labels.

### 13.1 LaneKeeping

```text
IF lateral/heading deviation becomes relevant
THEN phase = MONITOR_DEVIATION

IF corrective steering is applied
THEN phase = CORRECT

IF |d| and |phi| decrease
THEN phase = CONVERGE

IF steering returns near zero and lane error is safe
THEN phase = CRUISE
```

Output:

```text
Monitor → Correct → Converge → Cruise
```

### 13.2 StopCompliance

```text
IF stop is present and unsatisfied
THEN phase = APPROACH

IF speed command decreases
THEN phase = DECELERATE

IF speed is below stop threshold
THEN phase = STOP

IF stop remains unsatisfied while speed stays near zero
THEN phase = HOLD

IF stop changes to satisfied and speed rises
THEN phase = RESUME
```

Output:

```text
Approach → Decelerate → Stop → Hold → Resume
```

### 13.3 CurveNegotiation

```text
IF |curvature| enters relevant range
THEN phase = DETECT_CURVE

IF velocity decreases
THEN phase = REDUCE_SPEED

IF steering follows curvature sign
THEN phase = STEER

IF lane/heading remain bounded through curve
THEN phase = TRACK

IF curvature and steering decrease after exit
THEN phase = STABILIZE
```

Output:

```text
Detect curve → Reduce speed → Steer → Track → Stabilize
```

### 13.4 PedestrianYield

```text
IF pedestrian becomes relevant to ego corridor
THEN phase = DETECT_PEDESTRIAN

IF velocity decreases
THEN phase = DECELERATE

IF velocity remains near zero while threat is active
THEN phase = YIELD_HOLD

IF pedestrian clears the corridor
THEN phase = CLEAR

IF velocity increases after clear
THEN phase = RESUME
```

Output:

```text
Detect pedestrian → Decelerate → Yield hold → Clear → Resume
```

Arc ini adalah deterministic rendering dari computed fields. Ia bukan cerita yang dihasilkan bebas oleh LLM.

---

## 14. Contoh output akhir yang mudah dibaca

### 14.1 StopCompliance

```json
{
  "primitive_family": "StopCompliance",
  "cluster_id": "StopCompliance_C07",
  "status": "CERTIFIED_PRIMITIVE",
  "why": {
    "trigger": "stop obligation is not yet satisfied",
    "decision_levers": {
      "stop_satisfied_abs_delta": 1.0,
      "speed_flip": 1.0
    }
  },
  "what_if": {
    "factual": "decelerate and stop",
    "foil": "continue with contrast action",
    "horizon_steps": 15,
    "factual_steering_jerk": 14.331,
    "foil_steering_jerk": 15.900
  },
  "verification": {
    "stop_monotonicity": "PASS",
    "premature_resume": false
  },
  "temporal_arc": [
    "Approach",
    "Decelerate",
    "Stop",
    "Hold",
    "Resume"
  ],
  "explanation": "The policy approaches the unsatisfied stop obligation, reduces speed to a full stop, holds until the obligation is satisfied, and then resumes. The paired foil has a less safe outcome under the same initial conditions."
}
```

### 14.2 CurveNegotiation

```json
{
  "primitive_family": "CurveNegotiation",
  "status": "CERTIFIED_PRIMITIVE",
  "why": {
    "trigger": "lane geometry requires sustained directional steering"
  },
  "what_if": {
    "factual": "reduce speed and track the curve",
    "foil": "force fast-straight",
    "foil_outcome": "immediate collision or lane failure"
  },
  "verification": {
    "paired_manifest": "identical",
    "exogenous_noise": "shared",
    "curvature_speed_relation": "PASS for the certified instance"
  },
  "temporal_arc": [
    "Detect curve",
    "Reduce speed",
    "Steer",
    "Track",
    "Stabilize"
  ]
}
```

### 14.3 PedestrianYield

```json
{
  "primitive_family": "PedestrianYield",
  "cluster_id": "PedestrianCrossing_C03",
  "status": "PRIMITIVE_CANDIDATE",
  "why": {
    "trigger": "pedestrian is relevant to the ego corridor"
  },
  "what_if": {
    "factual": "decelerate and hold",
    "foil": "continue moving",
    "compared_outcomes": [
      "minimum pedestrian clearance",
      "collision",
      "yield duration",
      "resume timing"
    ]
  },
  "verification": {
    "pedestrian_risk_monotonicity": "evaluated",
    "certification": "not all gates passed"
  },
  "temporal_arc": [
    "Detect pedestrian",
    "Decelerate",
    "Yield hold",
    "Clear",
    "Resume"
  ]
}
```

---

## 15. Family assignment versus certification

Alur status yang benar:

```text
Local explanations
    → temporal segment
    → explanation signature
    → discovered cluster or noise
    → family descriptor
    → certification gates
    → certified/candidate
    → frozen runtime assignment or UNKNOWN
```

Tidak benar:

```text
state + action → nama primitive → otomatis certified
```

Benar:

```text
Why evidence
+ factual/foil outcome
+ verification evidence
+ temporal continuity
+ cluster support
+ held-out reproduction
+ bootstrap stability
+ outcome coherence
+ traceability
= certification decision
```

---

## 16. Batas interpretasi

1. `Q-margin` bukan probabilitas.
2. IG bukan causal proof.
3. Immediate reward bukan alasan lengkap policy memilih aksi.
4. Tree surrogate bukan policy asli.
5. Top-five descriptor features bukan keseluruhan alasan family.
6. Raw macro overlay adalah presentation evidence, bukan certification input.
7. `PRIMITIVE_CANDIDATE` tidak boleh disebut certified.
8. `UNKNOWN` tidak boleh diberi narasi seolah-olah policy behavior telah terbukti.
9. Pedestrian separation berbeda antar-representation dan harus dilaporkan apa adanya.
10. Klaim counterfactual dibatasi pada simulator-based intervention di bawah manifest yang dikontrol.

---

## 17. Cara membaca hasil

Untuk memeriksa satu primitive:

1. buka [`primitive_catalogue.md`](../runs/explanations/cedp_v2_4policy/primitive_catalogue.md);
2. pilih cluster;
3. lihat `decision_summary` untuk Why;
4. lihat `outcome_summary` untuk What-if;
5. lihat `verification_summary` untuk property evidence;
6. lihat `temporal_summary` untuk continuity/persistence;
7. cocokkan dengan [`primitive_certificates.jsonl`](../runs/explanations/cedp_v2_4policy/primitive_certificates.jsonl);
8. pastikan status;
9. lihat runtime coverage di [`runtime_assignments.json`](../runs/explanations/cedp_v2_4policy/runtime_assignments.json);
10. gunakan temporal arc renderer untuk presentation.

Untuk audit keseluruhan:

```text
discovery_summary.json
    → berapa cluster/support/noise

certification_summary.json
    → berapa certified/candidate

primitive_certificates.jsonl
    → gate per cluster

runtime_assignments.json
    → certified assignment versus UNKNOWN

cedp_v2_report.json
    → consolidated experiment report
```

---

## 18. Validasi codebase

Test suite terakhir:

```text
179 tests passed
1 third-party SciPy deprecation warning
```

Warning tersebut bukan kegagalan experiment. Yang menentukan adalah assertion dan test result.

Aspek yang telah diuji:

- schema/adapters;
- state validity;
- deterministic replay;
- paired outcome;
- metamorphic relations;
- Q-table exact checks;
- SAC diagnostics;
- rule extraction;
- clustering/reconciliation;
- C-EDP collection, segmentation, discovery, certification, dan runtime.

---

## 19. Kesimpulan

Pipeline memenuhi tujuan utama: explanation M1–M13 tidak berhenti sebagai kumpulan angka per-state. Explanation tersebut:

1. disusun menjadi temporal segment;
2. direpresentasikan dengan explanation-only signature;
3. ditemukan regularitasnya;
4. diberi vocabulary driving primitive;
5. diuji dengan certification gates;
6. dipresentasikan sebagai:

```text
Why
    + What-if
    + Verification
    + Human-readable temporal arc
    + Certification status
```

Empat family yang dihasilkan adalah:

```text
LaneKeeping
StopCompliance
CurveNegotiation
PedestrianYield
```

Contoh arc paling penting:

```text
LaneKeeping:
Monitor → Correct → Converge → Cruise

StopCompliance:
Approach → Decelerate → Stop → Hold → Resume

CurveNegotiation:
Detect curve → Reduce speed → Steer → Track → Stabilize

PedestrianYield:
Detect pedestrian → Decelerate → Yield hold → Clear → Resume
```

Jadi driving primitive di sini bukan label langsung dari `(s,a)`. Ia adalah ringkasan temporal yang diturunkan dari decision explanation, counterfactual outcome, verification evidence, dan certification status.
