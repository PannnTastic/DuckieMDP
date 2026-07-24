# Plan Explanation-Derived Driving Primitives

## 0. Status dan keputusan utama

Status dokumen: **pilot EDDP v1 telah dieksekusi dan divalidasi**.

Dokumen ini dipertahankan sebagai rekaman desain dan hasil pilot v1. Rencana
implementasi utama berikutnya adalah
[certified_explanation_derived_primitives_plan.md](certified_explanation_derived_primitives_plan.md),
yang hanya memakai explanation instance berstatus CERTIFIED, membentuk
primitive dari rangkaian explanation temporal, dan menambahkan certificate
tingkat primitive.

Hasil dan keputusan ilmiah tersedia di
[`explanation_derived_driving_primitives_results.md`](explanation_derived_driving_primitives_results.md),
sedangkan gate machine-readable ada di
[`eddp_v1_acceptance.json`](eddp_v1_acceptance.json). Keputusan pilot adalah
`GO_WITH_LIMITED_CLAIM`: lima cluster menjadi kandidat lintas Q-learning/SARSA,
sementara satu cluster SAC tetap dilabeli solver-specific behavior.

Dokumen ini memperluas pipeline explainable MDP yang sudah selesai. Eksperimen
M1--M13, primitive lexicon M2, hasil counterfactual, serta checkpoint policy
lama tidak ditimpa. Semuanya dipertahankan sebagai baseline dan provenance.

Gagasan utamanya adalah membalik hubungan lama:

```text
Pipeline lama:
state + action -> primitive buatan manusia -> explanation

Pipeline baru:
state + action -> explanation -> pola explanation berulang
               -> emergent driving primitive
```

Nama kerja yang digunakan dalam dokumen ini adalah:

> **Explanation-Derived Driving Primitives (EDDP)**

Nama tersebut bersifat deskriptif, bukan klaim bahwa EDDP merupakan nama
metode baku dalam literatur.

Primitive lama dari M2 disebut **top-down primitives**. Primitive baru yang
ditemukan dari explanation disebut **explanation-derived primitives**.

---

## 1. Motivasi penelitian

Primitive yang ditentukan dengan aturan state--action dapat membedakan aksi
yang tampak berbeda, tetapi satu aksi yang sama dapat mempunyai fungsi yang
berbeda. Contohnya, `brake` dapat berarti:

- mematuhi stop sign;
- memberi jalan kepada pedestrian;
- menghindari keluar lane;
- berhenti secara tidak perlu.

Karena itu, primitive seharusnya tidak hanya menjawab **apa action-nya**, tetapi
juga:

1. kondisi apa yang membuat keputusan berubah;
2. konsekuensi apa yang dicegah atau diperoleh;
3. properti keselamatan apa yang dipertahankan;
4. bagaimana perilaku berkembang selama beberapa langkah.

EDDP mendefinisikan primitive berdasarkan kesamaan alasan dan konsekuensi
keputusan, bukan hanya kesamaan action mentah.

### 1.1 Two-sentence pitch

Policy driving sering diringkas memakai primitive yang ditentukan manual,
padahal action identik dapat merepresentasikan maksud keselamatan yang berbeda.
Kami menemukan driving primitives dari pola counterfactual state, paired
action-outcome, dan verification yang berulang sehingga primitive merepresentasikan
fungsi temporal keputusan policy.

### 1.2 Kontribusi yang diuji

Kontribusi yang diusulkan bukan clustering saja, melainkan komposisi berikut:

1. mengubah hasil explanation menjadi signature solver-neutral;
2. menemukan kelompok signature tanpa primitive label;
3. membentuk primitive temporal dari kelompok tersebut;
4. membandingkan primitive emergent lintas Q-learning, SARSA, dan SAC;
5. merekonsiliasinya dengan leksikon top-down yang sudah dibekukan.

---

## 2. Pertanyaan penelitian dan hipotesis

### RQ1 — Emergence

Apakah local explanations yang berulang membentuk cluster yang stabil dan
berbeda secara outcome?

**H1:** explanation signatures membentuk lebih dari satu cluster stabil dengan
outcome fisik yang berbeda.

### RQ2 — Solver invariance

Apakah keputusan Q-learning, SARSA, dan SAC yang mempunyai fungsi berkendara
sama dapat masuk ke primitive emergent yang sama?

**H2:** sebagian cluster mempunyai anggota dari lebih dari satu solver walaupun
identitas solver tidak digunakan sebagai fitur.

### RQ3 — Added semantic value

Apakah explanation-derived primitives memberikan pemisahan yang lebih bermakna
daripada action atau primitive top-down saja?

**H3:** satu top-down primitive dapat terpecah menjadi beberapa primitive dengan
konsekuensi berbeda, khususnya untuk `brake`, stop, dan pedestrian.

### RQ4 — Failure discovery

Apakah pipeline menemukan primitive kegagalan yang tidak dirancang sebelumnya?

**H4:** cluster tertentu mengisolasi perilaku seperti unnecessary holding,
premature resume, unsafe proceeding, atau oscillatory correction.

Kegagalan H1--H4 adalah hasil empiris, bukan kegagalan engineering. Pipeline
tetap dinilai benar bila seluruh acceptance gate teknis lulus.

---

## 3. Batas eksperimen

### 3.1 Termasuk dalam scope

- checkpoint Q-learning, SARSA, dan SAC yang sudah diaudit;
- deterministic evaluation policy;
- state counterfactual dari M6;
- paired action-outcome rollout dari M4--M5;
- metamorphic dan safety results dari M7--M8;
- physical outcome dan reward profile;
- clustering dan held-out reconciliation;
- primitive naming setelah cluster dibekukan.

### 3.2 Tidak termasuk dalam scope

- retraining policy;
- menjadikan primitive sebagai input policy;
- visuomotor/POMDP;
- LLM sebagai sumber explanation;
- real-world causal claim;
- optimasi cluster menggunakan primitive M2.

### 3.3 Policy target

Manifest harus menunjuk checkpoint dan config persis yang sudah lolos audit
M12/M13. Tidak boleh mengganti checkpoint berdasarkan hasil EDDP.

- Q-learning dan SARSA: greedy evaluation, teacher nonaktif;
- SAC: deterministic actor mean, sampling actor nonaktif.

---

## 4. Unit analisis

EDDP membedakan tiga objek.

### 4.1 Decision anchor

Decision anchor adalah state nyata dari rollout evaluation:

$$
x_i=(s_i,a_i,\mathcal{M}_i),
$$

dengan $\mathcal{M}_i$ sebagai provenance scenario, seed, episode, decision
step, checkpoint, dan config. Provenance tidak digunakan sebagai fitur.

### 4.2 Explanation atom

Untuk setiap anchor, pipeline menghasilkan:

$$
e_i = [c_i, o_i, v_i, t_i],
$$

dengan:

- $c_i$: state-counterfactual profile;
- $o_i$: paired action-outcome profile;
- $v_i$: metamorphic dan safety profile;
- $t_i$: temporal response profile.

Satu explanation atom belum disebut driving primitive.

### 4.3 Temporal explanation segment

Explanation atom berdekatan digabungkan bila pola explanation-nya stabil:

$$
E_j=(e_b,e_{b+1},\ldots,e_{b+L-1}).
$$

Segment dibentuk menggunakan fixed windows dan/atau change-point pada
explanation signature. Batas segment dari primitive M2 dilarang digunakan.

### 4.4 Explanation-derived primitive

Cluster $C_k$ baru layak disebut kandidat primitive bila:

1. mempunyai dukungan data memadai;
2. stabil pada resampling dan seed held-out;
3. mempunyai physical-outcome profile yang kohesif;
4. dapat dijelaskan menggunakan representative cases;
5. tidak terbentuk hanya karena identitas solver;
6. menunjukkan struktur temporal, bukan satu action sesaat saja.

---

## 5. Counterfactual explanation yang digunakan

### 5.1 State counterfactual

Untuk policy $\pi$, dicari perubahan state valid minimum:

$$
\delta_i^* = \arg\min_{\delta} \lVert W\delta\rVert
$$

dengan syarat:

$$
\pi(s_i+\delta) \neq \pi(s_i)
$$

untuk Q-learning/SARSA, atau perubahan command melewati tolerance yang
dibekukan untuk SAC.

Signature menyimpan perubahan per konsep, bukan kalimat natural-language:

```text
delta_lateral
delta_heading
delta_speed
delta_curvature
delta_stop_distance
delta_stop_satisfied
delta_duck_longitudinal
delta_duck_lateral
counterfactual_distance
counterfactual_valid
```

### 5.2 Action-outcome counterfactual

Pada state dan kondisi awal yang sama:

$$
\tau_i^*=\operatorname{Rollout}(s_i,a_i^*,\pi),
$$

$$
\tau_i^{cf}=\operatorname{Rollout}(s_i,a_i^{cf},\pi).
$$

Setelah action pertama, kedua cabang kembali mengikuti policy evaluation yang
sama. Yang dikontrol adalah RNG stream, kondisi awal, parameter controller,
clock, dan prefix action. Reaksi Duckie terhadap trajektori ego tetap endogen
dan menjadi bagian sah dari outcome.

### 5.3 Foil protocol tanpa primitive label

Pemilihan foil harus deterministik dan tidak memakai primitive M2.

Q-learning dan SARSA:

1. second-best table action;
2. seluruh enam action alternatif sebagai audit;
3. hasil agregat best-safe, worst-safety, dan nearest-return foil.

SAC:

1. canonical action lattice yang dibekukan;
2. nearest command yang melewati perubahan kecepatan/steering tolerance;
3. actor-supported alternatives dari replay support;
4. hard brake sebagai safety probe terpisah.

Foil yang digunakan dalam main result dipilih oleh aturan sebelum outcome
dilihat. Foil lain disimpan sebagai sensitivity analysis.

### 5.4 Horizon

Fixed horizons:

$$
h\in\{1,5,10,20,30\}.
$$

Event-aligned horizons berakhir saat salah satu kondisi berikut terjadi:

- lane kembali stabil;
- stop terpenuhi atau stop line terlewati;
- Duckie meninggalkan crossing;
- collision atau lane departure;
- episode terminal;
- maximum horizon tercapai.

---

## 6. Explanation signature

### 6.1 Main signature: solver-neutral

Fitur utama hanya berasal dari besaran yang mempunyai arti sama bagi seluruh
solver.

#### A. State-counterfactual profile

- normalized minimal change per semantic state concept;
- jumlah dimensi yang harus berubah;
- counterfactual validity;
- distance to decision boundary.

#### B. Physical action-outcome difference

- delta progress;
- delta mean/max absolute lateral error;
- delta mean/max absolute heading error;
- delta minimum Duckie clearance;
- delta time-to-collision bila valid;
- delta stop duration dan stop violation;
- delta lane departure dan collision;
- delta steering reversal dan jerk;
- delta termination reason;
- time-to-outcome pada event-aligned horizon.

#### C. Verification profile

- safety-property pass/fail/not-applicable;
- metamorphic relation pass/fail/not-applicable;
- jumlah dan severity violation;
- validity dan support stratum.

#### D. Temporal profile

- perubahan speed sepanjang horizon;
- perubahan steering sign dan magnitude;
- recovery time;
- hold duration;
- resume latency;
- outcome onset step.

### 6.2 Reward profile

Perbedaan cumulative reward per komponen disimpan, tetapi bukan satu-satunya
dasar primitive. Dua konfigurasi diuji:

1. **Physical-only signature** sebagai hasil utama;
2. **Physical + reward signature** sebagai ablation.

Ini mencegah reward shaping otomatis dianggap sebagai outcome manusia.

### 6.3 Fitur yang dilarang

Fitur berikut tidak boleh masuk clustering utama:

```text
solver
policy/checkpoint name
seed/episode/scenario ID
action_id atau action_name
primitive M2
trigger M2
natural-language explanation
cluster name
Q-margin
raw Q-values
Integrated Gradients
critic values
reward total tanpa decomposition
```

Q-margin, IG, dan critic probe tetap disimpan untuk menjelaskan cluster setelah
discovery, tetapi tidak dipakai membentuk cluster karena tidak sebanding lintas
solver.

### 6.4 Normalisasi

- scaler hanya di-fit pada development split;
- binary dan not-applicable mask dipisahkan dari nilai numerik;
- distance/velocity dinormalisasi dengan batas config yang dibekukan;
- clipping harus dicatat dalam manifest;
- tidak ada preprocessing yang dipilih menggunakan primitive label.

---

## 7. Sampling dan dataset

### 7.1 Sumber data

Anchor berasal dari rollout evaluation baru menggunakan checkpoint yang sama,
bukan dari training replay buffer. Sampling tidak menggunakan primitive label.

### 7.2 Stratifikasi fisik tanpa primitive label

Untuk menghindari kegagalan M11 lama pada stop/yield yang langka, sampling
diimbangi berdasarkan kondisi simulator yang dapat diobservasi:

1. lane-only dan curvature;
2. stop absent/present, distance band, satisfied flag;
3. Duckie absent/present/active dan relative-position band;
4. boundary/failure proximity;
5. normal cruise region.

Ini merupakan scenario coverage, bukan pemberian label primitive.

### 7.3 Pilot budget

Pilot minimum:

- 3 solver;
- 5 seed per solver;
- 4 physical context groups;
- minimal 20 anchor per solver--seed--context bila tersedia;
- maksimal 2 primary foil per anchor;
- horizon maksimum 30 decision steps.

Target kasar pilot adalah 1.000--1.200 anchor. Bila context tertentu tidak
tersedia, kekurangan dilaporkan; tidak boleh mengarang state sintetis tanpa
manifold validation.

### 7.4 Split

- development seed untuk scaler, clustering search, dan naming examples;
- held-out seed untuk generalization metrics;
- optional held-out scenario untuk transfer;
- tidak ada anchor dari episode yang sama pada dua split.

---

## 8. Temporal segmentation

Dua mekanisme diuji.

### 8.1 Fixed-window baseline

Non-overlapping atau sliding window dengan panjang:

$$
L\in\{3,5,10\}.
$$

Baseline utama menggunakan $L=5$ agar sebanding dengan M11 lama.

### 8.2 Explanation change-point

Segment baru dimulai ketika jarak antar explanation atom melewati threshold
yang dipilih pada development data tanpa primitive label.

### 8.3 Segment aggregation

Setiap segment menyimpan:

- mean, std, minimum, maksimum;
- slope fitur temporal;
- onset dan recovery time;
- fraction safety/metamorphic pass;
- factual dan counterfactual trajectory summaries.

Fixed-window menjadi baseline; change-point dianggap lebih baik hanya bila
stability dan outcome coherence meningkat pada held-out data.

---

## 9. Discovery dan clustering

### 9.1 Metode utama

HDBSCAN tetap menjadi metode utama karena:

- jumlah primitive tidak dipaksakan;
- rare/unsupported behavior dapat menjadi noise;
- bentuk cluster tidak harus spherical.

Hyperparameter dipilih dengan objective unsupervised pada development split:

- silhouette;
- coverage;
- cluster-size regularity;
- bootstrap stability.

Primitive M2 tidak digunakan dalam objective.

### 9.2 Sensitivity methods

- K-means sebagai forced-partition sensitivity;
- hierarchical agglomerative clustering;
- physical-only versus physical+reward features;
- fixed-window versus change-point segment;
- dengan dan tanpa state-context features.

### 9.3 Assignment held-out

Metode utama harus mempunyai aturan assignment held-out yang eksplisit.
Apabila implementasi HDBSCAN tidak menyediakan `predict`, gunakan salah satu:

1. implementation yang mendukung approximate prediction;
2. classifier assignment yang dilatih hanya dari development cluster;
3. transductive analysis yang dilabeli jujur dan dipisahkan dari hasil
   inductive held-out.

Main generalization claim harus berasal dari assignment inductive.

---

## 10. Dari cluster menjadi driving primitive

### 10.1 Cluster freeze

Sebelum melihat primitive M2:

1. feature manifest dibekukan;
2. scaler dan cluster model disimpan;
3. cluster assignments disimpan;
4. hash seluruh artefak dicatat;
5. representative factual/counterfactual cases dipilih berdasarkan jarak ke
   medoid, bukan karena terlihat menarik.

### 10.2 Primitive eligibility

Cluster diberi status:

- `PRIMITIVE_CANDIDATE`;
- `FAILURE_MODE_CANDIDATE`;
- `INSUFFICIENT_SUPPORT`;
- `SOLVER_SPECIFIC_BEHAVIOR`;
- `NOISE_OR_TRANSITION`.

Tidak semua cluster harus dipaksa menjadi primitive.

### 10.3 Naming protocol

Nama diberikan setelah freeze berdasarkan cluster card:

```text
Observed control tendency
State change that flips decision
Physical outcome protected or pursued
Safety/metamorphic relation
Temporal onset and recovery
Representative and boundary cases
```

Nama dianjurkan berbentuk fungsi, misalnya:

```text
LaneRecoveryLeft
StopComplianceHold
PedestrianSafetyHold
PostYieldResume
UnnecessarySafetyHold
OscillatoryLaneRecovery
UnsafeCrossingProceed
```

Nama tidak boleh hanya berupa action seperti `Brake` atau `SlowLeft`.

Jika memungkinkan, dua annotator memberi nama secara independen. Perbedaan
diselesaikan setelah agreement awal dilaporkan.

### 10.4 Primitive classifier opsional

Setelah primitive dibekukan, classifier atau rule list dapat dilatih untuk
memberi assignment baru secara cepat. Classifier tersebut adalah approximation
terhadap cluster assignment, bukan sumber discovery.

---

## 11. Rekonsiliasi dengan primitive top-down M2

Primitive M2 dibuka hanya setelah cluster dan nama sementara dibekukan.

Analisis rekonsiliasi meliputi:

- confusion matrix cluster x M2 primitive;
- purity, NMI, dan ARI sebagai deskripsi, bukan objective;
- split: satu M2 primitive menjadi beberapa EDDP;
- merge: beberapa M2 primitive menjadi satu EDDP;
- novel: EDDP tanpa padanan kuat;
- missing: M2 primitive yang selalu noise;
- comparison per solver dan held-out seed.

Contoh interpretasi yang sah:

```text
M2 YieldHold terpecah menjadi PedestrianSafetyHold dan
PersistentDuckOverreaction karena paired outcomes menunjukkan fungsi yang
berbeda walaupun factual action sama-sama brake.
```

Primitive M2 tetap berguna sebagai human-domain hypothesis. Alignment rendah
tidak otomatis berarti EDDP salah atau M2 salah; representative outcomes harus
diaudit.

---

## 12. Validasi

### 12.1 Engineering acceptance gates

Gate berikut bersifat wajib:

1. deterministic replay M4 tetap lulus;
2. state counterfactual lulus manifold validator;
3. factual dan counterfactual branch mempunyai provenance lengkap;
4. label-leakage test lulus;
5. development dan held-out tidak overlap;
6. feature/scaler/model hash tersimpan;
7. hasil reproduktif pada seed yang sama;
8. unit, targeted, dan full regression tests lulus;
9. unsupported region tidak menghasilkan klaim natural-language seolah-olah
   behavior telah dipelajari.

### 12.2 Scientific evaluation metrics

#### Cluster quality

- number of clusters;
- coverage dan noise rate;
- silhouette;
- cluster-size distribution.

#### Stability

- bootstrap ARI/NMI;
- seed-to-seed consistency;
- fixed-window versus change-point agreement.

#### Outcome coherence

- within-cluster variance physical outcomes;
- between-cluster outcome separation;
- consistency factual-versus-foil conclusion.

#### Cross-solver behavior

- solver composition per cluster;
- number of multi-solver clusters;
- predictability solver dari main signature;
- held-out solver sensitivity bila jumlah data cukup.

#### External reconciliation

- M2 purity/NMI/ARI;
- split/merge/novel/missing counts;
- primitive-name agreement.

Tidak ada threshold alignment M2 sebagai engineering gate. Alignment adalah
hasil penelitian.

### 12.3 Minimum evidence untuk memberi nama primitive

Sebuah cluster hanya boleh masuk main-results primitive bila:

- support minimum dibekukan sebelum melihat label;
- muncul pada lebih dari satu seed;
- held-out assignment tersedia;
- representative dan boundary examples valid;
- outcome coherence lebih baik daripada random/permuted baseline;
- nama dapat dijelaskan dari computed fields tanpa spekulasi.

Cluster yang gagal tetap dilaporkan sebagai noise, transition, atau insufficient
support.

---

## 13. Ablation study

Wajib:

1. physical-only versus physical+reward;
2. tanpa state counterfactual;
3. tanpa paired outcome;
4. tanpa metamorphic/safety profile;
5. fixed-window versus explanation change-point;
6. pooled solver versus cluster per solver;
7. HDBSCAN versus K-means;
8. balanced physical-context sampling versus rollout-natural frequency.

Ablation menjawab komponen explanation mana yang benar-benar membuat primitive
lebih stabil dan semantik.

---

## 14. Milestone eksekusi

Milestone baru diberi prefix `EDP` agar tidak menimpa M1--M13 lama.

### EDP0 — Freeze provenance

Output:

- exact checkpoint/config manifest;
- hash primitive lexicon M2;
- hash artefak M4--M13;
- dependency lock dan Git commit ID.

Gate: semua target policy dan existing explanation dapat direproduksi.

### EDP1 — Explanation dataset schema

Implementasikan schema anchor, foil, branch, outcome, validation mask, dan
provenance. Tidak ada clustering.

Gate: schema round-trip dan cross-solver contract test lulus.

### EDP2 — Label-free anchor collection

Jalankan rollout stratified secara fisik untuk tiga solver dan lima seed.

Gate: context coverage table tersedia; tidak ada sampling memakai primitive M2.

### EDP3 — Batched counterfactual generation

Jalankan state counterfactual dan paired action-outcome untuk setiap anchor.

Gate: replay determinism, branch validity, dan foil protocol lulus.

### EDP4 — Explanation signature builder

Buat physical-only dan physical+reward signature serta leakage guard.

Gate: mengubah solver name, primitive M2, action name, dan natural-language
text tidak mengubah main feature vector.

### EDP5 — Temporal segmentation

Bangun fixed-window baseline dan optional explanation change-point.

Gate: tidak menggunakan boundary primitive M2; segment deterministik.

### EDP6 — Unsupervised discovery

Fit scaler dan search clustering pada development split, lalu assign held-out.

Gate: label blindness, deterministic rerun, dan artifact hashing lulus.

### EDP7 — Cluster cards dan primitive eligibility

Bangun physical-outcome profiles, medoid cases, boundary cases, dan status
eligibility. Freeze sebelum membuka M2.

Gate: setiap klaim cluster dapat ditelusuri ke computed JSON/CSV.

### EDP8 — Naming

Beri nama fungsional pada eligible clusters dan catat rationale.

Gate: nama tidak menggunakan action saja dan agreement annotator dilaporkan
bila annotator kedua tersedia.

### EDP9 — M2 reconciliation

Buka frozen M2 labels dan hitung alignment, split, merge, novel, dan missing.

Gate: tidak ada refit clustering setelah label dibuka untuk main result.

### EDP10 — Cross-solver dan ablation

Uji kestabilan, shared clusters, solver-specific clusters, serta ablation.

Gate: development dan held-out selalu dipisahkan dalam tabel.

### EDP11 — Unified report

Hasil akhir:

- primitive catalogue;
- cluster cards;
- state-counterfactual plots;
- paired-outcome plots;
- primitive timeline;
- cross-solver comparison;
- failure-mode catalogue;
- machine-readable JSON/CSV;
- short explanation video per eligible primitive.

Gate: binding explanation-validity criteria lulus; hasil negatif tetap masuk
audit report.

---

## 15. Struktur kode yang direncanakan

```text
configs/explainability/
  eddp_v1.yaml

src/explainability/eddp/
  __init__.py
  schema.py
  anchor_sampler.py
  foil_protocol.py
  counterfactual_dataset.py
  signature.py
  temporal_segmenter.py
  cluster.py
  cluster_cards.py
  naming.py
  reconcile.py
  validate.py
  report.py

scripts/
  run_eddp_collect.py
  run_eddp_counterfactuals.py
  run_eddp_discovery.py
  run_eddp_reconciliation.py
  run_eddp_report.py

tests/
  test_eddp_schema.py
  test_eddp_label_leakage.py
  test_eddp_foil_protocol.py
  test_eddp_signature.py
  test_eddp_segmentation.py
  test_eddp_clustering.py
  test_eddp_reconciliation.py

runs/explanations/eddp_v1/
  provenance_manifest.json
  anchors.csv
  explanation_atoms.parquet
  temporal_segments.parquet
  signatures_unlabeled.parquet
  cluster_assignments.csv
  cluster_cards.json
  primitive_catalogue.json
  reconciliation.json
  ablations/
  figures/
```

`runs/` tetap artefak lokal/ignored kecuali artefak kecil yang dipilih untuk
reproducibility. Video dan dataset besar tidak dimasukkan Git.

---

## 16. Format primitive catalogue

Contoh record setelah cluster dibekukan dan diberi nama:

```json
{
  "primitive_id": "EDDP-04",
  "name": "PedestrianSafetyHold",
  "status": "PRIMITIVE_CANDIDATE",
  "support": {
    "segments": 84,
    "seeds": 5,
    "solvers": ["q_learning", "sarsa", "sac"]
  },
  "decision_flip": {
    "dominant_concept": "duck_longitudinal",
    "median_normalized_change": 0.31
  },
  "protected_outcome": {
    "metric": "minimum_duck_clearance",
    "median_factual_minus_foil": 0.24
  },
  "verification": {
    "pedestrian_safety_pass_rate": 0.98
  },
  "temporal_profile": {
    "median_hold_steps": 11,
    "median_resume_latency": 3
  },
  "m2_reconciliation": {
    "dominant_label": "YieldHold",
    "purity": 0.79
  }
}
```

Angka di atas hanya contoh schema, bukan hasil eksperimen.

---

## 17. Risiko dan mitigasi

### Risiko 1 — Circular explanation

Jika primitive M2 atau natural-language explanation masuk feature vector,
cluster hanya mengulang label lama.

Mitigasi: leakage guard, label mutation test, dan freeze sebelum reconciliation.

### Risiko 2 — Cluster hanya mengenali solver

M11 lama menunjukkan cluster yang solver-specific.

Mitigasi: physical outcome menjadi fitur utama; raw policy diagnostics dan
action identity dikeluarkan; laporkan solver composition dan solver
predictability.

### Risiko 3 — Rare safety primitive menjadi noise

Stop/yield langka pada rollout natural.

Mitigasi: stratifikasi berdasarkan kondisi fisik simulator tanpa primitive
label, serta support status yang jujur.

### Risiko 4 — Cluster bukan primitive temporal

Cluster satu-step dapat menjadi sekadar kondisi state.

Mitigasi: temporal aggregation, event-aligned outcome, duration/recovery
profile, dan primitive eligibility test.

### Risiko 5 — Counterfactual tidak valid

State sintetis atau branch tidak deterministik dapat menghasilkan pola palsu.

Mitigasi: manifold validator dan deterministic replay gate tetap binding.

### Risiko 6 — Reward shaping mendominasi makna

Mitigasi: physical-only sebagai main signature dan reward sebagai ablation.

### Risiko 7 — Nama cluster terlalu subjektif

Mitigasi: medoid-based cluster cards, functional naming template, optional
dual annotation, dan seluruh nama dapat ditelusuri ke computed outcomes.

---

## 18. Keputusan go/no-go bertahap

### Pilot go

Lanjut ke full experiment bila:

- counterfactual dataset dapat dibuat secara deterministik;
- stop dan pedestrian mempunyai support memadai;
- main signatures lolos leakage test;
- sedikitnya dua non-trivial clusters terbentuk;
- cluster cards dapat dijelaskan dari physical outcomes.

### Pilot revise

Revisi signature atau sampling bila:

- cluster terutama memisahkan solver;
- stop/yield tetap seluruhnya noise;
- clustering tidak stabil pada bootstrap;
- representative cases tidak mempunyai outcome yang kohesif.

### No-go claim

Jangan mengklaim discovery driving primitive bila cluster hanya merefleksikan:

- action identity;
- scenario ID;
- solver architecture;
- reward magnitude;
- primitive M2 yang bocor ke fitur.

Dalam kondisi tersebut hasil tetap dilaporkan sebagai batas pendekatan.

---

## 19. Kriteria selesai

Eksperimen EDDP selesai bila:

1. provenance policy lama tetap utuh;
2. explanation dataset tersedia untuk tiga solver;
3. main signatures terbukti bebas primitive/solver leakage;
4. temporal segmentation tidak memakai primitive M2;
5. cluster model dan assignment dapat direproduksi;
6. eligible clusters mempunyai cluster cards dan physical-outcome evidence;
7. M2 hanya digunakan setelah discovery freeze;
8. held-out dan cross-solver results dilaporkan;
9. seluruh ablation wajib selesai;
10. failure/noise/unsupported clusters tidak disembunyikan;
11. primitive catalogue dan unified report dapat dibuat ulang dari manifest;
12. unit, targeted, dan full regression tests lulus.

---

## 20. Urutan eksekusi yang direkomendasikan

```text
EDP0 provenance freeze
  -> EDP1 schema
  -> EDP2 label-free anchor collection
  -> EDP3 counterfactual generation
  -> EDP4 signature
  -> EDP5 temporal segmentation
  -> EDP6 clustering
  -> EDP7 cluster eligibility/freeze
  -> EDP8 naming
  -> EDP9 M2 reconciliation
  -> EDP10 cross-solver + ablation
  -> EDP11 report
```

Titik check-in pertama adalah setelah EDP4. Sebelum clustering dijalankan,
audit harus membuktikan bahwa signature benar-benar dibangun dari computed
explanations dan tidak membawa primitive label atau identitas solver.
