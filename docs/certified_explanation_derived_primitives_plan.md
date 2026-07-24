# Plan v2: Certified Explanation-Derived Driving Primitives

## 0. Status dan hubungan dengan eksperimen lama

Status: **implementasi CEDP0--CEDP10 selesai; pilot lima seed telah dieksekusi**.

Hasil implementasi dan batas klaim pilot dicatat di
[`certified_explanation_derived_primitives_results.md`](certified_explanation_derived_primitives_results.md).
Pilot memvalidasi pipeline, tetapi belum memenuhi budget eksperimen utama.


Plan ini tidak mengganti atau membatalkan:

- certified explanation pipeline M1--M13;
- primitive lexicon M2 yang telah dibekukan;
- EDDP v1 dan seluruh hasil pilotnya;
- checkpoint Q-learning, SARSA, dan SAC.

M1--M13 tetap menjadi sumber certificate tingkat keputusan. EDDP v1 tetap
menjadi feasibility pilot bahwa explanation signatures dapat dikelompokkan.
Plan v2 menambahkan integrasi yang belum ada: explanation yang telah
tervalidasi menjadi bahan pembentuk primitive, lalu primitive tersebut menerima
certificate tingkat primitive.

Nama kerja metode:

> **Certified Explanation-Derived Driving Primitives (C-EDDP)**

---

## 1. Masalah yang diselesaikan

### 1.1 Pipeline lama

Primitive lama dan explanation lama berjalan paralel:

```text
state + action ──> rule-based primitive M2
       │
       └─────────> certified explanation M1--M13
```

Keduanya membaca state dan action yang sama, tetapi primitive tidak dibentuk
dari explanation. Akibatnya, sebuah action seperti `brake` masih membutuhkan
aturan state--action manual untuk membedakan:

- stop-sign compliance;
- pedestrian yield;
- emergency recovery;
- unnecessary braking.

### 1.2 Pipeline baru

```text
state + action
      ↓
certified explanation instance
      ↓
temporal explanation signature
      ↓
label-free segmentation dan discovery
      ↓
evidence-grounded functional descriptor
      ↓
primitive-level certificate checker
      ↓
Certified Driving Primitive atau Unknown
```

Primitive tidak lagi sekadar nama untuk action. Primitive menjadi ringkasan
temporal dari alasan keputusan, konsekuensi foil, dan properti yang telah
diverifikasi.

---

## 2. Perubahan dari EDDP v1

| Bagian | EDDP v1 | C-EDDP v2 |
|---|---|---|
| Unit input | valid explanation atom pilot | hanya explanation berstatus `CERTIFIED` |
| Sampling | sparse three-step anchors | full contiguous explanation trajectory |
| Segmentasi utama | fixed window | explanation change-point; fixed window sebagai baseline |
| Fitur discovery | counterfactual, outcome, verification | sama, ditambah certificate mask dan temporal phase |
| Naming | sebagian memakai context composition | hanya memakai computed explanation evidence |
| Context `duck/stop/lane/nominal` | membantu naming | hanya sampling/audit metadata; dilarang menentukan nama |
| Primitive M2 | evaluasi setelah freeze | tetap evaluasi eksternal setelah freeze |
| Status cluster | candidate atau solver-specific behavior | certified, candidate, solver-specific, atau unknown |
| Certification | instance validity | instance certificate + primitive certificate |
| Runtime | cluster artefact | inductive primitive classifier dengan abstention |
| Human-readable output | cluster card | trigger + contrast + outcome + verification + temporal role |

Tidak ada refit cluster menggunakan label M2 atau nama primitive manusia.

---

## 3. Pertanyaan penelitian

### RQ1 — Compression

Dapatkah explanation M1--M13 yang berdimensi tinggi dikompresi menjadi
primitive temporal yang tetap mempertahankan trigger, outcome contrast, dan
verification evidence?

### RQ2 — Faithfulness

Apakah primitive yang ditemukan dapat ditelusuri kembali ke certificate
instance tanpa menambahkan klaim yang tidak dihitung?

### RQ3 — Temporal structure

Apakah explanation change-point menghasilkan primitive yang lebih stabil dan
koheren daripada fixed-window segmentation?

### RQ4 — Cross-policy structure

Primitive mana yang muncul lintas Q-learning, SARSA, SAC, dan kelak TD3, serta
primitive mana yang hanya merupakan solver-specific behavior?

### RQ5 — Human readability

Apakah primitive certificate lebih mudah dibaca daripada raw explanation
tables tanpa mengurangi factual correctness?

---

## 4. Hipotesis dan falsification criteria

### H1

Explanation-derived signatures memiliki outcome coherence lebih baik daripada
random/permuted assignment.

Falsified bila within-cluster outcome error tidak lebih rendah daripada
permuted baseline.

### H2

Change-point segmentation meningkatkan temporal purity atau boundary agreement
dibanding fixed window.

Falsified bila tidak ada peningkatan pada held-out seeds atau hasilnya tidak
stabil terhadap threshold.

### H3

Setidaknya satu certified primitive muncul pada lebih dari satu solver family.

Falsified bila semua cluster hanya mengenali solver.

### H4

Nama fungsional dapat dihasilkan dari explanation evidence tanpa memakai
action name, sampling context, solver, atau M2 label.

Falsified bila annotator tidak dapat memberi nama dari computed certificate
fields atau nama memerlukan raw state/action rule lama.

Hasil negatif tidak dibuang. Cluster dapat tetap berstatus
`SOLVER_SPECIFIC_PRIMITIVE` atau `UNKNOWN`.

---

## 5. Definisi formal

### 5.1 Certified explanation instance

Untuk keputusan pada waktu \(t\):

\[
\mathcal E_t =
(E_t^{\mathrm{decision}},
 E_t^{\mathrm{outcome}},
 E_t^{\mathrm{verification}},
 E_t^{\mathrm{provenance}}).
\]

Komponennya:

- `decision`: selected action, pre-registered foil, state counterfactual dan
  minimum action-changing intervention;
- `outcome`: factual/foil trajectory dan physical/reward deltas;
- `verification`: applicable/pass/fail untuk metamorphic dan safety relations;
- `provenance`: checkpoint/config/manifest hashes, seed, policy mode, dan
  teacher status.

Hanya record yang memenuhi binding criteria M1--M13 masuk discovery:

```text
counterfactual_valid
AND branch_invariants_pass
AND paired_outcome_valid
AND deterministic_policy_mode
AND teacher_inactive
AND supported_or_reachable_state
```

Record lain disimpan untuk audit, tetapi tidak boleh membentuk primitive.

### 5.2 Explanation signature

\[
z_t = \Phi(\mathcal E_t).
\]

`Phi` hanya memakai nilai yang dihitung dari certificate:

#### Decision block

- per-concept action-flip indicator;
- signed dan absolute intervention distance;
- minimum flip distance;
- number/fraction of valid interventions;
- boundary proximity.

#### Outcome block

- factual-versus-foil progress difference;
- lane dan heading error difference;
- pedestrian clearance difference;
- stop violations;
- collision/lane departure difference;
- braking duration;
- steering reversals dan jerk;
- termination difference.

#### Verification block

- relation applicable;
- relation pass/fail;
- safety-property pass/fail;
- certificate/branch validity masks.

#### Temporal phase block

- speed trend;
- steering trend;
- hold/decelerate/resume evidence;
- trigger persistence;
- outcome-risk trend;
- verification-state transition.

### 5.3 Fitur yang dilarang

Main discovery tidak boleh memakai:

- solver name;
- checkpoint ID;
- seed;
- raw action ID/name;
- raw Q-value, Q-margin, critic value, atau IG attribution;
- sampling context name;
- primitive M2;
- trigger string;
- natural-language explanation.

Raw state juga tidak dipakai langsung. State hanya memengaruhi primitive melalui
hasil counterfactual dan verification yang sudah dihitung.

### 5.4 Temporal explanation segment

Untuk interval kontinu \([i,j]\):

\[
Z_{i:j} = \operatorname{Aggregate}(z_i,\ldots,z_j).
\]

Aggregate minimal berisi mean, standard deviation, slope, start-to-end delta,
duration, extrema, dan certificate coverage.

### 5.5 Explanation-derived primitive

\[
P_k = \{Z_{i:j}: C(Z_{i:j})=k\},
\]

dengan \(C\) sebagai discovery model yang dilatih hanya pada development seeds.

---

## 6. Dua tingkat certification

### 6.1 Instance certificate

Sudah berasal dari M1--M13. Certificate menjamin bahwa explanation lokal
memenuhi validity contract dan semua klaimnya dapat ditelusuri ke artefak.

### 6.2 Primitive certificate

Satu cluster tidak otomatis certified hanya karena anggota-anggotanya valid.
Primitive certificate harus memuat:

```json
{
  "primitive_id": "...",
  "status": "CERTIFIED_PRIMITIVE",
  "cluster_freeze_hash": "...",
  "member_certificate_rate": 1.0,
  "support": 0,
  "seed_support": 0,
  "solver_support": [],
  "heldout_assignment_rate": 0.0,
  "bootstrap_stability": 0.0,
  "outcome_coherence_ratio": 0.0,
  "dominant_decision_evidence": {},
  "dominant_outcome_evidence": {},
  "verified_properties": {},
  "boundary_cases": [],
  "representative_certificates": []
}
```

### 6.3 Status akhir

```text
CERTIFIED_PRIMITIVE
PRIMITIVE_CANDIDATE
SOLVER_SPECIFIC_PRIMITIVE
UNKNOWN
```

`UNKNOWN` adalah output sah dan wajib digunakan di luar support region.

### 6.4 Gate certified primitive

Threshold pilot dibekukan sebelum melihat M2 labels:

- member certificate rate = 100%;
- support minimal 12 temporal segments;
- minimal 3 seed;
- held-out assignment tersedia;
- bootstrap cluster stability ARI minimal 0.70;
- outcome coherence ratio terhadap permutation maksimal 0.80;
- claimed verification property pass rate = 100% pada applicable members;
- representative dan boundary certificates valid;
- functional descriptor dapat diturunkan tanpa forbidden fields.

Cross-solver support bukan syarat semua primitive. Primitive satu solver dapat
certified tetapi harus berstatus `SOLVER_SPECIFIC_PRIMITIVE`.

Angka threshold ini adalah engineering convention, bukan theorem.

---

## 7. Temporal segmentation

### 7.1 Data utama

Rekam explanation pada seluruh keputusan episode:

```text
E_0, E_1, E_2, ..., E_T
```

Jangan hanya menyimpan anchor yang dipilih setelah melihat context.

### 7.2 Change-point utama

Boundary candidate muncul bila explanation distance berubah tajam:

\[
d_t = \|\tilde z_t-\tilde z_{t-1}\|_2.
\]

Gunakan PELT atau kernel change-point dengan penalty yang dipilih hanya pada
development seeds. Minimum segment duration mencegah fragmentasi satu-step.

### 7.3 Baseline

- fixed window L=3;
- fixed window L=5;
- action-change boundary;
- M2 primitive boundary hanya sebagai post-freeze oracle comparison.

### 7.4 Gate segmentasi

- setiap segment berisi keputusan yang benar-benar contiguous;
- tidak ada boundary M2 saat discovery;
- deterministik untuk config/seed yang sama;
- tidak menggabungkan gap akibat crash atau missing record;
- invalid record memutus segment dan menghasilkan `Unknown` interval.

---

## 8. Discovery dan held-out assignment

### 8.1 Split

Untuk setiap solver:

- seed 1--3: development;
- seed 4--5: held-out;
- tambahan seed hanya boleh masuk sesuai manifest sebelum fit.

### 8.2 Main method

- robust/standard scaling fit pada development;
- HDBSCAN untuk discovery dan explicit noise;
- cluster hyperparameter dipilih pada development;
- held-out assignment memakai frozen centroid/radius atau frozen classifier;
- tidak ada refit setelah M2 labels dibuka.

### 8.3 Sensitivity

- K-means;
- Gaussian mixture;
- per-solver clustering;
- pooled clustering;
- bootstrap resampling;
- feature-block ablation.

---

## 9. Evidence-grounded primitive naming

### 9.1 Nama tidak berasal dari context

`duck`, `stop`, `lane`, dan `nominal` hanya sampling/audit metadata. Nama
primitive tidak boleh ditentukan oleh mayoritas context.

### 9.2 Descriptor dari tiga pilar

Nama dibentuk dari:

\[
\operatorname{Name}(P_k)
=f(\text{trigger evidence},
   \text{outcome contrast},
   \text{verified property},
   \text{temporal role}).
\]

Contoh descriptor:

| Decision evidence | Outcome evidence | Verification | Temporal evidence | Candidate name |
|---|---|---|---|---|
| stop-distance flips action | foil increases violation | MR-STOP pass | decelerate-hold-resume | `StopCompliance` |
| duck-risk flips action | factual preserves clearance | MR-PED pass | yield-hold-resume | `PedestrianYield` |
| curvature flips steering | foil increases lane error | MR-CURVE pass | curve entry-turn-exit | `CurveFollowing` |
| lateral/heading intervention flips steering | factual reduces error | symmetry pass | diverge-correct-stabilize | `LaneRecovery` |
| no safety-changing foil | factual preserves progress | safety checks pass | sustained motion | `NominalCruise` |

Nama di atas contoh, bukan target label yang dimasukkan sebelum clustering.

### 9.3 Naming protocol

1. Cluster, assignments, medoid, dan certificate statistics dibekukan.
2. Generator membuat structured functional descriptor.
3. Dua annotator melihat descriptor tanpa solver/action/context/M2.
4. Annotator memberi nama dan rationale.
5. Agreement dan disagreement dicatat.
6. M2 baru dibuka untuk external reconciliation.

LLM, bila digunakan, hanya boleh memparafrase structured descriptor; LLM tidak
boleh menambah klaim atau menentukan status certificate.

---

## 10. Runtime primitive inference

Pada runtime:

```text
current certified explanation window
→ frozen signature builder
→ frozen scaler
→ frozen primitive assigner
→ support-radius check
→ primitive certificate lookup
→ CERTIFIED_PRIMITIVE atau UNKNOWN
```

Output human-readable:

```text
Primitive: PedestrianYield
Why: pedestrian-risk intervention changes proceed to brake.
What-if: the foil reduces minimum clearance by 0.24 m.
Verification: pedestrian monotonicity passes for all applicable members.
Temporal role: decelerate → hold → resume.
Evidence: 84 segments, 5 seeds, 4 policies.
```

Semua angka dan kalimat harus dapat ditelusuri ke member certificate IDs.

---

## 11. Experimental design

### 11.1 Policies

- greedy teacher-free Q-learning;
- greedy teacher-free SARSA;
- deterministic actor-mean SAC;
- deterministic TD3 setelah checkpoint tersedia.

TD3 tidak menjadi blocker pilot C-EDDP pada tiga policy, tetapi wajib sebelum
klaim paper yang menyebut TD3 dibekukan.

### 11.2 Context coverage

Setiap policy harus menghasilkan episode yang mencakup:

- nominal lane following;
- curve following;
- lane correction/recovery;
- stop approach/hold/resume;
- pedestrian approach/yield/resume;
- unsafe or failed behavior bila muncul secara alami.

Context digunakan untuk memastikan coverage, bukan sebagai clustering feature.

### 11.3 Minimum pilot budget

- 5 seed per solver;
- minimal 5 full episodes per seed;
- minimal 1,000 certified explanation instances per solver;
- seluruh episode ditulis, termasuk `Unknown` dan failure;
- no post-hoc seed removal.

### 11.4 Primitive comparison baseline

Bandingkan:

1. old direct state--action M2 primitive;
2. EDDP v1 sparse fixed-window candidate;
3. C-EDDP v2 full-trajectory certified primitive.

---

## 12. Evaluation

### 12.1 Engineering gates

- certificate schema round-trip;
- feature leakage tests;
- full-trajectory contiguity;
- deterministic segmentation;
- frozen discovery before M2;
- held-out assignment without refit;
- runtime abstention outside support;
- every primitive claim resolves to certificate IDs.

### 12.2 Scientific metrics

#### Compression

- primitive coverage;
- `Unknown` rate;
- compression ratio from explanation instances to segments;
- average primitive duration.

#### Cluster quality

- development/held-out silhouette;
- DBCV where available;
- bootstrap ARI;
- assignment stability.

#### Explanation preservation

- within-cluster decision-evidence variance;
- within-cluster outcome coherence;
- verification consistency;
- factual-versus-foil sign consistency.

#### Cross solver

- solver support per primitive;
- solver predictability from signatures;
- shared versus solver-specific primitives;
- held-out performance per solver.

#### External reconciliation

- purity, NMI, ARI against frozen M2;
- split/merge matrix;
- primitive M2 yang tidak ditemukan;
- C-EDDP primitive baru yang tidak memiliki M2 equivalent.

#### Readability

- agreement dua annotator atas functional descriptor;
- time-to-answer untuk raw explanation versus certified primitive;
- accuracy menjawab `why`, `what-if`, dan `verification`;
- optional human study untuk klaim human expectations.

---

## 13. Ablation wajib

1. raw state--action primitive versus explanation-derived primitive;
2. fixed window versus explanation change-point;
3. decision-only versus decision+outcome;
4. tanpa verification block;
5. tanpa temporal features;
6. physical outcome versus physical+reward;
7. context-based naming v1 versus evidence-based naming v2;
8. pooled versus per-solver discovery;
9. HDBSCAN versus K-means;
10. tanpa primitive certificate gate;
11. sparse anchors versus full trajectories;
12. supported-only versus seluruh representable region.

Ablation tidak boleh hanya membandingkan silhouette. Laporkan juga coverage,
outcome coherence, verification consistency, dan held-out stability.

---

## 14. Milestone implementasi

### CEDP0 — Freeze input certificates

Input:

- M1--M13 manifests;
- exact checkpoints/configs;
- EDDP v1 freeze dan results;
- M2 lexicon hash.

Gate: tidak ada retraining atau silent artifact replacement.

### CEDP1 — Certified explanation adapter

Bangun adapter yang membaca M1--M13 outputs menjadi satu
`CertifiedExplanationInstance` schema.

Gate: invalid/unsupported explanation tidak masuk discovery.

### CEDP2 — Full explanation trajectory recorder

Jalankan seluruh keputusan episode dan simpan certificate atau abstention pada
setiap timestep.

Gate: step contiguous dan provenance lengkap.

### CEDP3 — Explanation-only signature v2

Implementasikan decision, outcome, verification, certificate, dan temporal
feature blocks.

Gate: mengubah solver/action/context/M2 tidak mengubah feature vector.

### CEDP4 — Temporal segmentation

Implementasikan change-point utama dan fixed-window baseline.

Gate: segment deterministik, contiguous, dan label-free.

### CEDP5 — Discovery dan held-out assignment

Fit hanya pada development seeds, bekukan cluster, lalu assign held-out.

Gate: artifact hash dan deterministic rerun tersedia.

### CEDP6 — Structured functional descriptors

Bangun descriptor dari dominant explanation evidence tanpa forbidden fields.

Gate: setiap descriptor field memiliki member certificate references.

### CEDP7 — Primitive certificate checker

Terapkan support, stability, coherence, verification, provenance, dan boundary
gates.

Gate: status dihasilkan oleh checker, bukan nama atau keputusan manual.

### CEDP8 — Freeze, naming, dan M2 reconciliation

Freeze sebelum annotator naming dan sebelum membuka M2.

Gate: tidak ada cluster refit sesudah evaluasi eksternal.

### CEDP9 — Runtime assigner

Implementasikan frozen inference dan support-aware abstention.

Gate: out-of-support input menghasilkan `UNKNOWN`.

### CEDP10 — Ablation dan unified report

Jalankan ablation, comparison, catalogue, timeline, certificate cards, dan
failure catalogue.

Gate: hasil negatif dan solver-specific tetap dilaporkan.

---

## 15. Struktur kode yang direncanakan

```text
src/explainability/certified_primitives/
├── __init__.py
├── schema.py
├── certificate_adapter.py
├── trajectory.py
├── signature.py
├── segmentation.py
├── discovery.py
├── descriptor.py
├── certificate_checker.py
├── runtime.py
├── reconciliation.py
└── reporting.py

scripts/
├── run_cedp_freeze.py
├── run_cedp_collect.py
├── run_cedp_segment.py
├── run_cedp_discovery.py
├── run_cedp_certify.py
├── run_cedp_reconcile.py
└── run_cedp_report.py

configs/explainability/
└── cedp_v2.yaml

tests/
├── test_cedp_schema.py
├── test_cedp_leakage.py
├── test_cedp_segmentation.py
├── test_cedp_certificate.py
├── test_cedp_runtime.py
└── test_cedp_reconciliation.py
```

---

## 16. Artefak akhir

```text
runs/explanations/cedp_v2/
├── provenance_manifest.json
├── certified_explanation_trajectories.jsonl
├── abstained_explanations.jsonl
├── temporal_segments.jsonl
├── signatures_label_free.csv
├── cluster_freeze_pre_naming.json
├── structured_descriptors.json
├── primitive_certificates.json
├── primitive_catalogue.json
├── runtime_assigner.joblib
├── m2_reconciliation.json
├── ablation_results.json
├── failure_catalogue.json
├── figures/
└── clips/
```

---

## 17. Go/no-go

### GO

- minimal satu `CERTIFIED_PRIMITIVE` lintas solver;
- outcome coherence lebih baik dari permutation;
- held-out assignments stabil;
- certificate claims seluruhnya traceable;
- runtime abstention bekerja;
- v2 mengungguli old state--action primitive pada explanation preservation atau
  readability tanpa menurunkan correctness.

### REVISE

- cluster terbentuk tetapi certificate stability gagal;
- primitive hanya solver-specific;
- change-point terlalu terfragmentasi;
- naming masih memerlukan context atau action;
- coverage terlalu rendah.

### NO-GO untuk klaim utama

- explanation signatures hanya mengenali solver;
- cluster tidak lebih coherent dari random;
- primitive descriptor tidak dapat dibuat tanpa label M2;
- primitive certificate tidak dapat mempertahankan instance-level claims.

No-go tetap merupakan hasil ilmiah dan harus masuk paper sebagai batas metode.

---

## 18. Kontribusi paper yang ditargetkan

Kalimat kontribusi utama:

> We transform instance-level certified policy explanations into temporally
> coherent driving primitives. Unlike direct state--action labels, every
> primitive is grounded in recurring counterfactual triggers, action-outcome
> contrasts, and verified behavioral properties.

Hubungan dengan traffic-primitives literature:

- literatur lama menyegmentasi trajectory dan speed menjadi temporal traffic
  building blocks;
- C-EDDP menyegmentasi certified explanation traces menjadi temporal
  decision-making building blocks;
- primitive kita menjelaskan fungsi keputusan policy, bukan hanya bentuk
  trajectory.

---

## 19. Urutan eksekusi

```text
CEDP0 provenance freeze
→ CEDP1 certificate adapter
→ CEDP2 full explanation trajectories
→ CEDP3 leakage-free signatures
→ CEDP4 change-point segmentation
→ CEDP5 development discovery + held-out assignment
→ CEDP6 explanation-grounded descriptors
→ CEDP7 primitive certificate checker
→ CEDP8 freeze + naming + M2 reconciliation
→ CEDP9 runtime inference/Unknown
→ CEDP10 ablation + report
```

Plan v1 tidak dihapus. Ia menjadi baseline `sparse-anchor candidate discovery`,
sedangkan plan v2 menjadi eksperimen utama `certified explanation to certified
driving primitive`.
