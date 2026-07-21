# Plan Explainable MDP Policy

Dokumen ini merancang pipeline untuk menjelaskan dua policy MDP Duckietown:

- tabular Q-learning dengan state dan action diskrit;
- Soft Actor-Critic (SAC) dengan state dan action kontinu.

Tujuan utamanya bukan hanya menampilkan angka internal solver, tetapi
menerjemahkan perilaku policy menjadi **driving primitives** yang mudah
dipahami, dibandingkan, dan diverifikasi.

Status dokumen: **revision 3 — final untuk fase MDP, siap untuk M1**.

Framework dibekukan dengan nama deskriptif:

> **Primitive-Grounded Contrastive Explanation and Verification Framework for
> MDP Driving Policies.**

Revisi ini memisahkan tiga pilar yang menjawab pertanyaan berbeda:

1. **Decision explanation:** mengapa policy memilih action tersebut;
2. **Outcome explanation:** apa akibat mengambil action alternatif;
3. **Policy verification:** apakah keputusan konsisten dan aman.

Selain mempertahankan manifold contract, visit-count stratification, baseline
IG, critic-support caveat, dan bottom-up validation dari revisi sebelumnya,
revision 3 menambahkan paired simulator intervention yang COViz-inspired,
LEGIBLE-inspired metamorphic testing, temporal physical/reward outcomes,
formal foil protocol, deterministic replay acceptance test, dan kontrol
eksogen yang membedakan dunia reaktif dari dunia scripted.

---

## 1. Pertanyaan yang Harus Dijawab

Pipeline explainability harus dapat menjawab:

1. Apa yang sedang dilakukan policy?
2. Kondisi state apa yang memicu keputusan tersebut?
3. Mengapa action itu dipilih dibandingkan action alternatif?
4. Kondisi apa yang harus berubah agar policy memilih action lain?
5. Apa akibatnya jika action alternatif dipaksakan pada state yang sama?
6. Kapan konsekuensi fisik dan reward tersebut muncul?
7. Apakah keputusan konsisten dengan metamorphic relation dan safety property?

Contoh penjelasan pedestrian:

> Ego memilih `YieldHold` karena duck aktif berada dekat koridor
> penyeberangan. Ketika jarak longitudinal duck diperbesar dari 0,25 m
> menjadi 0,80 m, policy berpindah dari brake ke slow-straight. Jika
> `slow_straight` dipaksakan pada state awal, clearance minimum turun dan ego
> memasuki koridor pedestrian sebelum Duckie selesai menyeberang.

Contoh penjelasan lane following:

> Ego memilih `LaneCorrectRight` karena lateral error positif dan heading
> error mengarah keluar lane. Pada Q-learning, margin terhadap action terbaik
> kedua adalah 2,41.

---

## 2. Formulasi Tiga Pilar

Tidak ada satu alat yang menjawab seluruh pertanyaan explanation. Pipeline
memakai tiga pilar yang saling melengkapi.

### 2.1 Decision explanation: mengapa action dipilih?

Kedua policy mula-mula diperlakukan sebagai fungsi input-output:

$$
\pi(s) \rightarrow a.
$$

Decision explanation memuat:

- state dan trigger semantik;
- selected action dan selected primitive;
- contrast action atau **foil**;
- minimal state counterfactual;
- response curve;
- Q-margin untuk Q-learning;
- Integrated Gradients dan critic probe sebagai diagnosis tambahan SAC.

State counterfactual mempertahankan policy tetapi mengubah state:

$$
s'=s+\delta,
\qquad
P(s',\pi(s'))\neq P(s,\pi(s)).
$$

Ia menjawab: **kondisi apa yang harus berubah agar keputusan berubah?**

Contoh sweep jarak stop sign:

$$
d_{\text{stop}} \in
\{3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.1, 0.0\}\ \text{m}.
$$

Eksperimen tersebut menunjukkan kapan Q-learning berpindah macro-action dan
kapan SAC mulai mengubah command kontinu.

### 2.2 Outcome explanation: apa akibat action alternatif?

Outcome explanation mempertahankan state branch yang sama, tetapi memaksakan
action pertama yang berbeda:

$$
\tau^*=\operatorname{Rollout}(s_t,a^*,\pi),
\qquad
\tau^{cf}=\operatorname{Rollout}(s_t,a^{cf},\pi).
$$

Setelah action pertama, kedua cabang kembali mengikuti policy evaluation yang
sama. Pendekatan ini disebut **COViz-inspired paired action-outcome rollout**.
Ia menjawab: **apa yang terjadi jika action lain diambil?**

Istilah COViz-inspired dibatasi untuk outcome explanation. COViz tidak
menggantikan trigger analysis, state counterfactual, atau internal policy
diagnostics.

### 2.3 Policy verification: apakah perilaku konsisten dan aman?

Verification menggunakan dua kelas spesifikasi:

- safety properties;
- metamorphic relations dengan precondition, intervention, expected relation,
  dan tolerance yang eksplisit.

Q-learning diperiksa secara eksak pada finite representation dan dilaporkan
menurut validity/support stratum. SAC diperiksa melalui valid sampling,
intervention, boundary search, dan rollout.

Penggunaan metamorphic relation disebut **LEGIBLE-inspired metamorphic policy
testing**, bukan implementasi LEGIBLE penuh. Rule-guided policy improvement
berada di luar scope fase MDP explanation versi pertama.

### 2.4 Bahasa penyatu

Ketiga pilar menghasilkan driving primitive yang sama. Dengan demikian,
perbedaan action mentah antarsolver tidak menghalangi perbandingan semantik:

```text
Primitive contrast : YieldHold vs ResumeAfterYield
Q-learning         : brake vs slow_straight
SAC                : (v=0.00, omega=0.00) vs (v=0.17, omega=0.00)
```

---

## 3. Mengapa Bentuk Hasilnya Berbeda?

Metode behavioral yang digunakan sama, tetapi representasi policy dan action
space berbeda.

### 3.1 Q-learning

Q-learning menghasilkan satu dari tujuh action diskrit:

$$
\mathcal{A}_{Q} =
\{\text{fast-left},\text{fast-straight},\text{fast-right},
\text{slow-left},\text{slow-straight},\text{slow-right},\text{brake}\}.
$$

Response curve Q-learning berbentuk fungsi tangga. Contoh:

```text
d_stop > 1.0 m        -> fast_straight
0.3 <= d_stop <= 1.0 -> slow_straight
d_stop < 0.3 m        -> brake
```

Seluruh Q-value juga dapat dibaca langsung dari Q-table.

### 3.2 SAC

SAC menghasilkan action differential-drive kontinu:

$$
a=(v_{\text{cmd}},\omega_{\text{cmd}}).
$$

Response curve SAC dapat berubah secara mulus. Contoh:

```text
d_stop = 1.0 m -> v_cmd = 0.35
d_stop = 0.7 m -> v_cmd = 0.27
d_stop = 0.5 m -> v_cmd = 0.18
d_stop = 0.3 m -> v_cmd = 0.06
d_stop = 0.1 m -> v_cmd = 0.00
```

Jadi, metode probing-nya sama, tetapi hasil Q-learning berbentuk pergantian
macro-action sedangkan hasil SAC berbentuk profil kontrol kontinu.

---

## 4. State Semantik Bersama

Representasi state internal kedua solver berbeda, tetapi konsep berkendaranya
dapat disatukan.

| Kelompok konsep | Q-learning | SAC |
|---|---|---|
| Posisi lane | bin $d$ | $d$ kontinu |
| Kesalahan arah | bin $e=\phi+d$ | $\phi$ kontinu |
| Kecepatan | bin $v$ | $v$ kontinu |
| Geometri jalan | kategori curvature | $\kappa$ kontinu |
| Stop sign | bin $d_{\text{stop}}$, $\sigma_{\text{stop}}$ | present, distance, satisfied, hold progress |
| Pedestrian | kategori threat | posisi, kecepatan relatif, active, crossing available |

### 4.1 State Q-learning

State diskrit Q-learning adalah:

$$
\bar{s}=
(\operatorname{bin}(d),
 \operatorname{bin}(\phi+d),
 \operatorname{bin}(v),
 \kappa_{\text{category}},
 \operatorname{bin}(d_{\text{stop}}),
 \sigma_{\text{stop}},
 h_{\text{duck}}).
$$

Shape state adalah:

```text
(5, 5, 3, 3, 4, 2, 5)
```

sehingga terdapat 9.000 state diskrit. Dengan tujuh action, shape Q-table
adalah:

```text
(5, 5, 3, 3, 4, 2, 5, 7)
```

### 4.2 State SAC

SAC menggunakan observation kontinu 15 dimensi:

```text
[d,
 phi,
 v,
 kappa,
 stop_present,
 d_stop,
 sigma_stop,
 duck_present,
 duck_longitudinal,
 duck_lateral,
 duck_v_longitudinal_relative,
 duck_v_lateral_relative,
 duck_active,
 duck_crossing_available,
 stop_hold_progress]
```

### 4.3 Canonical Concept Vector dan Entanglement

Untuk laporan bersama, fitur mentah mula-mula dikelompokkan menjadi enam
konsep:

$$
C = [
C_{\text{lane}},
C_{\text{heading}},
C_{\text{speed}},
C_{\text{road}},
C_{\text{stop}},
C_{\text{pedestrian}}
].
$$

Pengelompokan ini membuat beberapa fitur duck pada SAC dapat dibandingkan
dengan satu kategori duck pada Q-learning tanpa menganggap representasi
mentahnya identik.

Namun, representasi internal Q-learning memakai $e=\phi+d$, bukan $\phi$
murni. Oleh karena itu flip atau atribusi pada dimensi kedua Q-table tidak boleh
dilabeli sebagai pengaruh heading murni. Hasil tersebut diberi nama:

$$
C_{\text{lane-heading-entangled}}.
$$

Pelaporan dilakukan dalam dua tingkat:

1. **Shared level:** `lane_control`, `speed`, `road`, `stop`, dan
   `pedestrian`. `lane_control` menggabungkan kontribusi posisi dan heading
   agar perbandingan lintas solver tetap sah.
2. **Detailed level:** SAC dan raw-state behavioral sweep boleh memisahkan
   $d$ dan $\phi$. Internal flip analysis Q-learning tetap dilaporkan sebagai
   `lane-heading-entangled` dan tidak dibandingkan langsung dengan atribusi
   $\phi$ murni milik SAC.

Primitive labeler dan trajectory recorder selalu menyimpan raw $d$ dan
$\phi$. Dengan demikian analisis perilaku masih dapat membedakan posisi lane
dan heading, walaupun representasi internal Q-table menggabungkan keduanya.

---

## 5. Leksikon Driving Primitives

Action mentah belum otomatis menjadi driving primitive. Primitive ditentukan
dari kombinasi state, action, dan konteks kejadian.

### 5.1 Lane-following primitives

- `CruiseStraight`
- `CruiseCurveLeft`
- `CruiseCurveRight`
- `LaneCorrectLeft`
- `LaneCorrectRight`
- `DecelerateLane`
- `EmergencyLaneRecovery`

### 5.2 Stop-sign primitives

- `ApproachStop`
- `DecelerateStop`
- `StopHold`
- `StopSatisfied`
- `ResumeAfterStop`

### 5.3 Pedestrian primitives

- `ApproachCrossing`
- `YieldDecelerate`
- `YieldHold`
- `WaitForClearance`
- `ResumeAfterYield`

### 5.4 Primitive yang tidak diinginkan

- `UnnecessaryBrake`
- `UnsafeProceed`
- `StopViolation`
- `LaneDeparture`
- `OscillatorySteering`
- `PrematureResume`

### 5.5 Action bukan primitive

Action `brake` dapat memiliki beberapa makna:

- `StopHold` ketika ego berada di stop sign;
- `YieldHold` ketika duck sedang menyeberang;
- `UnnecessaryBrake` ketika jalan kosong;
- `EmergencyLaneRecovery` jika berhenti digunakan untuk menghindari keluar
  lane.

Karena itu primitive labeler wajib membaca konteks state dan event, bukan hanya
nama action.

---

## 6. Pipeline Keseluruhan

```text
Q-learning checkpoint --------+                  +--> exact global analysis
                               |                  |
SAC checkpoint ---------------+--> policy adapter+--> sampled global analysis
                                                  |
                                                  v
                                      Canonical semantic state
                                                  |
                     +----------------------------+---------------------------+
                     |                            |                           |
                     v                            v                           v
           Decision explanation         Outcome explanation          Policy verification
           - trigger/primitive          - paired branching           - safety properties
           - state counterfactual       - chosen vs foil             - metamorphic tests
           - Q/IG diagnostics           - temporal profiles          - state strata
                     |                            |                           |
                     +----------------------------+---------------------------+
                                                  |
                                                  v
                                Explanation JSON / CSV / timeline / video
                                                  |
                                                  v
                     Optional bottom-up clustering and lexicon reconciliation
```

Pipeline memiliki lima lapisan kerja:

1. **Policy adapter** membaca checkpoint tanpa mengubah policy.
2. **Global characterization** meringkas mapping policy dan rule behavior.
3. **Local decision explanation** menjelaskan trigger dan batas keputusan.
4. **Temporal outcome explanation** membandingkan chosen dan foil rollout.
5. **Verification** menguji consistency, safety, validity, dan reproducibility.

Bottom-up clustering bukan explanation utama. Ia adalah validasi opsional
apakah leksikon top-down juga muncul sebagai struktur perilaku policy.

### 6.1 Target policy yang dijelaskan

Q-learning selalu dijelaskan dalam mode greedy dan teacher-free. SAC selalu
dijelaskan menggunakan deterministic actor mean:

$$
\pi_{\text{eval}}(s)=\mu_\theta(s).
$$

Metadata wajib menyimpan:

```json
{
  "policy_mode": "deterministic_actor_mean",
  "actor_sampling": false,
  "teacher_active": false
}
```

Audit stochastic SAC dipisahkan sebagai eksperimen lain dan tidak dicampur
dengan hasil deterministic policy.

### 6.2 Independensi leksikon

File aturan primitive, threshold configuration, unit tests, versi schema,
Git commit, dan SHA-256 dibekukan pada akhir M2. Pembekuan dilakukan walaupun
bottom-up clustering akhirnya tidak dijalankan. Clustering dilarang memakai
primitive label, trigger label, atau nama event sebagai fitur.

Commit bukan satu-satunya bukti freeze, tetapi hash dan versi artefak wajib.

---

## 7. Penjelasan Khusus Q-learning

Q-learning berupa tabel berhingga, sehingga dapat diperiksa secara eksak.

### 7.1 Exact policy map

Untuk setiap state:

$$
\pi_Q(s)=\arg\max_a Q(s,a).
$$

Setiap baris hasil menyimpan:

- state diskrit;
- action terbaik;
- action terbaik kedua;
- driving primitive;
- status aman atau tidak aman;
- `training_visit_count`, bila artefaknya tersedia;
- `evaluation_reach_count` dari rollout greedy yang dibekukan;
- `representable`, `valid_manifold`, `reachable`, dan `supported`;
- provenance status: `trained`, `reached_only`, `unseen`, atau `unknown`.

Empat validity/support strata didefinisikan sebagai berikut:

1. **representable:** seluruh 9.000 indeks yang dapat dialamatkan Q-table;
2. **valid-manifold:** kombinasi lolos semantic-state validator;
3. **reachable:** pernah muncul pada rollout manifest yang dibekukan;
4. **supported:** visitation memenuhi threshold yang dibekukan.

Checkpoint historis saat ini hanya menyimpan Q-table dan tidak menyimpan
training visit count. Pipeline tidak boleh mengarang nilai tersebut. Untuk
artefak lama:

- `training_visit_count` diberi `null`;
- `evaluation_reach_count` direkonstruksi dari rollout greedy dengan seed
  tetap, tetapi tidak boleh disebut sebagai training visitation;
- state dengan `evaluation_reach_count>0` diberi `reached_only`; state lain
  tetap `unknown`. Label `unseen` hanya digunakan jika training count benar-benar
  tersedia dan bernilai nol;
- training berikutnya wajib menyimpan state dan state-action visit counts
  bersama checkpoint.

### 7.2 Q-margin

Jika $a_1$ dan $a_2$ adalah action dengan Q-value terbesar dan kedua terbesar:

$$
M_Q(s)=Q(s,a_1)-Q(s,a_2).
$$

Interpretasinya:

- margin besar: action terbaik terpisah jelas dari alternatif;
- margin kecil: keputusan sensitif atau ambigu;
- margin nol: terdapat tie.

Q-margin bukan probabilitas keyakinan. Nilai ini hanya mengukur separation
antar-action dalam Q-table.

Q-margin selalu dilaporkan bersama visit-count stratum. Margin pada state
`unseen` atau `unknown` tidak boleh ditafsirkan sebagai keyakinan policy karena
dapat sekadar berasal dari nilai inisialisasi dan tie-breaking.

### 7.3 One-bin counterfactual

Satu dimensi state digeser satu bin:

$$
s'_j=s_j\pm1.
$$

Kemudian diperiksa apakah:

$$
\pi_Q(s')\neq\pi_Q(s).
$$

Contoh penjelasan:

> Ketika kategori duck berubah dari `SIDE_NEAR` menjadi `CROSSING_NEAR`,
> action berpindah dari `slow_straight` menjadi `brake`.

One-bin counterfactual hanya masuk analisis pengaruh utama jika state anchor
dan state tetangganya memiliki dukungan visitation yang memadai. Flip pada
state `unseen` tetap boleh dicatat untuk audit keselamatan, tetapi dipisahkan
dari klaim perilaku yang telah dipelajari.

### 7.4 Verifikasi ekshaustif

Karena hanya terdapat 9.000 state, seluruh state dapat diperiksa terhadap
properti keselamatan.

Contoh:

```text
IF duck == CROSSING_NEAR
THEN action in {slow_left, slow_straight, slow_right, brake}
```

```text
IF stop_distance == NEAR AND sigma_stop == false
THEN action not in {fast_left, fast_straight, fast_right}
```

Kelebihan Q-learning adalah properti tersebut dapat diperiksa untuk seluruh
state diskrit, bukan hanya state yang muncul saat rollout.

Hasil wajib dilaporkan terpisah untuk `representable`, `valid-manifold`,
`reachable`, dan `supported`, kemudian diberi breakdown provenance
`trained/reached_only/unseen/unknown`.

Verifikasi keselamatan tetap dijalankan pada seluruh 9.000 state karena state
yang tidak terlatih masih mungkin tercapai saat deployment. Namun statistik
explainability, Q-margin, flip rate, dan primitive distribution tidak boleh
menggabungkan keempat strata tersebut menjadi satu angka tanpa breakdown.

### 7.5 Foil Q-learning

Q-learning memakai tiga foil yang dibekukan sebelum eksperimen:

1. **Q2 foil:** action dengan Q tertinggi kedua;
2. **semantic foil:** action yang merealisasikan primitive contrast;
3. **safety foil:** action yang menguji properti tertentu.

Karena action hanya tujuh, seluruh enam alternatif juga dihitung sebagai audit.
Q2 foil hanya masuk explanation utama bila anchor `reachable` atau `supported`.
Pada state unsupported, Q2 dapat berasal dari noise inisialisasi atau tie dan
hanya boleh diberi label `UNSUPPORTED_POLICY_REGION` dalam appendix.

---

## 8. Penjelasan Khusus SAC

SAC mempunyai state dan action kontinu sehingga tidak dapat dienumerasi secara
ekshaustif.

### 8.1 Response curves

Satu fitur disapu sambil fitur lain ditahan tetap. Contoh:

$$
d_{\text{stop}}: 3.0 \rightarrow 0.0.
$$

Output yang diplot adalah:

$$
v_{\text{cmd}}(d_{\text{stop}})
\quad\text{dan}\quad
\omega_{\text{cmd}}(d_{\text{stop}}).
$$

Kurva tersebut menunjukkan braking profile dan kemungkinan perubahan steering
saat mendekati stop sign.

### 8.2 Minimal counterfactual

Dicari perubahan state terkecil yang mengubah primitive:

$$
\delta^*=\arg\min_{\delta}\|\delta\|
$$

dengan syarat:

$$
P(\pi(s+\delta))\neq P(\pi(s)).
$$

Contoh:

> Menggeser duck 0,32 m menjauhi koridor mengubah primitive dari
> `YieldHold` menjadi `ResumeAfterYield`.

### 8.3 Feature attribution

Integrated Gradients direkomendasikan sebagai baseline atribusi karena actor
SAC berbasis PyTorch dan output-nya kontinu. Atribusi dihitung terpisah untuk:

$$
v_{\text{cmd}}
\quad\text{dan}\quad
\omega_{\text{cmd}}.
$$

Nilai fitur kemudian dijumlahkan berdasarkan canonical concept group:

- lane influence;
- heading influence;
- speed influence;
- road influence;
- stop influence;
- pedestrian influence.

SHAP dapat ditambahkan sebagai validasi tambahan, tetapi tidak dijadikan
blocker karena biaya komputasinya lebih tinggi.

#### Baseline Integrated Gradients

IG selalu dihitung relatif terhadap baseline. Baseline utama dibekukan sebagai
state netral kanonis yang valid:

```text
d                         = 0
phi                       = 0
v                         = 0.17
kappa                     = 0
stop_present              = false
d_stop                    = None       # encoded sentinel = 1
sigma_stop                = false
stop_hold_progress        = 0
duck_present              = false
duck geometry             = absent sentinel
duck relative velocity    = 0
duck_active               = false
duck_crossing_available   = false
```

Dengan normalisasi encoder saat ini, absent Duckie berarti longitudinal
sentinel $+1$, lateral $0$, dan kedua relative velocity $0$. Nilai
$v=0.17$ berasal dari `v_slow` pada action configuration dan mewakili gerak
maju nominal, bukan state diam yang berpotensi mengandung semantik brake.

Sebagai sensitivity check, attribution juga dihitung terhadap baseline
alternatif berupa centroid empirical dari state `CruiseStraight` pada rollout
development. Laporan memuat cosine similarity dan rank correlation antara dua
baseline. Klaim fitur dominan hanya dibuat jika arah kesimpulannya stabil.

### 8.4 Critic action probes

Action actor dibandingkan dengan beberapa action referensi:

$$
Q(s,a_{\text{actor}})
$$

dibandingkan dengan:

$$
Q(s,a_{\text{brake}}),
Q(s,a_{\text{straight}}),
Q(s,a_{\text{left}}),
Q(s,a_{\text{right}}).
$$

Probe ini membantu menjelaskan bagaimana critic menilai action aktual terhadap
alternatif driving primitive. Probe diskrit hanya digunakan sebagai titik
referensi; actor SAC tetap menghasilkan action kontinu.

Critic probe adalah heuristik, bukan advantage yang terkalibrasi. Action
referensi yang jarang atau tidak pernah dihasilkan actor dapat berada di luar
distribusi replay, sehingga $Q(s,a_{ref})$ rentan terhadap extrapolation error
atau overestimation. Karena itu:

- istilah yang dipakai selalu **critic probe comparison**, bukan `critic
  advantage`, confidence, atau jaminan keselamatan;
- setiap probe dilaporkan bersama jarak action ternormalisasi dan actor
  log-probability bila dapat dihitung;
- bila replay snapshot tersedia, support ditentukan dengan nearest-neighbor
  distance pada pasangan $(s,a)$ ternormalisasi. Tanpa replay snapshot, probe
  hanya boleh diberi label `LOW_ACTOR_SUPPORT`, bukan dinyatakan pasti OOD;
- kesimpulan utama tetap membutuhkan counterfactual rollout atau simulator
  intervention, bukan critic value saja.

### 8.5 Foil SAC

SAC tidak mempunyai second-best action langsung. Foil dipilih dengan protokol
yang dibekukan sebelum evaluation:

1. **nearest primitive-changing foil**

   $$
   a^{cf}=\arg\min_a\|a-a^*\|
   \quad\text{dengan}\quad
   P(s,a)\neq P(s,a^*);
   $$

2. **canonical semantic foil:** brake, slow-straight, cruise-straight,
   corrective-left, atau corrective-right;
3. **critic-supported foil:** alternatif yang dekat dengan actor/replay
   support dan selalu membawa support caveat.

Hasil lintas solver memakai primitive contrast yang sama walaupun raw action
berbeda. Nearest foil dicari hanya di dalam action bounds dan wajib lulus
semantic-action validator. Pemilihan foil, tie-breaking, distance metric, dan
search budget disimpan dalam provenance untuk mencegah cherry-picking.

---

## 9. Paired Simulator Intervention

### 9.1 Scenario manifest

Setiap explanation temporal mempunyai manifest yang cukup untuk mereproduksi
branch point:

```text
map dan environment config
initial ego pose dan heading
Duckie pose, route, dan controller config
stop tracker state
simulation clock dan physics step
simulator seed dan RNG states
action prefix sebelum branch
exogenous disturbance trace bila digunakan
policy checkpoint dan policy mode
```

Branch point dibentuk dengan deterministic replay dari reset dan action prefix.
`deepcopy(env)` tidak dijadikan kontrak karena renderer, OpenGL context, dan
simulator object mungkin tidak serializable.

### 9.2 Eksogen dan endogen

Kondisi yang dibekukan antarcabang adalah:

- map, initial conditions, dan controller parameters;
- RNG streams dan pre-drawn exogenous disturbance trace;
- simulation clock pada branch point;
- action prefix;
- selected checkpoint dan evaluation mode.

State yang merupakan respons dunia terhadap action ego dibiarkan berbeda.
Khusus repeated Duckie, rearm atau crossing yang dipicu jarak ego adalah
**endogen**. Jika cabang brake dan slow-straight memicu Duckie berbeda, hal itu
adalah bagian sah dari outcome counterfactual, bukan experimental leakage.

Mode utama disebut:

> **reactive-world paired simulator intervention under controlled exogenous
> conditions.**

Mode ablation opsional memakai crossing schedule yang di-pre-draw berdasarkan
clock dan tidak bereaksi terhadap jarak ego:

> **scripted-world paired simulator intervention.**

Pipeline tidak mengklaim real-world causality. Klaim dibatasi pada
simulator-based interventional counterfactual dalam environment yang
didefinisikan.

### 9.3 Deterministic replay acceptance test

M5 tidak boleh dimulai sebelum manifest dan action sequence identik menghasilkan
dua replay identik menurut kontrak berikut:

- continuous state `allclose(atol=1e-7, rtol=0)` pada setiap decision step;
- discrete state, action, primitive, dan event identik;
- reward total dan setiap reward term `allclose(atol=1e-7, rtol=0)`;
- Duckie controller phase, crossing count, dan active state identik;
- termination reason dan termination step identik.

Tes wajib mencakup lane-only, stop sign, satu Duckie crossing, dan repeated
crossing/rearm. Toleransi hanya boleh diubah berdasarkan bukti numerik dan harus
dibekukan sebelum paired-outcome experiment.

### 9.4 Paired action-outcome rollout

Pada branch point yang sama:

1. cabang factual menjalankan selected action $a^*$;
2. cabang counterfactual menjalankan foil $a^{cf}$;
3. setelah action pertama, keduanya kembali mengikuti policy evaluation yang
   sama;
4. rollout berhenti pada fixed horizon, event-aligned horizon, atau terminal;
5. semua perbedaan outcome dihitung dan disimpan, termasuk hasil negatif.

Satu rollout deterministik tidak boleh dilaporkan sebagai probabilitas. Jika
ingin menyebut collision probability atau outcome frequency, eksperimen harus
diulang pada manifest/seed ensemble dan dilaporkan sebagai frekuensi empiris
beserta interval ketidakpastian.

### 9.5 Temporal horizon

Dua horizon digunakan bersama.

Fixed-step horizon:

$$
h\in\{1,5,10,20,30\}.
$$

Event-aligned rollout berhenti ketika salah satu kondisi berikut tercapai:

- lane kembali stabil;
- ego melampaui stop line;
- stop dwell berhasil dipenuhi;
- Duckie meninggalkan crossing;
- primitive berubah;
- lane departure atau collision;
- episode berakhir;
- maximum horizon tercapai.

Hasil disebut **simulator-based temporal counterfactual outcome profile**,
bukan Temporal Policy Decomposition penuh. Implementasi TPD penuh memerlukan
Expected Future Outcomes yang dipelajari dengan fixed-horizon TD dan berada di
luar scope versi pertama.

---

## 10. Reward dan Physical Outcome Explanation

Kode saat ini sudah menyediakan `RewardBreakdown` dengan komponen:

- `progress`;
- `lateral`;
- `heading`;
- `time`;
- `pedestrian`;
- `stagnation`;
- `steering`;
- `events`.

Reward per step dapat ditulis sebagai:

$$
r_t =
r_{\text{progress}}+
r_{\text{lateral}}+
r_{\text{heading}}+
r_{\text{time}}+
r_{\text{pedestrian}}+
r_{\text{stagnation}}+
r_{\text{steering}}+
r_{\text{events}}.
$$

Temporal reward profile dihitung per komponen dan horizon:

$$
G_k(h)=\sum_{t=0}^{h-1}\gamma^t r_{k,t}.
$$

Reward adalah objective optimisasi dan dapat mengandung shaping. Oleh karena
itu ia harus dipisahkan dari semantic physical outcomes:

- progress dalam meter;
- mean dan maximum $|d|$;
- mean dan maximum $|\phi|$;
- minimum Duckie clearance;
- time-to-collision bila definisinya valid;
- stop violation count dan stop dwell duration;
- lane departure dan collision;
- steering reversals dan steering jerk;
- primitive sequence dan transition times;
- termination reason dan termination step.

Untuk satu paired rollout, laporan menyimpan:

$$
\Delta G_k(h)=G_k^*(h)-G_k^{cf}(h)
$$

serta selisih setiap physical outcome. Contoh utama memakai satuan fisik,
bukan hanya reward shaping:

```text
brake         : minimum Duckie clearance = 0.38 m
slow_straight : minimum Duckie clearance = 0.11 m
```

Terdapat batas penting:

> Reward breakdown menjelaskan konsekuensi langsung setelah action, bukan
> sepenuhnya alasan policy memilih action tersebut.

Karena itu reward breakdown harus digabungkan dengan:

- Q-values dan Q-margin untuk Q-learning;
- actor attribution dan critic probes untuk SAC;
- paired temporal rollout dan physical outcome comparison untuk keduanya.

Reward-decomposed Q dapat menjadi eksperimen lanjutan, tetapi membutuhkan
retraining critic dengan beberapa output head. Fitur tersebut tidak menjadi
blocker versi pertama pipeline explainability.

---

## 11. Format Penjelasan Seragam

Kedua policy harus menghasilkan schema yang sama. Contoh keluaran SAC:

```json
{
  "solver": "sac",
  "policy_mode": "deterministic_actor_mean",
  "teacher_active": false,
  "scenario_manifest_id": "pedestrian_seed_17_branch_0042",
  "state": {},
  "selected_action": {"v_cmd": 0.02, "omega_cmd": 0.01},
  "selected_primitive": "YieldHold",
  "trigger": "duck_crossing_near",
  "foil": {
    "type": "semantic_foil",
    "primitive": "ResumeAfterYield",
    "action": {"v_cmd": 0.17, "omega_cmd": 0.00},
    "support_label": "ACTOR_SUPPORTED"
  },
  "influence": {
    "lane": 0.05,
    "heading": 0.03,
    "speed": 0.00,
    "road": 0.02,
    "stop": 0.04,
    "pedestrian": 0.86
  },
  "state_counterfactual": {
    "change": "duck_longitudinal +0.35 m",
    "new_primitive": "ResumeAfterYield"
  },
  "action_outcome_counterfactual": {
    "mode": "reactive_world",
    "fixed_horizons": [1, 5, 10, 20, 30],
    "selected": {
      "minimum_duck_clearance_m": 0.38,
      "collision": false,
      "primitive_sequence": ["YieldHold", "ResumeAfterYield"]
    },
    "foil": {
      "minimum_duck_clearance_m": 0.11,
      "collision": false,
      "primitive_sequence": ["UnsafeProceed"]
    },
    "temporal_reward_profiles": {}
  },
  "metamorphic_results": {},
  "validity": {
    "state_manifold": "PASS",
    "replay_reproducibility": "PASS",
    "support_status": "reachable"
  },
  "provenance": {}
}
```

Q-learning menggunakan schema yang sama, tetapi bagian action memuat
`action_id` dan `action_name`. Bagian internal explanation memuat Q-values dan
Q-margin, sedangkan SAC memuat critic probes.

---

## 12. Tahapan Implementasi

### M1 — Explanation schema dan policy adapters

Buat struktur awal:

```text
src/explainability/
|-- __init__.py
|-- schema.py
|-- semantic_state.py
|-- q_policy_adapter.py
`-- sac_policy_adapter.py
```

Target:

- kedua checkpoint dapat menerima canonical query;
- keluaran action mempunyai format seragam;
- tidak mengubah training maupun checkpoint;
- evaluasi tetap teacher-free dan deterministic/greedy.

### M2 — Primitive labeler

Buat:

```text
src/explainability/primitives.py
```

Target:

- primitive ditentukan dari state, action, dan event;
- aturan yang sama dipakai Q-learning dan SAC;
- setiap primitive mempunyai alasan atau trigger;
- action `brake` tidak otomatis dianggap `StopHold`;
- threshold configuration dan unit test primitive dibekukan di akhir M2;
- freeze manifest menyimpan `primitive_schema_version`, SHA-256 rules/config,
  dan Git commit bila tersedia;
- freeze dilakukan sebelum clustering atau rekonsiliasi apa pun.

Setelah freeze, aturan tidak boleh diubah berdasarkan cluster yang sudah
terlihat. Jika revisi leksikon diperlukan, ia menjadi schema version baru dan
seluruh evaluation terkait diulang.


### M3 — Trajectory recorder dan segmenter

Buat:

```text
src/explainability/trajectory.py
```

Setiap step menyimpan:

```text
state, action, primitive, reward terms, events, termination reason
```

Primitive berulang digabung menjadi segmen:

```text
CruiseStraight:   step 0-51
DecelerateStop:   step 52-65
StopHold:         step 66-72
ResumeAfterStop:  step 73-91
```

### M4 — Deterministic replay dan simulator branching

Buat:

```text
src/explainability/scenario_manifest.py
src/explainability/simulator_branching.py
```

Target:

- merekam seluruh provenance branch point;
- membangun ulang branch point dari reset dan action prefix;
- menangkap RNG state dan controller state;
- membedakan frozen exogenous conditions dari endogenous world response;
- menyediakan mode `reactive_world` sebagai default;
- menyediakan `scripted_world` hanya sebagai ablation opsional.

Acceptance test M4 bersifat blocking: manifest dan action sequence sama harus
menghasilkan trajectory, reward terms, event, Duckie phase, dan termination
yang identik menurut toleransi Section 9.3. M5 dilarang dimulai sebelum test
lane, stop, one-crossing, dan repeated-crossing lulus.

### M5 — COViz-inspired paired outcomes

Buat:

```text
src/explainability/action_outcomes.py
src/explainability/temporal_outcomes.py
```

Target:

- selected dan foil branch berasal dari branch point identik;
- hanya action pertama yang dipaksakan, lalu policy evaluation kembali aktif;
- fixed-step dan event-aligned horizons tersedia;
- reward profile dan physical outcome profile disimpan terpisah;
- satu rollout tidak disebut probability;
- hasil negatif dan reactive Duckie divergence tetap dilaporkan.

Minimum viable explanation harus berhasil pada tiga skenario:

1. lane correction;
2. stop-sign approach/hold;
3. pedestrian crossing/yield.

### M6 — State counterfactual dan response curves

Buat:

```text
src/explainability/counterfactual.py
src/explainability/response_curves.py
```

Eksperimen minimum:

- lateral-offset sweep;
- heading-error sweep;
- curvature sweep;
- stop-distance sweep;
- stop-hold-progress sweep;
- duck longitudinal sweep;
- duck lateral sweep;
- duck active/absent intervention;
- duck crossing available/unavailable intervention.

Metode yang sama dijalankan pada kedua policy.

#### Kontrak validitas state sintetis

Semua sweep dimulai dari **anchor state yang berasal dari rollout riil**, bukan
state yang dikarang bebas. Generator counterfactual hanya mengubah fitur target
dan wajib memperbaiki dependent fields agar state tetap berada pada manifold
task yang masuk akal.

Kontrak minimumnya adalah:

1. Semua nilai finite dan berada dalam bounds observation.
2. Jika `stop_present=false`, maka `d_stop=None`, encoded stop distance memakai
   sentinel $1$, `sigma_stop=false`, dan `stop_hold_progress=0`.
3. Jika `stop_present=true`, maka $d_{stop}\in[0,3]$ m. Jika
   `sigma_stop=true`, `stop_hold_progress=1`. Jika belum satisfied, hold
   progress harus konsisten dengan fase dwell.
4. Jika `duck_present=false`, geometri mentah tidak dipakai dan encoder wajib
   menghasilkan sentinel `(long=+1, lat=0, v_long=0, v_lat=0)`;
   `duck_active=false` dan `duck_crossing_available=false`.
5. Sweep posisi atau velocity Duckie mewajibkan `duck_present=true`.
6. Kombinasi `duck_active` dan `duck_crossing_available` harus konsisten dengan
   konfigurasi controller. Pada one-crossing atau repeated crossing dengan
   re-arm distance, Duckie aktif umumnya membuat `crossing_available=false`.
   Pada mode unlimited tanpa re-arm, kedua bit dapat bernilai true karena
   `crossing_available` menyatakan budget/armed state, bukan perintah memulai
   crossing pada tick yang sama.
7. Intervensi phase controller yang tidak dapat dibentuk hanya dari state
   vector dijalankan melalui reset/step simulator, bukan dengan memalsukan bit.
8. Setiap synthetic state menyimpan `anchor_id`, daftar fitur yang diubah, hasil
   validator, dan alasan penolakan bila invalid.

Response curve mempunyai dua laporan: `valid-manifold only` sebagai hasil
utama dan rejected/off-manifold queries sebagai audit terpisah. Output policy
pada state yang gagal validator tidak boleh dipakai sebagai temuan perilaku.

### M7 — LEGIBLE-inspired metamorphic policy testing

Buat:

```text
src/explainability/metamorphic.py
```

Setiap metamorphic relation menyimpan:

```text
relation_id
precondition
source-state generator
valid intervention
expected action/primitive relation
tolerance
applicable solver
result provenance
```

Minimum relation set:

1. **MR-STOP:** bila stop present, belum satisfied, tidak ada Duckie, jalan
   cukup lurus, dan lane error aman, mengecilkan $d_{stop}$ tidak boleh
   menaikkan speed command di luar tolerance.
2. **MR-PEDESTRIAN:** pada konteks lane/stop tetap, peningkatan pedestrian risk
   tidak boleh meningkatkan speed level atau mengubah yield menjadi proceed.
3. **MR-CURVATURE:** dengan konteks keselamatan lain tetap, peningkatan
   $|\kappa|$ tidak boleh menaikkan speed command di luar tolerance.
4. **MR-LANE-SYMMETRY:** hanya pada local road context yang simetris,
   transformasi $(d,\phi)\rightarrow(-d,-\phi)$ diharapkan menghasilkan
   $\omega\rightarrow-\omega$ dalam tolerance.

Precondition formal MR-STOP minimum adalah:

$$
\begin{aligned}
&\text{stop\_present}=1,\quad \sigma_{stop}=0,\quad
\text{duck\_present}=0,\\
&|\kappa|\leq\kappa_{straight},\quad |d|\leq d_{safe},\quad
|\phi|\leq\phi_{safe},\\
&d'_{stop}<d_{stop},
\end{aligned}
$$

dengan seluruh fitur non-target dan dependent fields dipertahankan konsisten.

Untuk SAC, contoh MR-STOP adalah:

$$
v_{cmd}(s')\leq v_{cmd}(s)+\epsilon_v.
$$

Untuk Q-learning digunakan ordinal speed level:

```text
brake  = 0
slow-* = 1
fast-* = 2
```

sehingga:

$$
d'_{stop}<d_{stop}
\Longrightarrow
\operatorname{speedLevel}(\pi(s'))
\leq\operatorname{speedLevel}(\pi(s)).
$$

Semua precondition dan tolerance dipilih pada development seeds lalu dibekukan
sebelum held-out evaluation. Q-learning diperiksa menurut empat state strata;
SAC diperiksa dengan valid sampling dan intervention. Metamorphic relation
adalah ekspektasi domain, bukan hukum universal, sehingga `NOT_APPLICABLE`
harus dibedakan dari `PASS` dan `FAIL`.

Relasi yang sama dapat digunakan sebagai runtime monitor pada fase POMDP
kelak, tetapi implementasi monitor berada di luar scope MDP sekarang.

### M8 — Exact Q-table characterization dan verification

Buat:

```text
src/explainability/explain_q.py
```

Isi:

- exact policy enumeration;
- Q-margin;
- one-bin flip analysis;
- exhaustive safety-property checker;
- primitive distribution untuk seluruh state;
- `training_visit_count` bila tersedia;
- `evaluation_reach_count` dari manifest rollout tetap;
- breakdown `representable/valid-manifold/reachable/supported` untuk setiap
  property dan explanation metric;
- provenance `trained/reached_only/unseen/unknown` tanpa mengarang visit count.

### M9 — SAC internal diagnostics

Buat:

```text
src/explainability/explain_sac.py
```

Isi:

- Integrated Gradients untuk actor dengan baseline netral kanonis dan
  sensitivity baseline alternatif;
- critic action probes dengan distance-to-actor dan label support yang jujur;
- attribution stability dan canonical concept aggregation;
- adversarial sampling dekat decision boundary untuk diagnosis internal.

State counterfactual tetap dimiliki M6 dan metamorphic/property testing tetap
dimiliki M7; M9 tidak membuat protokol tandingan.

### M10 — Solver-aware rule extraction

Buat:

```text
src/explainability/rule_extraction.py
```

Q-learning memakai direct rule simplification, rule list per primitive, dan
decision tree atas mapping diskrit. Seluruh 9.000 mapping dapat dipakai, tetapi
fidelity tetap dilaporkan per validity/support stratum. Tree hanya disebut
exact bila fidelity benar-benar 100% pada domain yang diklaim.

SAC terlebih dahulu dipetakan:

$$
s\rightarrow a_{continuous}\rightarrow P(s,a).
$$

Surrogate kemudian mempelajari primitive-level mapping, bukan diklaim sebagai
exact continuous actor. Laporan wajib memuat:

- primitive fidelity;
- $\operatorname{MAE}_v$ dan $\operatorname{MAE}_\omega$;
- return dan safety violation ketika surrogate dijalankan sendiri;
- tree depth, leaf count, dan coverage;
- hasil per held-out seed.

Rule extraction adalah global summary. Ia tidak menggantikan paired rollout
untuk local outcome explanation.

### Comparison artefacts

Buat:

```text
src/explainability/compare_policies.py
```

Bandingkan:

- primitive frequency;
- primitive transition matrix;
- primitive duration;
- braking threshold;
- steering response;
- unnecessary braking;
- unsafe proceed;
- stop compliance;
- pedestrian yield compliance;
- feature influence signature;
- property violation rate.

### Presentation artefacts

Tambahkan informasi berikut ke renderer:

```text
Primitive  : YieldHold
Trigger    : duck crossing near
Influence  : pedestrian 86%
Alternative: slow_straight
Q margin   : 2.41
```

Untuk SAC, `Q margin` diganti dengan `critic probe comparison`. Istilah
`critic advantage` tidak digunakan karena probe dapat berada di luar
distribusi action actor.
Video hanya menjadi media visualisasi. Data utama tetap disimpan dalam
JSON/CSV agar analisis dapat direproduksi tanpa video.

### M11 — Optional bottom-up primitive discovery dan rekonsiliasi

Buat:

```text
src/explainability/signatures.py
src/explainability/cluster_primitives.py
src/explainability/reconcile_clusters.py
```

Tujuan milestone ini adalah menguji apakah primitive yang didefinisikan secara
top-down juga muncul sebagai struktur perilaku bottom-up.

#### Input signature

Satu signature dibuat per trajectory segment dan dapat memuat:

- canonical concept influence yang sudah dinormalisasi;
- statistik action: mean, variance, acceleration/deceleration, dan steering;
- raw semantic context tanpa nama primitive;
- duration dan transition context;
- solver dan scenario hanya sebagai metadata evaluasi, bukan fitur cluster.

Primitive label, trigger label, dan nama event dilarang menjadi fitur.

#### Penjaga metodologis

1. Leksikon dan rule-based labeler dibekukan pada akhir M2, lengkap dengan
   Git commit dan SHA-256.
2. Sampling distratifikasi berdasarkan solver, scenario, dan seed; label
   primitive tidak dipakai untuk sampling atau clustering.
3. Q-learning memakai visit-count mask. Default dukungan adalah
   `training_visit_count>=5`; untuk checkpoint historis tanpa count digunakan
   `evaluation_reach_count>=3` pada manifest 30 episode dan diberi label
   `reached_only`. State `unseen/unknown` tidak masuk clustering utama dan hanya
   dianalisis sebagai failure/coverage appendix. Threshold 1/5/10 dilaporkan
   sebagai sensitivity analysis.
4. Temporal noise dikurangi sebelum segment aggregation menggunakan filter
   yang dibekukan. Raw sequence tetap disimpan untuk audit.
5. Primary clustering memakai HDBSCAN karena jumlah cluster tidak dipaksakan.
   K-means dengan $k$ dipilih melalui silhouette score digunakan sebagai
   sensitivity analysis.
6. Hyperparameter dipilih pada development seeds dan dibekukan sebelum
   held-out seeds dibuka.

#### Rekonsiliasi

Setelah clustering selesai, label primitive baru dibuka. Hubungan antara
cluster emergent dan label top-down dilaporkan menggunakan:

- cluster × primitive confusion matrix;
- purity dan normalized mutual information;
- Adjusted Rand Index (ARI);
- coverage/noise rate HDBSCAN;
- contoh segmen representatif dan failure cases setiap cluster.

ARI atau purity rendah tidak otomatis dianggap kegagalan implementasi. Hasil
tersebut dapat menunjukkan bahwa leksikon terlalu kasar, satu primitive
memiliki beberapa mode, atau policy memakai perilaku yang belum dinamai.

### M12 — Unified report, timeline, dan video

Buat:

```text
src/explainability/explanation_report.py
```

Gunakan juga:

```text
src/explainability/compare_policies.py
```

Output utama adalah versioned JSON/CSV yang memuat decision explanation,
state counterfactual, paired action outcomes, temporal profiles, metamorphic
results, validity, support, dan provenance. Dashboard/video adalah presentasi
sekunder yang harus dapat diregenerasi dari data utama.

Minimum presentation:

- primitive timeline;
- selected dan foil trajectory side-by-side;
- temporal physical outcome dan reward component plots;
- trigger, state counterfactual, Q-margin/critic probe, dan support label;
- metamorphic PASS/FAIL/NOT_APPLICABLE;
- reactive-world/scripted-world mode;
- explicit `Unknown` dan `UNSUPPORTED_POLICY_REGION`.

Perbandingan Q-learning dan SAC memakai scenario manifest, seed ensemble,
branch horizon, primitive contrast, dan safety specification yang sama.

---

## 13. Eksperimen Minimum

Setiap policy harus diuji pada skenario yang sama.

### 13.1 Lane following

1. Ego tepat di centerline dan heading sejajar.
2. Ego bergeser ke kiri.
3. Ego bergeser ke kanan.
4. Heading mengarah keluar lane.
5. Memasuki tikungan kiri dan kanan.

### 13.2 Stop sign

1. Stop sign jauh.
2. Mendekati stop line.
3. Berada di stop zone tetapi masih bergerak.
4. Sedang memenuhi dwell time.
5. Stop telah dipenuhi dan ego boleh melanjutkan.

### 13.3 Pedestrian

1. Duck tidak ada.
2. Duck berada di samping tetapi belum crossing.
3. Duck mulai crossing.
4. Duck berada dekat lane ego.
5. Duck telah meninggalkan koridor.
6. Duck melakukan repeated crossing.


### 13.4 Protokol lintas solver

Untuk setiap scenario family:

1. gunakan manifest dan branch point yang sama;
2. pilih primitive contrast sebelum melihat outcome;
3. petakan contrast tersebut ke raw action masing-masing solver;
4. jalankan factual dan foil branch pada mode `reactive_world`;
5. simpan fixed dan event-aligned profiles;
6. ulangi pada seed ensemble jika menghitung frequency;
7. laporkan valid, invalid, terminal, dan failure cases.

Main paired explanations minimum:

- `LaneCorrect` versus `CruiseStraight`;
- `StopHold` versus `ResumeAfterStop/UnsafeProceed`;
- `YieldHold` versus `ResumeAfterYield/UnsafeProceed`.

---

## 14. Validasi Penjelasan

Penjelasan tidak cukup hanya terlihat masuk akal. Penjelasan harus diuji.
Seluruh subsection ini adalah **binding acceptance criteria lintas-fase**,
bukan inspirasi opsional. Setiap milestone yang menghasilkan explanation harus
menyimpan status validasinya; output yang gagal tidak masuk main results dan
tetap disimpan dalam audit report.

### 14.1 Counterfactual validity

State counterfactual wajib benar-benar mengubah action atau primitive ketika
dimasukkan kembali ke policy dan harus lulus manifold validator. Action-outcome
counterfactual wajib berasal dari branch point yang sama, menerapkan foil yang
tercatat, lalu kembali ke policy evaluation yang sama.

### 14.2 Fidelity

Jika nanti digunakan surrogate decision tree, ukur persentase kesesuaian
antara prediksi surrogate dan policy asli. Surrogate tidak boleh disebut
sebagai policy asli.

### 14.3 Stability

Perubahan state yang sangat kecil seharusnya tidak menyebabkan explanation
berubah drastis, kecuali state tersebut memang berada dekat decision boundary.

Stability dioperasionalkan sebagai berikut:

1. Ambil anchor valid dari rollout riil.
2. Bentuk tetangga valid dengan perturbasi kecil, misalnya
   $\Delta d=\pm0.005$ m, $\Delta\phi=\pm0.01$ rad,
   $\Delta v=\pm0.01$, dan perubahan jarak objek $\pm0.02$ m.
3. Ubah attribution menjadi canonical signed influence vector yang
   dinormalisasi L1, $z(s)$.
4. Hitung explanation distance:

$$
D_{exp}(s,s')=1-\frac{z(s)\cdot z(s')}
{\|z(s)\|_2\|z(s')\|_2}.
$$

Jika kedua vector nol, jarak ditetapkan nol; jika hanya satu yang nol, jarak
ditetapkan satu.

Anchor diberi label `near_boundary` jika:

- Q-learning mempunyai Q-margin kurang dari 5% local Q-range atau action
  berubah pada one-bin neighbor; atau
- SAC mengubah primitive atau normalized action lebih dari 0,05 pada salah
  satu perturbasi valid.

Stability utama hanya dihitung pada anchor non-boundary. Laporan memuat median,
p95, dan distribusi per primitive. Acceptance awal adalah p95
$D_{exp}\leq0.10$ pada development seeds; threshold dibekukan sebelum held-out
evaluation. State dekat boundary tetap dilaporkan terpisah sebagai sensitivity
map, bukan dibuang diam-diam.

### 14.4 Coverage

Primitive labeler harus memberi label pada minimal 95% step. State yang belum
memenuhi aturan diberi label `Unknown`, bukan dipaksa masuk primitive yang
salah.

### 14.5 Safety properties

Spesifikasi properti harus identik untuk kedua solver. Perbedaannya hanya pada
kekuatan pemeriksaan:

- Q-learning: pemeriksaan ekshaustif pada 9.000 state diskrit;
- SAC: sampling, boundary search, dan rollout karena state kontinu tidak
  berhingga.


### 14.6 Replay dan paired-outcome validity

Replay reproducibility wajib lulus kontrak Section 9.3. Paired report harus
membuktikan:

- manifest dan branch point factual/counterfactual identik;
- hanya action pertama yang berbeda sebelum policy kembali aktif;
- frozen exogenous state sama;
- endogenous Duckie reaction tidak dipaksa sama;
- horizon dan stopping event tercatat;
- probability hanya dipakai untuk seed ensemble.

### 14.7 Support, provenance, dan coverage penjelasan

Main explanation hanya dibuat pada state `reachable` atau `supported` yang
lulus manifold validator. State `unseen/unknown` tetap dapat diaudit, tetapi
wajib berlabel `UNSUPPORTED_POLICY_REGION` dan tidak boleh menghasilkan klaim
natural-language seolah-olah behavior tersebut telah dipelajari.

Setiap explanation menyimpan checkpoint hash, config hash, primitive schema,
scenario manifest, foil protocol, policy mode, seed, horizon, validator result,
dan software revision. Coverage minimal 95% berlaku pada eligible
reachable/supported steps; sisanya diberi `Unknown` secara eksplisit.

---

## 15. Kriteria Selesai

Pipeline dianggap selesai jika:

1. Kedua policy menggunakan leksikon primitive dan primitive contrast yang sama.
2. Q-learning dijelaskan greedy/teacher-free dan SAC memakai deterministic
   actor mean tanpa sampling.
3. Decision explanation memuat trigger, selected primitive, formal foil, dan
   state counterfactual.
4. State counterfactual benar-benar mengubah action/primitive dan lulus
   manifold validator.
5. Outcome explanation memakai chosen-versus-foil paired rollout dari branch
   point identik dan hanya memaksakan action pertama.
6. M4 deterministic replay lulus lane, stop, one-crossing, dan repeated-crossing
   acceptance tests sebelum M5 dijalankan.
7. Frozen exogenous conditions dan endogenous Duckie reactions dibedakan dalam
   manifest; mode utama adalah `reactive_world`.
8. Foil protocol dibekukan dan main explanation hanya memakai anchor
   `reachable/supported`; sisanya berlabel `UNSUPPORTED_POLICY_REGION`.
9. Temporal reward profile dan semantic physical outcomes dilaporkan terpisah.
10. Fixed-step dan event-aligned horizons keduanya tersedia.
11. Satu rollout tidak disebut probability; frequency hanya berasal dari seed
    ensemble.
12. Q-learning diperiksa pada representable, valid-manifold, reachable, dan
    supported strata dengan provenance `trained/reached_only/unseen/unknown`.
13. SAC diuji dengan valid sampling, intervention, boundary search, dan factual
    rollout.
14. Metamorphic relation memiliki precondition, intervention, expected
    relation, tolerance, dan status `PASS/FAIL/NOT_APPLICABLE`.
15. Primitive labeler mencapai coverage minimal 95% pada eligible steps dan
    sisanya diberi `Unknown`.
16. Evaluasi dan explanation dijalankan tanpa teacher dan tanpa retraining
    policy versi pertama.
17. Reward breakdown tidak disalahartikan sebagai alasan lengkap keputusan.
18. Semua hasil utama diekspor ke versioned JSON/CSV dengan provenance lengkap.
19. Perbandingan memakai manifest, map, seed ensemble, horizon, scenario,
    primitive contrast, dan safety specification yang sama.
20. Hasil negatif, pelanggaran, invalid query, dan unsupported region tetap
    disimpan dalam audit report.
21. Baseline IG utama dan alternatif dibekukan serta sensitivity dilaporkan.
22. Critic reference disebut `critic probe comparison`, membawa support caveat,
    dan tidak menjadi bukti utama outcome.
23. Leksikon dibekukan pada akhir M2 dengan schema version dan SHA-256 sebelum
    clustering atau rekonsiliasi.
24. Jika M11 dijalankan, bottom-up report memuat confusion matrix, ARI, NMI,
    purity, coverage/noise, dan failure cases pada held-out seeds.
25. Internal Q-learning menandai $e=\phi+d$ sebagai
    `lane-heading-entangled`; perbandingan murni terhadap $\phi$ SAC dilarang.
26. Rule surrogate melaporkan fidelity, action error, rollout return, safety,
    dan complexity; surrogate tidak disebut policy asli.
27. Seluruh acceptance criteria Section 14 bersifat binding dan statusnya
    melekat pada setiap explanation artefact.

---

## 16. Rekomendasi Final

Formulasi final fase MDP adalah:

$$
\boxed{
\text{Decision Explanation}
+\text{Counterfactual Outcome}
+\text{Verification}
}.
$$

Nama framework:

> **Primitive-Grounded Contrastive Explanation and Verification Framework for
> MDP Driving Policies.**

Kedudukan metode:

| Komponen | Q-learning | SAC |
|---|---|---|
| Global characterization | exact maps, Q-margin, direct rules | sampled maps, response curves, surrogate rules |
| Local decision explanation | trigger, one-bin/minimal state counterfactual | trigger, minimal continuous counterfactual, IG |
| Local outcome explanation | COViz-inspired paired rollout | COViz-inspired paired rollout |
| Internal diagnostic | exact Q-vector | critic probe dengan support caveat |
| Consistency verification | exact metamorphic checking per stratum | sampled/interventional metamorphic checking |
| Safety verification | representable dan valid/reachable/supported report | sampling, boundary search, rollout |

COViz-inspired rollout adalah metode utama untuk **outcome explanation**, bukan
seluruh explanation. Trigger analysis dan state counterfactual menjelaskan
decision boundary; metamorphic dan safety testing memverifikasi consistency.

Bottom-up clustering adalah validasi taksonomi opsional. Reward-decomposed
critic, scripted-world ablation, stochastic-SAC audit, rule-guided policy
improvement, dan TPD penuh merupakan future work/ablation dan bukan blocker
versi pertama.

Kontribusi yang diuji adalah penyatuan dua representasi policy dalam:

- satu canonical semantic state;
- satu leksikon driving primitive yang dibekukan;
- satu formal foil protocol;
- satu paired temporal counterfactual protocol;
- satu metamorphic/safety specification;
- satu versioned explanation schema;
- satu acceptance battery dan provenance contract.

## 17. Referensi Metodologis Utama

1. Amitai, Septon, dan Amir (2024), *Explaining Reinforcement Learning Agents
   through Counterfactual Action Outcomes*, AAAI 2024 (COViz).
2. Tappler et al. (2025), *Rule-Guided Reinforcement Learning Policy Evaluation
   and Improvement*, IJCAI 2025 (LEGIBLE).
3. Ruggeri et al. (2025), *Explainable Reinforcement Learning via Temporal
   Policy Decomposition*, arXiv:2501.03902.
4. Xiong et al. (2024), *XRL-Bench: A Benchmark for Evaluating and Comparing
   Explainable Reinforcement Learning Techniques*, arXiv:2402.12685.
5. Bastani, Pu, dan Solar-Lezama (2018), *Verifiable Reinforcement Learning via
   Policy Extraction*, NeurIPS 2018 (VIPER).
