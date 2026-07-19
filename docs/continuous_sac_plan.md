# Lampiran Plan: Continuous-State dan Continuous-Action SAC

Dokumen ini melanjutkan baseline tabular yang sudah selesai. Q-learning dan
SARSA beserta artefak ablation-nya tidak diganti atau dibuang. Tujuan tahap ini
adalah memisahkan tiga sumber kesulitan eksperimen:

1. kontrol diskrit dengan state terdiskritisasi (baseline tabular);
2. kontrol kontinu dengan privileged state simulator (SAC-MDP);
3. kontrol kontinu dari kamera dan history (visuomotor POMDP).

Urutan tersebut memungkinkan kegagalan kontrol dibedakan dari kegagalan
persepsi.

## 1. Audit terhadap kode saat ini

| Komponen | Status saat ini | Perubahan untuk SAC |
|---|---|---|
| Lane state `d`, `phi`, `v` | Sudah kontinu di `RawState`, lalu didiskritisasi untuk Q-table | Pakai nilai ternormalisasi langsung |
| Curvature look-ahead | Sudah ego-relative dan melihat 0,30 m ke depan, tetapi berupa 3 kelas | Tambah signed continuous curvature `kappa` |
| Stop sign | `d_stop` opsional dan `sigma_stop` tersedia | Tambah `stop_present` dan sentinel aman saat tidak ada sign |
| Duckie | Masih berupa 5 kelas ancaman | Tambah posisi/velocity relatif, `duck_present`, dan phase controller |
| Action sebelumnya | Belum ada | Tambahkan hanya jika diperlukan oleh reward/dynamics; wajib bila smoothness reward aktif |
| Action space | `Discrete(7)` | Wrapper baru dengan `Box([0,-omega_max], [v_max,omega_max])` |
| Observation space | Belum dideklarasikan untuk `RawState` dataclass | Deklarasikan `Box` dengan shape dan batas yang tepat |
| Neural solver | DQN baru berupa scaffolding; SAC belum ada | Tambah SAC setelah compatibility spike lulus |
| Runtime ML | Gym 0.23.1 dan NumPy 1.20; Torch/SB3/Gymnasium/Shimmy belum terpasang | Jangan mengubah environment utama sebelum spike terisolasi lulus |

Poin penting: kode saat ini **tidak lagi** memakai label `curve_left` atau
`curve_right` secara mentah dari frame peta. `src/state.py` memilih directed
Bezier lane yang searah ego dan menentukan tanda belokan dari tangent. Jadi
pekerjaan M7 adalah memperkaya fitur itu menjadi nilai kontinu, bukan
memperbaiki label mentah dari nol.

## 2. Formulasi SAC-MDP

MDP tetap ditulis sebagai

\[
\mathcal{M}=(\mathcal{S},\mathcal{A},P,R,\gamma,\rho_0,\mathcal{T}).
\]

Simulator Duckietown tetap menjadi generative transition model. Artinya kita
tidak membangun matriks transisi eksplisit; pemanggilan `step(a_t)` menghasilkan
sample `(s_t, a_t, r_t, s_{t+1})` dari dinamika simulator.

### 2.1 Curvature kontinu bertanda

Untuk tangent lajur ternormalisasi `t_0` dan tangent look-ahead `t_1`, perubahan
heading bertanda dihitung dengan

\[
\Delta\psi=
\operatorname{atan2}\left((t_0\times t_1)_y,\;t_0\cdot t_1\right).
\]

Jika panjang arc yang disampling adalah `Delta s`, rata-rata curvature lokal
adalah

\[
\kappa_t=\frac{\Delta\psi}{\Delta s}.
\]

`kappa > 0` dan `kappa < 0` masing-masing mewakili dua arah belokan yang
konsisten dalam frame ego; `kappa` mendekati nol berarti lurus. Nilai ini harus
di-clip ke batas konfigurasi, kemudian dinormalisasi sebelum masuk network.

Tes minimum:

- straight menghasilkan nilai mendekati nol;
- belok kiri dan kanan menghasilkan tanda berlawanan;
- rotasi tile di peta tidak mengubah makna ego-relative;
- nilai finite tersedia dekat batas antar-tile;
- fitur 3-kelas lama tetap memberikan hasil yang sama untuk baseline tabular.

### 2.2 State kontinu

State network yang disarankan adalah

\[
x_t = [
\bar d,
\bar\phi,
\bar v,
\bar\kappa,
m_{stop},
\bar d_{stop},
\sigma_{stop},
m_{duck},
\bar x^{duck}_{rel},
\bar z^{duck}_{rel},
\bar v^{duck}_{x,rel},
\bar v^{duck}_{z,rel},
c_{duck},
q_{duck}
].
\]

Makna fitur:

- `d`, `phi`, dan `v` adalah lane-relative state aktual;
- `kappa` adalah signed continuous curvature look-ahead;
- `m_stop` adalah `stop_present`;
- `d_stop` adalah jarak stop yang dinormalisasi;
- `sigma_stop` menyatakan kewajiban full stop sudah dipenuhi;
- `m_duck` adalah `duck_present`;
- posisi dan velocity Duckie dinyatakan pada frame lane ego;
- `c_duck` menyatakan Duckie sedang aktif menyeberang;
- `q_duck` menyatakan controller masih dapat memulai crossing pada episode itu.

Heading error saat ini memang di-clip ke `[-pi/2, pi/2]`, sehingga tidak
memiliki diskontinuitas wrap-around di `-pi`/`pi`. Default encoder cukup memakai
`phi/(pi/2)`. Encoding `[sin(phi), cos(phi)]` boleh dijadikan ablation
normalisasi, tetapi tidak boleh dijustifikasi sebagai perbaikan wrap-around.

#### Nilai saat objek tidak ada

Mask wajib membedakan objek absent dari objek tepat di posisi ego:

- stop absent: `m_stop=0`, `d_stop_normalized=1`;
- Duckie absent: `m_duck=0`, posisi relatif diisi sentinel jarak maksimum dan
  velocity relatif nol;
- stop/Duckie present: mask bernilai 1 dan geometri berisi nilai aktual yang
  sudah di-clip serta dinormalisasi.

Eksperimen `small_loop` saat ini mengasumsikan satu Duckie. Jika kelak ada lebih
dari satu, encoder harus memilih ancaman relevan terdekat secara deterministik
atau memakai fixed top-K slots; jangan membuat panjang observation berubah.

#### Markov state controller Duckie

Posisi Duckie saja belum membedakan keadaan sebelum crossing pertama dan
keadaan setelah crossing selesai ketika `max_crossings_per_episode=1`.
Controller memakai `pedestrian_active` dan `crossings_started`, sehingga phase
yang memengaruhi transisi berikutnya juga harus terlihat oleh policy. Fitur
`c_duck` dan `q_duck` menutup hidden-state gap tersebut.

Kedua fitur ini juga menandai batas privileged MDP dan visuomotor POMDP.
Kamera dapat memberi petunjuk apakah Duckie sedang bergerak, tetapi tidak dapat
melihat counter internal `crossings_started` atau mengetahui dengan pasti apakah
controller masih akan memulai crossing. Pada fase visuomotor, `q_duck` menjadi
hidden state/intent yang harus diestimasi dari history, bukan target observation
yang diasumsikan tersedia langsung.

#### Action sebelumnya dan smoothness

Reward utama M7-M9 **tidak menambah** smoothness penalty supaya reward tetap
sama dengan baseline tabular. Jika ablation kemudian memakai

\[
r_{smooth}=-\lambda\lVert a_t-a_{t-1}\rVert^2,
\]

maka `a_(t-1) = [v_cmd_(t-1), omega_cmd_(t-1)]` wajib ditambahkan ke state.
Tanpa itu reward bukan fungsi Markov dari state/action saat ini. Action
sebelumnya juga boleh dimasukkan sejak awal bila actuator dynamics ternyata
memerlukannya, tetapi keputusan itu harus dicatat sebagai perubahan state.

### 2.3 Continuous action

SAC menghasilkan dua nilai sebelum scaling:

\[
u_t=\tanh(z_t),\qquad u_t\in[-1,1]^2.
\]

Nilai tersebut dipetakan menjadi

\[
v_{cmd}=\frac{u_v+1}{2}v_{max},\qquad
\omega_{cmd}=u_\omega\omega_{max}.
\]

Dengan demikian `v_cmd` berada pada `[0, v_max]` dan `omega_cmd` pada
`[-omega_max, omega_max]`. Perintah kemudian dikonversi ke wheel commands oleh
fungsi kinematika yang sama. `v_cmd` adalah command magnitude yang masih
diskalakan simulator, bukan klaim kecepatan fisik dalam meter per detik.

Brake direpresentasikan oleh `v_cmd` mendekati nol. Evaluasi perlu memeriksa
apakah policy menyalahgunakan kombinasi `v_cmd` nol dan `omega_cmd` besar untuk
berputar di tempat.

### 2.4 Reward dan termination

Istilah "reward yang sama dengan tabular" di dokumen ini secara spesifik
berarti reward **full-task Q-learning tanpa teacher** pada
`configs/small_loop_stop_duck_q_no_teacher.yaml`, bukan semua config tabular.
Config teacher-guided lama memakai koefisien pedestrian dan stagnation nol,
sedangkan no-teacher mengaktifkan keduanya agar collision yang jarang dan brake
permanen mempunyai sinyal belajar langsung.

Reward kanonis untuk perbandingan teacher-free Q-learning versus teacher-free
SAC adalah:

\[
r_t =
\alpha_p v_t\cos(\phi_t)
-\alpha_d d_t^2
-\alpha_\phi\phi_t^2
-c_{step}
+r_{pedestrian}
+r_{stagnation}
+r_{events}.
\]

Ini menjaga perbandingan solver tetap masuk akal. Smoothness hanya boleh masuk
sebagai ablation yang diberi label reward berbeda.

| Komponen | Nilai kanonis | Kondisi |
|---|---:|---|
| `alpha_progress` | 1,0 | selalu |
| `alpha_lateral` | 2,0 | selalu |
| `alpha_heading` | 0,5 | selalu |
| `step_cost` | 0,01 | selalu |
| `collision_duck` | -200 | event collision Duckie |
| `other_collision` | -200 | event collision objek lain |
| `offroad` | -200 | event keluar drivable area |
| `stop_violation` | -40 | melewati stop tanpa full stop |
| `full_stop` | +15 | bonus one-shot saat stop sah |
| `duck_yield` | 0 | crossing dan `v < 0,04` |
| `duck_unsafe` | -5 | crossing dan `v >= 0,04` |
| `unnecessary_stop` | -2 | diam tanpa crossing/kewajiban stop |
| `goal` | +50 | goal aktif dan tercapai |

Dengan demikian `r_pedestrian` dan `r_stagnation` bukan suku baru SAC. Keduanya
sudah ada di `src/reward.py` dan aktif pada baseline Q-learning tanpa teacher.
Perbandingan dengan artefak teacher-guided lama harus diberi label sebagai
perbandingan **solver + exploration assistance + reward regime**, bukan efek
solver murni.

`duck_yield=0` adalah keputusan sengaja, bukan placeholder. Yield yang benar
berarti agent bebas dari penalti `duck_unsafe`; ia tidak menerima bonus positif
per step. Ini mencegah reward farming dengan berhenti selama mungkin di depan
Duckie.

#### Changelog reward dari desain awal

`configs/default.yaml` menyimpan desain awal, sedangkan eksperimen `small_loop`
yang menjadi baseline kanonis memakai kalibrasi berikut:

- `alpha_lateral: 10 -> 2` dan `alpha_heading: 2 -> 0,5`: penalti geometri awal
  terlalu dominan terhadap progress pada macro-action dan membuat state yang
  masih recoverable sangat negatif. Nilai baru tetap menjaga lane tanpa
  mendorong policy diam karena takut bergerak;
- `collision_duck: -100 -> -200` serta `offroad/other_collision: -50 -> -200`:
  terminal buruk harus lebih mahal daripada mengakhiri episode cepat untuk
  menghindari akumulasi shaping negatif;
- `stop_violation: -20 -> -40` dan `full_stop: +10 -> +15` pada full task:
  sinyal kepatuhan bersifat sparse dan one-shot, sehingga diperkuat setelah stop
  sign ditambahkan;
- `duck_unsafe: 0 -> -5`: menyediakan sinyal lokal teacher-free sebelum event
  collision yang sangat jarang;
- `unnecessary_stop: 0 -> -2`: menutup reward hack brake permanen setelah
  crossing selesai.

Perubahan ini adalah reward calibration yang lahir dari ablation `small_loop`,
bukan perubahan definisi task. Config lengkap tetap menjadi sumber kebenaran
agar setiap angka dapat direproduksi.

Termination dan truncation juga tidak berubah:

- terminal: off-road, collision Duckie, collision objek lain, atau goal;
- truncation: timeout;
- timeout tetap melakukan bootstrap pada target critic.

### 2.5 Teacher-guided exploration pada baseline tabular

Teacher saat ini adalah heuristic controller berbasis privileged `RawState`,
bukan PID kontinu, bukan bagian transition model, dan bukan bagian reward.
Teacher membaca `d`, `phi`, curvature, jarak/flag stop, serta kelas ancaman
Duckie, lalu:

1. memilih brake untuk `CROSSING_FAR` atau `CROSSING_NEAR`;
2. memilih brake bila stop line sudah dekat dan `sigma_stop` belum terpenuhi;
3. memakai tracking error `e = phi + d_gain*d` untuk koreksi slow-left/right;
4. bila error kecil, memilih aksi berdasarkan curvature atau fast-straight.

Pada episode `k`, probabilitas override adalah

\[
\beta_k =
\begin{cases}
1, & k < E_{full},\\
1-\frac{k-E_{full}}{E_{decay}}(1-\beta_{min}),
& E_{full}\leq k < E_{full}+E_{decay},\\
\beta_{min}, & \text{setelahnya}.
\end{cases}
\]

Untuk full-task Q-learning yang teacher-guided, `E_full=100`,
`E_decay=200`, dan `beta_min=0`. Pada setiap decision step:

1. student lebih dahulu memilih action epsilon-greedy;
2. dengan probabilitas `beta_k`, teacher mengganti action tersebut;
3. simulator mengeksekusi action hasil akhir;
4. transition dan action yang benar-benar dieksekusi dipakai dalam Q-update.

Schedule epsilon student tetap maju pada setiap decision step, termasuk ketika
action akhirnya dioverride teacher. Jadi config harus mencatat schedule
`epsilon` dan `beta` secara terpisah untuk mereproduksi behavior policy.

Secara efektif behavior policy Q-learning adalah campuran teacher,
epsilon-random, dan greedy student. Hal ini sah karena Q-learning off-policy dan
targetnya tetap `max_a Q(s_next,a)`. Pada SARSA, action aktual dari behavior
campuran juga dipakai sebagai `a_next` pada target, sehingga implementasinya
on-policy terhadap behavior campuran saat itu.

`src/evaluate.py` dan renderer tidak memanggil teacher: action selalu dipilih
dengan `greedy=True`. Jadi hasil evaluasi/video yang sudah diarsipkan mengukur
student Q-table, walaupun teacher pernah membantu pengumpulan transition ketika
training.

Keterbatasannya adalah coverage bias: teacher mengurangi kunjungan ke state
buruk sehingga Q-value di pinggir lane atau situasi recovery dapat menerima
lebih sedikit data. Karena itu hasil harus dilabeli `teacher-guided Q-learning`
atau `teacher-guided SARSA`, bukan vanilla solver.

M9 memakai SAC teacher-free. Jika M10 menguji demonstration-assisted SAC,
mekanisme utamanya adalah prefill replay buffer dengan transition teacher dan,
bila perlu, behavior-cloning warm start. Eksperimen itu tidak dicampur dengan
hasil teacher-free dan tidak otomatis memakai live action override.

## 3. Wrapper kontinu

Buat kelas baru, misalnya `DuckieContinuousMDPEnv`; jangan mengubah kontrak
`DuckieMDPEnv` yang dipakai artefak tabular.

Kontrak minimum wrapper baru:

- `action_space = Box(low=[0,-omega_max], high=[v_max,omega_max])`;
- `observation_space = Box(...)` dengan dtype `float32`, shape tetap, dan semua
  nilai sesuai bounds;
- `reset()` mengembalikan encoded continuous state;
- `step([v_cmd, omega_cmd])` langsung memakai converter `v,omega -> wheels`;
- info tetap mencatat raw state, reward terms, termination reason, perintah
  `v/omega`, serta wheel commands;
- seed, frame skip, DuckController, StopTracker, dan evaluasi sama dengan
  baseline.

Tambahkan assertion bahwa setiap observation finite, dtype `float32`, berada
dalam `observation_space`, dan panjangnya tidak berubah ketika stop/Duckie
absent.

## 4. Compatibility spike sebelum training

Environment yang terverifikasi sekarang adalah Python 3.9.15, Gym 0.23.1, dan
NumPy 1.20.0. Torch, Stable-Baselines3, Gymnasium, dan Shimmy belum terpasang.
Selain itu, SB3 1.8.0 mendeklarasikan Gym 0.21, bukan Gym 0.23.1. Karena itu
versi library tidak boleh dipilih hanya dari asumsi API.

Spike dilakukan pada environment terpisah dan harus lulus seluruh kriteria:

1. import Gym-Duckietown dan solver tanpa dependency error;
2. validasi action/observation space;
3. reset dan 1.000 random transition tanpa NaN atau shape mismatch;
4. train SAC minimal 1.000 transition;
5. save, load, dan deterministic inference menghasilkan action valid;
6. evaluasi 10 episode lane-only selesai dan termination reason tercatat;
7. environment tabular asli tetap lulus seluruh test.

Urutan kandidat:

1. uji adapter Gymnasium/SB3 pada environment eksperimen terpisah;
2. jika adapter atau dependency lama rapuh, implementasikan SAC PyTorch lokal
   dengan test target equation, tanh log-prob correction, replay buffer, dan
   checkpoint round-trip;
3. jangan memasang SB3 1.8.0 dengan memaksa dependency secara diam-diam ke
   environment utama.

Pemilihan final dicatat setelah spike, bukan sebelum ada bukti runtime.

### 4.1 Hyperparameter awal SAC

Nilai berikut dibekukan sebagai starting config M8/M9. Perubahan setelah smoke
test harus menghasilkan config baru dan alasan keep/discard pada log eksperimen.

| Parameter | Nilai awal |
|---|---:|
| actor network | MLP `[256, 256]`, ReLU |
| twin critic network | dua MLP `[256, 256]`, ReLU |
| actor learning rate | `3e-4` |
| critic learning rate | `3e-4` |
| entropy learning rate | `3e-4` |
| optimizer | Adam |
| discount `gamma` | `0.99` |
| Polyak coefficient `tau` | `0.005` |
| replay capacity | `300000` transition |
| batch size | `256` |
| random warm-up | `5000` decision step |
| learning starts | decision step ke-`5000` |
| train frequency | setiap `1` decision step |
| gradient steps | `1` per train event |
| target update interval | `1` critic update |
| entropy coefficient | automatic, initial `0.2` |
| target entropy | `-dim(A) = -2` |
| reward scale | `1.0` |
| n-step return | `1` |
| checkpoint interval | `25000` step lane, `50000` step full task |
| evaluation interval | `10000` step lane, `25000` step full task |
| independent training seeds | minimal `3` |

`environment step` di tabel berarti satu decision/macro-action MDP, bukan satu
physics tick. Dengan `frame_skip=6`, satu decision step menjalankan enam physics
step. Kedua angka harus dicatat di log agar budget tidak ambigu.

Seluruh baseline aktif dan artefak ablation kanonis—Q-learning/SARSA,
teacher-guided maupun teacher-free, lane maupun full task—dilatih dengan
`frame_skip=6`. Hanya template historis `configs/default.yaml` yang masih
memakai nilai 1 dan template itu tidak menjadi pembanding M9. Evaluasi ulang
Q-table dan training SAC wajib tetap memakai `frame_skip=6`.

Budget maksimum per training seed:

| Curriculum | Budget decision step | Initialization |
|---|---:|---|
| M8 lane smoke | `10000` | random network |
| M9 lane-only | `300000` | random network |
| M9 stop sign | `300000` | best lane checkpoint |
| M9 stop + Duckie | `1000000` | best stop checkpoint |

Budget adalah batas maksimum, bukan janji bahwa policy pasti konvergen. Early
stop hanya boleh dilakukan setelah tiga evaluasi berturut-turut melewati bar
kelulusan. Untuk sample-efficiency, laporkan `steps-to-threshold` selain hasil
akhir pada fixed budget.

## 5. Milestone dan acceptance criteria

### M7 — Continuous privileged state

- Tambah signed continuous curvature tanpa mengubah output tabular lama.
- Tambah extractor geometri Duckie, mask absent, dan controller phase.
- Tambah encoder continuous dengan normalization dari config.
- Unit test seluruh edge case absent/present, rotasi tile, dan crossing phase.

Lulus bila seluruh test lama tetap hijau dan encoder baru selalu berada dalam
observation bounds.

### M8 — Continuous wrapper dan SAC smoke test

- Buat wrapper `Box` terpisah.
- Jalankan compatibility spike.
- Jalankan lane-only smoke train 10 episode, save/load, dan deterministic eval.

Lulus bila tidak ada NaN, action selalu valid, checkpoint dapat diputar ulang,
dan pipeline video/evaluation dapat membaca policy baru.

### M9 — Training SAC penuh

Urutan curriculum tetap:

1. lane following;
2. stop sign;
3. stop sign + Duckie crossing.

Evaluasi memakai held-out seeds, teacher nonaktif, deterministic action, reward
yang sama, serta metrik yang sama dengan baseline: timeout/off-road/collision,
mean dan p95 `|d|`, progress, stop compliance, false stop, brake ratio, minimum
Duckie distance, yield behavior, dan task success.

Evaluasi development boleh memakai 30 episode. Keputusan lulus final memakai
100 episode dari manifest seed held-out yang dibekukan, karena pada 30 episode
satu collision sudah berarti 3,33% dan tidak cukup granular untuk bar 1%.

#### Bar kelulusan lane-only

| Metrik | Ambang |
|---|---:|
| timeout/survival rate | `>= 95%` |
| off-road rate | `<= 5%` |
| other-collision rate | `<= 1%` |
| mean `|d|` | `<= 0,08 m` |
| p95 `|d|` | `<= 0,15 m` |
| mean forward progress | `>= 5,0 m` |
| brake/near-zero-speed ratio | `<= 5%` |
| spin-in-place abuse rate | `<= 1%` |

Spin-in-place abuse harus didefinisikan di evaluator sebagai proporsi decision
step dengan `v_cmd` mendekati nol tetapi `|omega_cmd|` besar. Threshold numerik
deteksinya disimpan di config evaluasi.

#### Bar kelulusan full task

| Metrik | Ambang |
|---|---:|
| Duckie collision rate | `<= 1%` |
| other-collision rate | `<= 1%` |
| off-road rate | `<= 5%` |
| stop compliance | `>= 95%` |
| false-stop rate | `<= 5%` |
| task success rate | `>= 90%` |
| mean forward progress | `>= 5,0 m` |
| resume-after-clear rate | `>= 95%` |

Sebelum M9, evaluator harus menambahkan `false_stop_rate`, minimum distance ke
Duckie, `resume_after_clear_rate`, dan spin-in-place rate secara eksplisit.
Task success tetap menolak policy yang hanya brake sampai timeout.

#### Perbandingan ilmiah, bukan acceptance yang bias

Perbandingan solver utama adalah **teacher-free SAC versus teacher-free
Q-learning** dengan reward kanonis, map, horizon, frame skip, DuckController,
dan 100 held-out episode yang identik. Q-table lama tidak perlu dilatih ulang,
tetapi harus dievaluasi ulang pada manifest 100 seed tersebut. Laporkan mean
return, p95 `|d|`, safety/task metrics, `steps-to-threshold`, dan interval
kepercayaan bootstrap.

SAC tidak diwajibkan mengalahkan Q-learning sebagai syarat agar eksperimennya
sah. Bar di atas menguji kelayakan engineering; apakah SAC mengungguli baseline
adalah hipotesis empiris. Jika SAC kalah setelah sanity check dan budget yang
adil, hasil negatif itu tetap dilaporkan. Teacher-guided tabular hanya
dibandingkan dengan assisted SAC pada track terpisah agar pengaruh solver tidak
tercampur dengan bantuan eksplorasi.

Setiap hasil harus menyimpan config, seed, checkpoint best/final, log training,
evaluation JSON, dan video representatif.

#### Definisi best checkpoint

Checkpoint dipilih hanya dari evaluation-development seeds, bukan final
held-out seeds. Pemilihannya lexicographic agar tidak ad-hoc:

- lane-only: task/survival success tertinggi, lalu failure rate terendah, mean
  return tertinggi, kemudian p95 `|d|` terendah;
- full task: task success tertinggi, lalu total collision/off-road terendah,
  stop compliance tertinggi, false-stop rate terendah, mean return tertinggi,
  kemudian p95 `|d|` terendah.

Jika seluruh nilai sama, checkpoint dengan decision step lebih kecil dipilih.
`best` dan `final` disimpan terpisah; final evaluation dijalankan sekali setelah
checkpoint selection dibekukan.

### M10 — Ablation opsional

- DQN: continuous privileged state dengan tujuh action diskrit;
- raw normalized `phi` vs `[sin(phi), cos(phi)]`;
- categorical curvature vs signed continuous curvature;
- tanpa vs dengan action sebelumnya;
- tanpa vs dengan smoothness reward, dengan label reward yang berbeda;
- teacher-free SAC vs initialization/demonstration-assisted SAC bila kelak
  diperlukan.

DQN tidak menjadi blocker M9. Ia hanya menjawab pertanyaan ilmiah tentang efek
continuous state ketika action tetap diskrit.

## 6. Urutan eksekusi yang disetujui

1. Bekukan dan pertahankan artefak tabular sebagai baseline.
2. Implementasikan serta uji M7.
3. Jalankan M8 di environment terisolasi dan pilih jalur library berdasarkan
   bukti smoke test.
4. Selesaikan SAC lane-following sebelum mengaktifkan stop/Duckie.
5. Train full task dan evaluasi pada protokol yang sama.
6. Kerjakan DQN atau smoothness hanya sebagai ablation setelah M9 stabil.
