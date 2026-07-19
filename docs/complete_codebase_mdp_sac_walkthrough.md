# Walkthrough Lengkap Codebase Duckie MDP: Tabular sampai SAC Kontinu

Dokumen ini menjelaskan implementasi aktual di repository ini dari ujung ke
ujung. Urutannya adalah formulasi masalah, environment, state, action, reward,
solver tabular, solver kontinu SAC, konfigurasi YAML, training, evaluation,
checkpoint selection, dan rendering video.

Dokumen ini membedakan tiga hal yang sering tercampur:

1. **latent state simulator**, yaitu seluruh keadaan dunia yang disimpan
   Gym-Duckietown;
2. **state/observation policy**, yaitu bagian keadaan yang diberikan kepada
   solver; dan
3. **solver**, yaitu algoritma yang belajar memilih action dari state tersebut.

Policy saat ini masih **privileged-state policy**, bukan visuomotor policy.
Kamera hanya dipakai untuk video. Tahap visuomotor/POMDP dijelaskan pada bagian
akhir.

---

## 1. Peta codebase

| File | Peran utama |
|---|---|
| `configs/*.yaml` | Seluruh parameter eksperimen, environment, reward, solver, training, dan evaluation |
| `src/state.py` | Ekstraksi `RawState` privileged untuk solver tabular |
| `src/discretizer.py` | Mengubah `RawState` menjadi indeks finite Q-table |
| `src/actions.py` | Action diskrit dan konversi `(v, omega)` ke command roda |
| `src/reward.py` | Reward kanonis, event flags, dan memori/dwell stop sign |
| `src/duck_controller.py` | Injeksi stop sign/Duckie dan logika pedestrian crossing |
| `src/env_wrapper.py` | MDP diskrit: reset, step, transition, reward, terminal, dan truncation |
| `src/lane_teacher.py` | Teacher heuristic opsional untuk guided exploration tabular |
| `src/agents/q_learning.py` | Solver tabular Q-learning |
| `src/agents/sarsa.py` | Solver tabular SARSA |
| `src/transition_model.py` | Estimasi sparse empiris `P(s'|s,a)` dari pengalaman |
| `src/value_iteration.py` | Value Iteration pada transition model empiris |
| `src/train.py` | Training loop Q-learning |
| `src/train_sarsa.py` | Training loop SARSA |
| `src/evaluate.py` | Evaluasi greedy solver tabular |
| `src/continuous_state.py` | State kontinu 15 dimensi dan normalisasinya |
| `src/continuous_env.py` | Wrapper action/observation kontinu untuk SAC |
| `src/agents/sac.py` | Actor, twin critic, target critic, replay buffer, dan update SAC |
| `src/train_sac.py` | Training loop SAC teacher-free dan logging W&B |
| `src/evaluate_sac.py` | Evaluasi deterministic SAC dan metrik task lengkap |
| `src/select_sac_checkpoint.py` | Memilih checkpoint SAC hanya dari development seeds |
| `src/render_multiview_video.py` | Video dashboard solver tabular |
| `src/render_sac_multiview_video.py` | Video dashboard SAC, critic probes, dan repeated Duckie |
| `src/agents/dqn.py`, `src/train_dqn.py` | Scaffold DQN untuk state kontinu + action diskrit; bukan policy utama saat ini |
| `tests/` | Regression test state, reward, solver, spawn, crossing, dan renderer |

Alur umum satu interaction adalah:

```text
YAML
  -> build environment
  -> reset: sample initial state
  -> policy menerima state/observation
  -> policy memilih action
  -> action dikonversi ke command roda
  -> simulator menjalankan fisika
  -> wrapper membentuk next state, event, reward, terminal/truncation
  -> solver melakukan update ketika training
```

---

## 2. Formulasi formal MDP

Environment diformulasikan sebagai:

```text
M = (S, A, P, R, gamma, rho_0, T)
```

- `S`: state space.
- `A`: action space.
- `P(s' | s,a)`: transition dynamics.
- `R(s,a,s')`: reward.
- `gamma`: discount factor.
- `rho_0`: initial-state distribution dari `reset()`.
- `T`: aturan terminal dan truncation.

### 2.1 Latent state simulator

Simulator mengetahui lebih banyak daripada policy, misalnya:

- posisi global ego `(x,z)`;
- heading global `psi`;
- kecepatan dan pose fisik;
- geometri seluruh lane/tile;
- pose, heading, velocity, dan fase Duckie;
- lokasi/orientasi stop sign;
- counter physics dan collision geometry.

Seluruh informasi ini merupakan keadaan dunia sebenarnya. Namun memasukkan
`x,z` langsung ke policy membuat policy menghafal lokasi peta. Karena itu
policy memakai representasi lane-relative yang lebih kecil dan lebih mudah
digeneralisasi.

### 2.2 Transition `P`

Code tidak menyimpan matriks transition penuh. Gym-Duckietown bertindak sebagai
**generative transition model**:

```text
(s_t, a_t) -- env.step(a_t) --> (s_(t+1), r_t, done, info)
```

Transition mencakup:

- kinematika differential drive;
- simulasi fisika dan collision;
- action holding melalui `frame_skip`;
- gerakan Duckie;
- perubahan jarak stop sign dan status stop;
- terminal atau timeout.

Jika `frame_skip = 6`, satu keputusan policy ditahan selama enam physics tick.
Karena simulator berjalan sekitar 30 Hz, satu keputusan berlangsung sekitar:

```text
decision_dt = 6 / 30 = 0.2 detik
```

Training, evaluation, dan deployment harus memakai semantik action-hold yang
sama.

### 2.3 Initial-state distribution `rho_0`

`DuckieMDPEnv.reset()` mendefinisikan `rho_0`. Simulator me-reset pose lalu
wrapper hanya menerima spawn yang memenuhi curriculum, contohnya:

- `abs(d) <= spawn_max_abs_d`;
- `abs(phi) <= spawn_max_abs_phi`;
- berada pada rectangle `spawn_position_bounds_xz`, bila diaktifkan;
- arah sirkulasi clockwise/counterclockwise sesuai config.

Spawn yang salah arah penting untuk ditolak. Policy clockwise yang diletakkan
di lane berlawanan tidak akan pernah melihat stop sign yang ditujukan untuk
arah perjalanannya.

### 2.4 Terminal dan truncation

Wrapper menghasilkan alasan akhir yang eksplisit:

| Alasan | `terminated` | `truncated` | Makna |
|---|---:|---:|---|
| `duck_collision` | 1 | 0 | Tabrakan pedestrian |
| `other_collision` | 1 | 0 | Tabrakan objek selain Duckie |
| `offroad` | 1 | 0 | Pose tidak lagi pada area drivable |
| `goal` | 1 | 0 | Goal tile tercapai, bila didefinisikan |
| `timeout` | 0 | 1 | Horizon eksperimen habis |

Timeout bukan absorbing physical failure. Karena itu TD target tetap
bootstrap pada timeout, tetapi tidak bootstrap pada terminal nyata.

---

## 3. MDP tabular: state diskrit

### 3.1 `RawState`

`src/state.py` mendefinisikan:

```python
RawState(d, phi, v, tile, d_stop, sigma_stop, duck)
```

Secara matematis:

```text
s_raw = (d, phi, v, kappa_class, d_stop, sigma_stop, h_duck)
```

| Komponen | Makna | Nilai/sumber |
|---|---|---|
| `d` | Error lateral dari centerline lane | `get_lane_pos2(...).dist`, clip `[-0.25,0.25]` m |
| `phi` | Error heading terhadap tangent lane | `get_lane_pos2(...).angle_rad`, clip `[-pi/2,pi/2]` |
| `v` | Kecepatan aktual ego | `env.speed`, minimum 0 |
| `tile` | Curvature ego-relative di depan | `STRAIGHT`, `CURVE_LEFT`, `CURVE_RIGHT` |
| `d_stop` | Jarak longitudinal ke stop line relevan | meter atau `None` |
| `sigma_stop` | Apakah full stop untuk sign saat ini sudah dipenuhi | boolean |
| `duck` | Kelas ancaman pedestrian | 5 kelas `DuckThreat` |

Posisi global `(x,z)` tidak masuk Q-table. Ia hanya dipakai simulator,
visualisasi, spawn filtering, dan logging trajectory.

### 3.2 Curvature ego-relative

Code tidak langsung mempercayai nama tile `curve_left`/`curve_right`, karena
nama itu berasal dari frame peta. `tile_ahead()` melakukan:

1. mengambil tangent lane ego;
2. mem-probe titik sekitar 0.30 m di depan;
3. memilih directed Bezier curve yang searah ego;
4. menghitung tangent awal dan akhir;
5. memakai tanda cross product untuk menentukan kiri/kanan relatif ego.

**Bezier curve** adalah kurva halus yang dibentuk dari control points. Ia
dipakai map Duckietown untuk merepresentasikan centerline lane. **Tangent**
adalah vektor arah kurva pada satu titik. Perubahan arah tangent menunjukkan
arah dan besar belokan.

### 3.3 Stop sign

`next_stop_candidate()` hanya menerima stop sign yang:

- berada di depan ego;
- lateral terhadap lane ego tidak lebih dari `stop_lateral_limit`;
- orientasinya menghadap ego;
- berjarak tidak lebih dari `stop_max_distance`.

Filter ini mencegah sign milik lane berlawanan masuk ke state.

`sigma_stop` diperlukan untuk Markov property. Pose yang sama dekat stop line
memerlukan keputusan berbeda bergantung pada apakah ego sudah melakukan full
stop. Tanpa flag ini, policy tidak dapat membedakan “belum berhenti” dan
“sudah berhenti, boleh lanjut”.

`StopTracker` juga menyimpan `hold_steps`. Pada config SAC terbaik,
`stop_hold_steps = 3`. Dengan decision time 0.2 detik, dwell valid adalah:

```text
3 keputusan x 0.2 detik = 0.6 detik
```

### 3.4 Pedestrian diskrit

`DuckThreat` memiliki lima nilai:

| ID | Nilai | Arti |
|---:|---|---|
| 0 | `NONE` | Tidak ada Duckie relevan di koridor depan |
| 1 | `SIDE_FAR` | Duckie diam di sisi, jauh |
| 2 | `SIDE_NEAR` | Duckie diam di sisi, dekat |
| 3 | `CROSSING_FAR` | Duckie sedang menyeberang, jauh |
| 4 | `CROSSING_NEAR` | Duckie sedang menyeberang, dekat |

Kelas dihitung dari posisi relatif terhadap forward/right lane frame, jarak,
lebar conflict corridor, dan `pedestrian_active`.

### 3.5 Diskritisasi dan shape Q-table

`src/discretizer.py` mengubah raw state menjadi:

```text
s_bar = (bin(d), bin(e), bin(v), tile,
         bin(d_stop), sigma_stop, duck)

e = phi + d
```

Tracking error `e` menyatukan heading dan lateral correction. Batas bin aktual:

```text
d bins       = [-0.15, -0.05, 0.05, 0.15]       -> 5 indeks
e bins       = [-0.50, -0.10, 0.10, 0.50]       -> 5 indeks
v bins       = [0.04, 0.16]                      -> 3 indeks
tile                                                  3 indeks
d_stop       = absent, >1.0, 0.3..1.0, <0.3     -> 4 indeks
sigma_stop                                            2 indeks
duck                                                  5 indeks
```

State shape dan Q shape wajib:

```text
STATE_SHAPE = (5, 5, 3, 3, 4, 2, 5)
Q_SHAPE     = (5, 5, 3, 3, 4, 2, 5, 7)
```

Ada 9.000 state diskrit dan 63.000 pasangan state-action. Karena itu checkpoint
tabular yang benar harus mempunyai shape `(5,5,3,3,4,2,5,7)`.

---

## 4. Action diskrit

Duckiebot adalah differential drive, bukan mobil Ackermann. Action naturalnya
adalah kecepatan linear dan angular:

```text
a = (v_cmd, omega_cmd)
```

Q-learning dan SARSA memakai tujuh macro-action:

| ID | Nama | `v_cmd` | `omega_cmd` |
|---:|---|---:|---:|
| 0 | `fast_left` | `v_fast` | `+w0` |
| 1 | `fast_straight` | `v_fast` | `0` |
| 2 | `fast_right` | `v_fast` | `-w0` |
| 3 | `slow_left` | `v_slow` | `+w0` |
| 4 | `slow_straight` | `v_slow` | `0` |
| 5 | `slow_right` | `v_slow` | `-w0` |
| 6 | `brake` | `0` | `0` |

Inverse kinematics ke command roda:

```text
u_left  = v_cmd - wheel_base * omega_cmd / 2
u_right = v_cmd + wheel_base * omega_cmd / 2
```

Nilai command kemudian di-clip ke `[-1,1]`. Angka `v_cmd` di code adalah
besaran command simulator, bukan klaim pengukuran m/s fisik yang sudah
dikalibrasi terhadap robot nyata.

Pada lane-only task, action 6 dimask agar policy tidak menemukan solusi palsu
berupa brake selamanya. Pada full task, ketujuh action diaktifkan.

---

## 5. Reward

`src/reward.py` merupakan sumber reward bersama. Reward total saat ini:

```text
r_total = r_progress
        + r_lateral
        + r_heading
        + r_time
        + r_pedestrian
        + r_stagnation
        + r_steering
        + r_events
```

### 5.1 Dense reward

```text
r_progress = alpha_progress * v * cos(phi)
r_lateral  = -alpha_lateral * d^2
r_heading  = -alpha_heading * phi^2
r_time     = -step_cost
```

- Progress memberi kredit bila ego bergerak sejajar lane.
- Lateral penalty menjaga ego dekat centerline.
- Heading penalty menjaga orientasi searah tangent lane.
- Time cost mencegah return gratis dari tidak melakukan apa-apa.

### 5.2 Pedestrian shaping

Ketika Duckie berada pada kelas crossing:

```text
v < duck_yield_speed  -> duck_yield
v >= threshold       -> duck_unsafe
```

Pada config teacher-free, `duck_yield` sengaja dapat bernilai 0. Berhenti yang
benar berarti bebas penalti, bukan bonus yang dapat di-farming. Tetap bergerak
saat crossing menerima `duck_unsafe`, misalnya `-5`.

### 5.3 Stagnation dan false braking

Jika ego diam pada state normal—bukan crossing dan bukan kewajiban stop—reward
menerapkan `unnecessary_stop`. Ini mencegah policy bertahan sampai timeout
dengan full brake.

### 5.4 Steering penalty pada jalan lurus

Untuk continuous SAC, wrapper memberikan `action_omega` dan curvature sebelum
action. Bila jalan lurus:

```text
r_steering = -straight_steer_penalty
             * (abs(omega_cmd) / max_steer_command)^2
```

Pada tabular wrapper, `compute_reward()` dipanggil tanpa curvature/action omega,
sehingga komponen ini nol. Jadi keluarga reward sama, tetapi eksperimen SAC
dapat mengaktifkan komponen action-conditioned yang tidak digunakan baseline
tabular lama.

### 5.5 Event reward

```text
r_events = collision_duck * I(duck collision)
         + other_collision * I(other collision)
         + offroad * I(offroad)
         + stop_violation * I(stop violation)
         + full_stop * I(valid full stop)
         + goal * I(goal)
```

`full_stop` adalah one-shot event. Bonus tidak diberikan terus-menerus selama
ego diam.

### 5.6 Reward config checkpoint SAC yang dipakai

Config long-horizon memakai nilai utama:

| Parameter | Nilai |
|---|---:|
| `alpha_progress` | 1.0 |
| `alpha_lateral` | 2.0 |
| `alpha_heading` | 0.5 |
| `step_cost` | 0.01 |
| `collision_duck` | -200 |
| `other_collision` | -200 |
| `offroad` | -200 |
| `stop_violation` | -40 |
| `full_stop` | +15 |
| `duck_unsafe` | -5 |
| `unnecessary_stop` | -2 |
| `straight_steer_penalty` | 0.5 |

---

## 6. DuckController

`src/duck_controller.py` melakukan dua pekerjaan.

Pertama, ia menyiapkan map task:

- menginjeksi Duckie bila tidak ada;
- mengubah Duckie menjadi objek dinamis;
- menginjeksi stop sign terpisah dari crossing;
- memvalidasi bahwa objek wajib tersedia.

Kedua, `before_step()` menentukan kapan Duckie mulai menyeberang. Crossing
hanya dipicu bila crossing berada di depan ego dan jaraknya masuk trigger
window.

Parameter penting:

- `p_cross`: probabilitas memulai crossing ketika eligible;
- `walk_distance`: panjang lintasan Duckie;
- `trigger_min_ego_distance` dan `trigger_max_ego_distance`;
- `max_crossings_per_episode`;
- `repeat_rearm_distance`.

Untuk repeated Duckie:

```yaml
max_crossings_per_episode: 0
repeat_rearm_distance: 1.0
```

Nol berarti tidak dibatasi. Hysteresis 1 meter berarti Duckie baru boleh aktif
lagi setelah ego meninggalkan crossing, sehingga Duckie tidak langsung
berbalik ketika ego masih berhenti di tempat yang sama.

---

## 7. Solver tabular

### 7.1 Q-learning

Q-learning menyimpan satu angka untuk setiap pasangan `(state, action)`.

Update aktual:

```text
target = r                                      jika terminal
target = r + gamma * max_a' Q(s',a')            selain itu

Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a))
```

Q-learning bersifat **off-policy** karena target memakai action greedy terbaik,
bukan action berikutnya yang benar-benar dipilih behavior policy.

Action training dipilih epsilon-greedy:

- probabilitas `epsilon`: random dari `allowed_actions`;
- sisanya: argmax Q;
- tie antar-Q yang sama dipecahkan secara random.

Pada evaluation, `greedy=True`, sehingga epsilon tidak dipakai.

### 7.2 Teacher pada Q-learning

Teacher bukan bagian environment MDP dan bukan isi Q-table. Ia hanya mengubah
**behavior policy saat training**:

```text
student memilih epsilon-greedy action
    -> dengan probabilitas beta, teacher override action
    -> environment mengeksekusi action hasil akhir
    -> Q-learning meng-update action yang benar-benar dieksekusi
```

Teacher memakai heuristic `phi + d_gain*d`, curvature, stop distance, dan
kelas Duckie. Karena Q-learning off-policy, transition teacher tetap valid
untuk update Q-learning. Namun hasilnya wajib diberi label
`teacher-guided Q-learning`, bukan vanilla Q-learning.

Evaluation selalu teacher-free.

### 7.3 SARSA

SARSA memakai tuple `(S,A,R,S',A')`:

```text
target = r                                      jika terminal
target = r + gamma * Q(s',a')                   selain itu
```

Perbedaannya dengan Q-learning adalah `a'` merupakan action behavior yang
benar-benar akan dipakai, termasuk efek epsilon dan teacher bila aktif. Karena
itu SARSA bersifat on-policy terhadap behavior campuran tersebut.

### 7.4 Transition model dan Value Iteration

Q-learning dan SARSA tidak membutuhkan matriks `P`. Keduanya belajar langsung
dari sampel simulator.

`EmpiricalTransitionModel` adalah opsi tambahan yang menghitung:

```text
P_hat(s'|s,a) = count(s,a,s') / sum_x count(s,a,x)
R_hat          = rata-rata reward outcome tersebut
```

Value Iteration kemudian menjalankan Bellman optimality backup hanya pada
state-action yang pernah diamati:

```text
V(s) = max_a sum_s' P_hat(s'|s,a)
       * [R_hat(s,a,s') + gamma * V(s')]
```

Unobserved state-action tidak diimajinasikan; ia dikeluarkan dari maximization.
Karena modelnya empiris, kualitas Value Iteration dibatasi coverage data.

---

## 8. Formulasi kontinu untuk SAC

Pada SAC, underlying simulator dan task tidak berubah. Yang berubah adalah
observation policy dan action policy.

### 8.1 State kontinu 15 dimensi

`src/continuous_state.py` mendefinisikan urutan tetap:

```text
x = [
  d, phi, v, kappa,
  stop_present, d_stop, sigma_stop,
  duck_present, duck_longitudinal, duck_lateral,
  duck_v_longitudinal_relative, duck_v_lateral_relative,
  duck_active, duck_crossing_available,
  stop_hold_progress
]
```

| Index | Nama | Normalisasi/input network |
|---:|---|---|
| 0 | `d` | `clip(d/0.25, -1, 1)` |
| 1 | `phi` | `clip(phi/(pi/2), -1, 1)` |
| 2 | `v` | `clip(v/max_speed, 0, 1)` |
| 3 | `kappa` | `clip(kappa/max_abs_curvature, -1, 1)` |
| 4 | `stop_present` | 0 atau 1 |
| 5 | `d_stop` | `d_stop/max_stop_distance`; absent sentinel = 1 |
| 6 | `sigma_stop` | 0 atau 1 |
| 7 | `duck_present` | 0 atau 1 |
| 8 | `duck_longitudinal` | dibagi `max_duck_distance`, clip `[-1,1]` |
| 9 | `duck_lateral` | dibagi `max_duck_distance`, clip `[-1,1]` |
| 10 | `duck_v_longitudinal_relative` | dibagi `max_relative_speed` |
| 11 | `duck_v_lateral_relative` | dibagi `max_relative_speed` |
| 12 | `duck_active` | 0 atau 1 |
| 13 | `duck_crossing_available` | 0 atau 1 |
| 14 | `stop_hold_progress` | progres dwell `[0,1]` |

`stop_present` dan `duck_present` adalah mask. Tanpa mask, nilai sentinel dapat
disalahartikan sebagai objek yang benar-benar berada pada posisi tersebut.

`duck_crossing_available` memberi tahu apakah controller masih boleh memulai
crossing. Ini merupakan privileged hidden mode controller. Fitur ini sah untuk
MDP simulator, tetapi kelak tidak tersedia langsung dari kamera dan menjadi
salah satu sumber partial observability.

### 8.2 Curvature kontinu `kappa`

SAC tidak hanya menerima kelas kiri/lurus/kanan. Ia menerima signed curvature:

```text
kappa = signed heading change / arc length
```

Heading change dihitung dari tangent Bezier awal dan akhir menggunakan
`atan2(cross, dot)`. Arc length diperkirakan dengan sampling titik pada kurva.
Tanda `kappa` menyatakan arah, magnitudo menyatakan seberapa tajam tikungan.

### 8.3 Action kontinu

Action space SAC adalah:

```text
A = Box(
  low  = [0, -w0],
  high = [v_fast, +w0]
)
```

Dengan config aktif:

```text
v_cmd     in [0.00, 0.41]
omega_cmd in [-1.50, +1.50]
```

SAC dapat memilih nilai di antara macro-action tabular, misalnya
`[0.243, 0.182]`. Inilah continuous control yang sebenarnya.

---

## 9. SAC: actor, critic, dan cara belajarnya

SAC adalah off-policy actor-critic. Ia tetap belajar nilai Q, tetapi berbeda
dari Q-learning tabular karena action kontinu tidak dapat di-`max` dengan
mencoba tujuh kolom Q-table.

### 9.1 Actor

Actor menerima observation 15 dimensi dan menghasilkan distribusi action.
Arsitektur default:

```text
observation(15)
  -> Linear(15,256) + ReLU
  -> Linear(256,256) + ReLU
  -> mean(2) dan log_std(2)
```

Actor membentuk Gaussian pada latent action, melakukan reparameterized sample,
lalu memakai `tanh` dan scaling ke action bounds:

```text
z ~ Normal(mean, std)
u = tanh(z)
a = u * action_scale + action_bias
```

Saat training, actor sampling secara stochastic untuk eksplorasi. Saat
evaluation, code memakai deterministic mean yang sudah di-squash dan di-scale.

### 9.2 Twin critics

Setiap critic menerima gabungan `(observation, action)`:

```text
[state(15), action(2)] -> MLP -> satu nilai Q
```

Ada dua critic, `Q1` dan `Q2`. Target dan actor memakai nilai minimum:

```text
min(Q1,Q2)
```

Tujuannya mengurangi overestimation bias.

### 9.3 Target critics

SAC juga mempunyai `target1` dan `target2`. Parameter target bergerak perlahan:

```text
theta_target <- (1-tau)*theta_target + tau*theta_online
```

Config memakai `tau = 0.005`.

### 9.4 Critic target

Target SAC:

```text
y = r + gamma*(1-terminated)
        * [min(Q1_target(s',a'), Q2_target(s',a'))
           - alpha*log pi(a'|s')]
```

Entropy term memberi nilai tambahan pada policy yang masih cukup eksploratif.
Timeout mempunyai `terminated=0`, jadi tetap bootstrap.

### 9.5 Loss critic

```text
critic_loss = MSE(Q1(s,a), y) + MSE(Q2(s,a), y)
```

### 9.6 Loss actor

```text
actor_loss = mean(alpha*log pi(a|s) - min(Q1(s,a),Q2(s,a)))
```

Meminimalkan loss ini berarti actor mencari action dengan Q tinggi sambil tetap
mempertahankan entropy yang sesuai.

### 9.7 Temperature `alpha`

`alpha` mengatur trade-off reward versus entropy. Code belajar `log_alpha`
secara otomatis menuju `target_entropy = -2`, sesuai dua dimensi action.

### 9.8 Replay buffer

Replay menyimpan:

```text
(observation, action, reward, next_observation, terminated)
```

Batch diambil random. SAC off-policy sehingga data tidak harus berasal dari
actor versi terbaru. Timeout disimpan sebagai `terminated=0` agar target tetap
bootstrap.

### 9.9 Actor versus critic secara sederhana

- **Actor** menjawab: “pada keadaan ini, command gas dan belok apa yang harus
  saya keluarkan?”
- **Critic** menjawab: “jika pada keadaan ini command tersebut dilakukan,
  seberapa bagus total future return-nya?”
- Critic belajar dari reward dan transition.
- Actor belajar memilih action yang dinilai tinggi oleh critic.
- Pada evaluation, hanya actor yang diperlukan untuk memilih action. Critic
  tetap berguna pada dashboard sebagai diagnosis Q-value.

---

## 10. Apa yang berbeda antara Q-learning dan SAC?

| Aspek | Q-learning tabular | SAC kontinu |
|---|---|---|
| Underlying task | Duckietown MDP | Duckietown MDP yang sama |
| Simulator transition | Gym-Duckietown | Gym-Duckietown yang sama |
| Policy input | State diskrit 7 indeks | Observation float 15 dimensi |
| Action | 7 macro-action | `(v_cmd, omega_cmd)` kontinu |
| Representasi Q | Array shape `(5,5,3,3,4,2,5,7)` | Dua neural critic `Q(s,a)` |
| Policy | Implisit: `argmax_a Q(s,a)` | Neural actor eksplisit |
| Eksplorasi | Epsilon-greedy | Gaussian policy + entropy |
| Replay buffer | Tidak ada pada implementasi tabular | Ada, capacity hingga 300k |
| Target network | Tidak ada | Dua target critic |
| Teacher | Opsional pada training tabular | Tidak dipakai dalam `train_sac.py` |
| Generalisasi | Hanya antar-state dalam bin yang sama | Network dapat menginterpolasi state/action |

Jawaban singkat untuk “apakah reward, state, atau solver yang berubah?” adalah:

- **solver berubah** dari tabular Q-learning menjadi neural actor-critic SAC;
- **state representation berubah** dari bin diskrit menjadi vector kontinu;
- **action berubah** dari tujuh pilihan menjadi dua nilai kontinu;
- **reward family dan task tetap sama**, tetapi koefisien YAML serta komponen
  yang diaktifkan dapat berbeda antar-experiment. SAC aktif memakai steering
  penalty pada ruas lurus, sedangkan baseline tabular tidak.

Karena lebih dari satu faktor berubah, perbandingan tabular-versus-SAC adalah
perbandingan pipeline kontrol, bukan isolasi solver murni kecuali dilakukan
ablation tambahan.

---

## 11. Cara membaca konfigurasi YAML

Satu YAML adalah manifest eksperimen. File yang menghasilkan checkpoint/video
SAC long-horizon saat ini adalah:

```text
configs/sac_full_long_horizon_5min_warmstart_20k_wandb.yaml
```

### 11.1 Header

```yaml
algorithm: sac
stage: full
seed: 73
```

- `algorithm`: solver.
- `stage`: `lane`, `stop`, atau `full`.
- `seed`: reproduksibilitas network, simulator, action space, dan controller.

### 11.2 `environment`

| Key | Fungsi |
|---|---|
| `map_name` | Nama map Gym-Duckietown |
| `domain_rand` | Randomisasi texture/physics |
| `max_steps` | Horizon physics tick |
| `frame_skip` | Physics tick per keputusan policy |
| `render_observations` | Apakah simulator membuat camera observation saat training |
| `accept_start_angle_deg` | Filter internal sudut spawn |
| `spawn_max_abs_d`, `spawn_max_abs_phi` | Curriculum pose awal |
| `spawn_route_direction` | Arah loop yang wajib |
| `spawn_position_bounds_xz` | Warm-start area tertentu |
| `user_tile_start` | Tile spawn bila ditentukan |
| `goal_tile` | Goal eksplisit; `null` berarti survival/progress task |

`max_steps=9000` dan `frame_skip=6` berarti satu episode mempunyai maksimum
1.500 keputusan atau 300 detik simulasi.

### 11.3 `state`

Bagian ini mengatur ekstraksi state bersama:

- filter stop sign;
- stop zone dan pass distance;
- kecepatan full stop;
- jumlah dwell decision;
- lane look-ahead;
- threshold curvature;
- jangkauan dan conflict corridor Duckie.

### 11.4 `continuous_state`

Bagian ini hanya untuk encoder SAC:

- batas normalisasi speed;
- batas curvature;
- batas stop distance;
- batas posisi/velocity relatif Duckie;
- jumlah sample Bezier untuk arc length.

### 11.5 `actions`

Menentukan bounds action sekaligus macro-action tabular:

```yaml
v_fast: 0.41
v_slow: 0.17
w0: 1.50
wheel_base: 0.102
```

### 11.6 `duck_controller`

Menentukan object injection, trigger crossing, batas crossing per episode,
hysteresis repeated crossing, serta posisi stop sign dan Duckie.

### 11.7 `reward`

Semua koefisien `RewardConfig`. Dengan menaruhnya di YAML, perubahan reward
tercatat bersama checkpoint dan dapat direproduksi.

### 11.8 Solver section

Q-learning menggunakan:

```yaml
q_learning:
  gamma: 0.99
  alpha_lr: 0.10
  epsilon_start: 0.50
  epsilon_end: 0.02
  epsilon_decay_steps: 20000
  allowed_actions: [0, 1, 2, 3, 4, 5]
```

SAC menggunakan:

```yaml
sac:
  gamma: 0.99
  tau: 0.005
  actor_lr: 0.0003
  critic_lr: 0.0003
  alpha_lr: 0.0003
  initial_alpha: 0.2
  batch_size: 256
  replay_capacity: 300000
  hidden_size: 256
  target_entropy: -2.0
```

### 11.9 `training`

Untuk tabular:

- `episodes`: budget episode;
- `initial_q_table`: curriculum checkpoint;
- `broadcast_lane_prior`: menyalin lane slice ke konteks stop/duck;
- `checkpoint_every` dan milestone;
- `output_dir`.

Untuk SAC:

- `total_steps`: budget decision environment;
- `random_steps`: warm-up random action;
- `gradient_steps`: update per environment step;
- `checkpoint_interval`;
- `initial_checkpoint`: curriculum warm-start;
- `save_initial_checkpoint`: simpan step-0 untuk perbandingan;
- `device`: `cuda` atau `cpu`;
- `output_dir`.

Ketika warm-start SAC, actor, critics, target critics, alpha, dan optimizer
dimuat. Replay buffer sengaja mulai kosong karena stage dynamics dapat berubah.

### 11.10 `evaluation`

Bagian ini bukan hyperparameter belajar. Ia membekukan protokol pengujian:

- development/final episode count;
- development/final held-out seeds;
- progress minimum;
- brake ratio maksimum;
- threshold untuk mendefinisikan brake, resume, dan spin.

### 11.11 `wandb`

Menentukan entity, project, run name, group, tags, notes, mode, dan apakah model
checkpoint diunggah sebagai artifact. API key tidak disimpan di YAML/source.

---

## 12. Isi training loop

### 12.1 Q-learning: `src/train.py`

Untuk setiap episode:

1. `env.reset(seed + episode)`;
2. `discretize(raw_state)`;
3. student memilih epsilon-greedy action;
4. teacher dapat override bila enabled;
5. `env.step(action)`;
6. bentuk `next_state`;
7. Q-learning update memakai `terminated`, bukan `done` mentah;
8. transition model opsional mengamati sample;
9. log return, termination, stop compliance, teacher ratio, epsilon;
10. simpan checkpoint atomik.

### 12.2 SARSA: `src/train_sarsa.py`

Perbedaannya adalah `next_action` dipilih sebelum TD update dan action tersebut
dibawa ke iterasi berikutnya. Pada timeout, `a'` dipakai untuk bootstrap tetapi
tidak dihitung sebagai action yang benar-benar dieksekusi.

### 12.3 SAC: `src/train_sac.py`

Loop SAC:

1. load YAML dan validasi CUDA;
2. bangun continuous environment;
3. buat actor, critics, targets, alpha, optimizer, replay;
4. load curriculum checkpoint bila ada;
5. pilih random action selama warm-up, lalu stochastic actor action;
6. jalankan `env.step(action)`;
7. masukkan transition ke replay;
8. sample minibatch dan update critic, actor, alpha, target critic;
9. agregasi semua reward component per episode;
10. log task metrics dan reward breakdown ke CSV/W&B;
11. simpan checkpoint berkala dan final.

Training SAC **tidak memanggil lane teacher**. Eksplorasi berasal dari actor
stochastic dan entropy SAC.

### 12.4 Curriculum

Urutan config yang dirancang:

```text
sac_lane.yaml
  -> sac_stop.yaml
  -> sac_full.yaml / config full-task berikutnya
  -> long-horizon recovery config
```

Curriculum berarti bobot solver sebelumnya dipakai sebagai initial checkpoint,
bukan teacher yang mengontrol action saat episode berikutnya.

---

## 13. Apa yang dilakukan evaluation script?

Evaluation hanya **mengukur**. Ia tidak melakukan gradient update, tidak mengisi
Q-table, tidak mengubah checkpoint, dan tidak memakai teacher.

### 13.1 Evaluasi tabular: `src/evaluate.py`

Urutannya:

1. load YAML dan Q-table;
2. buat environment pada held-out seeds;
3. pilih action dengan `greedy=True`;
4. jalankan episode hingga terminal/timeout;
5. agregasi metrik;
6. tulis `evaluation_report.json`.

Metrik utamanya:

- mean return;
- offroad/collision/timeout/goal rate;
- task success;
- stop compliance;
- Duckie encounter dan yield rate;
- mean/p95 absolute lateral error;
- progress, moving ratio, brake ratio, tile transitions.

Tabular task success saat ini mensyaratkan timeout, progress minimum, dan brake
ratio tidak berlebihan. Stop compliance dilaporkan sebagai metrik terpisah.

### 13.2 Evaluasi SAC: `src/evaluate_sac.py`

SAC evaluation memakai:

```python
agent.select_action(observation, deterministic=True)
```

Artinya action adalah mean actor, tanpa sampling entropy. Teacher juga tidak
ada di code path ini.

Metrik tambahan SAC:

- `total_failure_rate`;
- `stop_compliance_rate` dan stop opportunities;
- compliant-stop episode rate;
- completed stop dwell dalam decision dan detik;
- `false_stop_rate`;
- `spin_in_place_rate`;
- `mean_abs_omega_on_straight`;
- `duck_yield_step_rate`;
- `resume_after_clear_rate`;
- minimum Duckie distance.

Full-task SAC success lebih ketat. Satu episode sukses bila:

```text
termination == timeout
AND progress >= success_min_progress_m
AND brake_ratio <= success_max_brake_ratio
AND ada stop opportunity
AND tidak ada stop violation
```

### 13.3 Checkpoint selection

`src/select_sac_checkpoint.py` mengevaluasi semua `sac_step_*.pt` hanya pada
development seeds. Untuk full task, ranking lexicographic adalah:

1. task success lebih tinggi;
2. failure lebih rendah;
3. stop compliance lebih tinggi;
4. false stop lebih rendah;
5. steering lurus lebih halus;
6. mean return lebih tinggi;
7. p95 lateral error lebih rendah.

Final seeds tidak boleh dipakai memilih checkpoint. Final seeds hanya dipakai
sekali setelah checkpoint dibekukan.

---

## 14. Rendering dan dashboard

Renderer tabular dan SAC tidak melakukan training. Ia menjalankan greedy atau
deterministic policy lalu menyusun:

- agent camera;
- BEV full map;
- full-loop vantage;
- trajectory world-frame;
- state, action, event, dan Q/critic diagnostics.

SAC renderer memecah macro-decision menjadi physics tick agar video 20 FPS
halus, tetapi actor hanya memilih action baru setiap `policy_repeat` tick.
`stop_hold_steps` disesuaikan agar dwell dalam detik tidak berubah.

Opsi repeated Duckie hanya mengubah controller saat render:

```bash
--repeat-duck --repeat-rearm-distance 1.0
```

Checkpoint actor tidak dimodifikasi dan tidak di-retrain.

---

## 15. Bukti artefak policy SAC saat ini

Checkpoint aktif:

```text
runs/sac_full_long_horizon_5min_warmstart_20k_wandb/sac_best.pt
```

Evaluasi 30 episode long-horizon yang tersimpan melaporkan:

| Metrik | Hasil |
|---|---:|
| Timeout rate | 100% |
| Offroad rate | 0% |
| Duck collision rate | 0% |
| Other collision rate | 0% |
| Task success | 83.33% |
| Stop compliance | 97.60% |
| Mean progress | 28.05 m |
| Mean `abs(d)` | 0.0627 m |
| p95 `abs(d)` | 0.1048 m |
| Resume after Duckie clear | 100% |

Audit repeated crossing dengan hysteresis 1 meter menghasilkan rata-rata enam
crossing per episode 5 menit pada tiga seed, seluruhnya berakhir timeout dan
tanpa stop violation.

Artefak bukti:

```text
runs/sac_full_long_horizon_5min_warmstart_20k_wandb/final_long_horizon_eval_30.json
runs/sac_full_long_horizon_5min_warmstart_20k_wandb/repeated_duck_hysteresis_audit_3.json
runs/sac_full_long_horizon_5min_warmstart_20k_wandb/sac_best_repeat_duck_multiview_5min_20fps.mp4
```

---

## 16. Perintah penting

### Q-learning

```bash
python -m src.train --config configs/small_loop_lane_q_no_teacher.yaml

python -m src.evaluate \
  --config configs/small_loop_lane_q_no_teacher.yaml \
  --q-table runs/lane_q_no_teacher_explore/q_table.npy
```

### SARSA

```bash
python -m src.train_sarsa --config configs/small_loop_lane_sarsa.yaml
```

### Value Iteration dari transition empiris

```bash
python -m src.value_iteration \
  --config configs/small_loop_lane_vi.yaml \
  --model runs/lane_q_teacher/transition_model.npz \
  --output runs/lane_vi/q_table.npy
```

### SAC training

```bash
.venv-sac/bin/python -m src.train_sac \
  --config configs/sac_full_long_horizon_5min_warmstart_20k_wandb.yaml
```

### SAC evaluation

```bash
.venv-sac/bin/python -m src.evaluate_sac \
  --config configs/sac_full_long_horizon_5min_warmstart_20k_wandb.yaml \
  --checkpoint runs/sac_full_long_horizon_5min_warmstart_20k_wandb/sac_best.pt \
  --episodes 30 \
  --output runs/sac_full_long_horizon_5min_warmstart_20k_wandb/evaluation.json
```

### SAC video dengan Duckie berulang

```bash
.venv-sac/bin/python -m src.render_sac_multiview_video \
  --config configs/sac_full_long_horizon_5min_warmstart_20k_wandb.yaml \
  --checkpoint runs/sac_full_long_horizon_5min_warmstart_20k_wandb/sac_best.pt \
  --output runs/sac_full_long_horizon_5min_warmstart_20k_wandb/repeat_duck.mp4 \
  --seed 30101 \
  --fps 20 \
  --max-steps 9000 \
  --repeat-duck \
  --repeat-rearm-distance 1.0
```

### Tests

```bash
.venv-sac/bin/python -m pytest -q --disable-warnings
```

---

## 17. Hubungan ke POMDP visuomotor

Policy saat ini membaca ground truth simulator. Jika input diganti dengan
kamera, masalah menjadi POMDP:

```text
POMDP = (S, A, P, R, Omega, O, gamma)
```

- `S` tetap latent state fisik yang sama.
- `A` dapat tetap continuous `(v_cmd, omega_cmd)`.
- `P` dan `R` dapat tetap sama.
- `Omega` adalah observation space, misalnya frame kamera + proprioception.
- `O(o|s)` adalah proses pembentukan observation dari latent state.

Satu frame kamera tidak selalu cukup untuk mengetahui velocity Duckie, apakah
full stop sudah dilakukan, atau apakah controller Duckie akan memulai crossing.
Karena itu visuomotor policy sebaiknya menggunakan history dan memory:

```text
image_t -> CNN/ViT encoder -> visual feature
(visual feature, previous action, proprioception) -> GRU/LSTM
hidden state -> actor -> continuous action
```

Privileged SAC saat ini berfungsi sebagai upper bound/oracle kontrol. Tahap
visuomotor dapat membandingkan:

1. privileged-state SAC;
2. single-frame visuomotor SAC;
3. recurrent visuomotor POMDP policy.

Dengan pemisahan ini, kegagalan perception dapat dibedakan dari kegagalan
control solver.

---

## 18. Ringkasan satu kalimat per komponen

- `state.py`: apa yang diketahui policy tabular.
- `discretizer.py`: cara state menjadi alamat Q-table.
- `actions.py`: apa yang boleh dilakukan ego.
- `reward.py`: apa yang dianggap baik atau buruk.
- `duck_controller.py`: kapan pedestrian bergerak.
- `env_wrapper.py`: bagaimana satu MDP transition terjadi.
- Q-learning: belajar nilai setiap state-action dan memilih argmax.
- SARSA: belajar dari action berikutnya yang benar-benar dipilih.
- transition model + VI: mengestimasi model lalu melakukan planning.
- `continuous_state.py`: observation float yang lebih kaya untuk neural policy.
- `continuous_env.py`: action continuous dan reward/terminal yang sama.
- SAC actor: menghasilkan command kontinu.
- SAC critics: menilai command actor.
- replay buffer: menyimpan pengalaman off-policy.
- `train_sac.py`: mengumpulkan pengalaman dan memperbarui network.
- `evaluate_sac.py`: mengukur deterministic policy tanpa belajar.
- renderer: membuat bukti visual dan diagnostic dashboard.
