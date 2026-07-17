# Code Walkthrough: Duckie MDP

Dokumen ini menjelaskan alur code untuk policy tabular Duckietown, mulai dari
konfigurasi sampai training, evaluation, dan video. Dua solver aktif adalah
Q-learning dan SARSA. Keduanya memakai formulasi MDP, state, action, reward,
scene stop sign, dan pedestrian Duckie yang sama.

## 1. Gambaran besar

Alur satu transition adalah:

```text
YAML config
   ↓
build environment
   ↓
reset → raw state
   ↓
discretize → indeks Q-table
   ↓
policy memilih action
   ↓
action → command roda
   ↓
simulator menjalankan fisika
   ↓
state baru + event + reward + terminal
   ↓
Q-learning/SARSA update
```

Formulasi environment:

\[
\mathcal M=(\mathcal S,\mathcal A,P,R,\gamma,\rho_0,\mathcal T)
\]

Gym-Duckietown menjadi generative transition model. Code tidak harus menyimpan
seluruh matriks \(P(s'\mid s,a)\); simulator menghasilkan satu sampel
transition setiap kali `env.step(action)` dipanggil.

## 2. Struktur file

| File | Tanggung jawab |
|---|---|
| `configs/*.yaml` | Parameter eksperimen dan solver |
| `src/state.py` | Mengekstrak raw state dari simulator |
| `src/discretizer.py` | Mengubah state kontinu menjadi indeks Q-table |
| `src/actions.py` | Mendefinisikan tujuh macro-action |
| `src/reward.py` | Reward shaping, event reward, dan stop tracker |
| `src/duck_controller.py` | Injeksi objek dan crossing Duckie |
| `src/env_wrapper.py` | Menyatukan state, action, transition, reward, dan terminal |
| `src/lane_teacher.py` | Guided exploration saat training |
| `src/agents/q_learning.py` | Update tabular Q-learning |
| `src/agents/sarsa.py` | Update tabular SARSA |
| `src/train.py` | Loop training Q-learning |
| `src/train_sarsa.py` | Loop training SARSA |
| `src/evaluate.py` | Evaluasi greedy tanpa teacher |
| `src/render_multiview_video.py` | Video multiview dan diagnostic dashboard |
| `src/transition_model.py` | Estimasi transition empiris opsional |
| `src/value_iteration.py` | Value Iteration dari transition empiris |
| `src/agents/dqn.py` | Scaffold DQN; belum menjadi solver aktif |

## 3. Konfigurasi eksperimen

Full-task menggunakan:

- `configs/small_loop_stop_duck_q.yaml` untuk Q-learning;
- `configs/small_loop_stop_duck_sarsa.yaml` untuk SARSA.

Bagian environment:

```yaml
environment:
  map_name: small_loop
  domain_rand: false
  max_steps: 1500
  frame_skip: 6
  render_observations: false
  accept_start_angle_deg: 10
  spawn_max_abs_d: 0.08
  spawn_max_abs_phi: 0.175
  spawn_attempts: 50
  goal_tile: null
```

Maknanya:

- peta yang dipakai adalah `small_loop`;
- domain randomization dimatikan agar eksperimen reproducible;
- satu episode dibatasi 1.500 physics step;
- satu keputusan policy ditahan selama enam physics step;
- initial state diterima jika \(|d|\leq0.08\) dan
  \(|\phi|\leq0.175\);
- `goal_tile: null` berarti keberhasilan operasional adalah mencapai timeout
  tanpa terminal buruk.

## 4. Pembuatan environment

Factory berada di `src/env_wrapper.py`:

```python
env = build_env(config, seed)
```

`build_env()` membuat `DuckietownEnv`, menyiapkan objek dinamis, kemudian
membungkusnya dengan `DuckieMDPEnv`:

```python
base = DuckietownEnv(...)

if duck_cfg.make_dynamic:
    make_ducks_dynamic(base, duck_cfg)

return DuckieMDPEnv(base, ...)
```

Wrapper menyediakan interface:

```python
reset()       -> raw_state
step(action) -> next_raw_state, reward, done, info
```

## 5. Stop sign dan Duckie

`src/duck_controller.py` memodifikasi salinan `map_data`, sehingga map asli
tidak dimutasi secara langsung.

Duckie diinjeksikan sebagai objek dinamis:

```yaml
spawn_pos: [1.62, 0.50]
walk_distance: 0.90
```

Stop sign diinjeksikan sebagai objek statis:

```yaml
stop_spawn_pos: [1.20, 2.10]
stop_spawn_rotate: 180.0
```

Stop sign berada pada ruas bawah, sedangkan crossing Duckie berada pada ruas
kanan. Pemisahan ini memastikan berhenti untuk stop sign dan berhenti untuk
pedestrian menjadi dua event yang berbeda.

`prepare_task_map_data()` memastikan:

```python
duck["static"] = False
stop["static"] = True
```

## 6. Reset episode dan initial-state distribution

`DuckieMDPEnv.reset()` mendefinisikan \(\rho_0\), initial-state distribution.

Urutannya:

1. Mengatur seed simulator.
2. Mengembalikan Duckie ke posisi awal.
3. Memanggil reset simulator.
4. Mengekstrak raw state.
5. Memeriksa batas curriculum spawn.
6. Mengulang spawn maksimal 50 kali jika pose tidak diterima.
7. Me-reset stop tracker.
8. Menyimpan kandidat stop sign pertama.

Code utamanya:

```python
for _ in range(max(1, self.spawn_attempts)):
    self.env.reset()
    candidate = get_raw_state(self.env, False, self.state_cfg)
    if self._spawn_is_accepted(candidate):
        break
```

## 7. Raw state

State didefinisikan di `src/state.py`:

```python
@dataclass(frozen=True)
class RawState:
    d: float
    phi: float
    v: float
    tile: TileType
    d_stop: Optional[float]
    sigma_stop: bool
    duck: DuckThreat
```

Secara matematis:

\[
s_t=(d_t,\phi_t,v_t,\kappa_t,d_t^{stop},
\sigma_t^{stop},h_t^{duck})
\]

### 7.1 Lateral error `d`

`d` adalah penyimpangan lateral terhadap centerline. Nilainya dipotong ke:

\[
d\in[-0.25,0.25]
\]

### 7.2 Heading error `phi`

`phi` adalah perbedaan orientasi ego dengan tangent lajur:

\[
\phi\in[-\pi/2,\pi/2]
\]

### 7.3 Kecepatan `v`

```python
speed = max(0.0, float(env.speed))
```

Nilai ini adalah kecepatan aktual simulator, bukan hanya command policy.

### 7.4 Curvature `tile`

```python
class TileType(IntEnum):
    STRAIGHT = 0
    CURVE_LEFT = 1
    CURVE_RIGHT = 2
```

`tile_ahead()` mengambil titik probe 0.3 meter di depan. Directed Bezier curve
yang sesuai dengan arah ego dipilih, kemudian tanda cross product tangent
menentukan arah belokan ego-relative. Ini menghindari penggunaan label tile
peta yang dapat ambigu terhadap arah masuk kendaraan.

### 7.5 Jarak stop sign `d_stop`

`next_stop_candidate()` memfilter setiap stop sign berdasarkan:

- berada di depan ego;
- lateral distance maksimal 0.40 meter;
- orientasinya menghadap ego;
- jaraknya maksimal 3 meter.

Jarak line dihitung dengan offset:

```python
distance = max(0.0, ahead - sign_to_line_offset)
```

Jika tidak ada stop sign relevan, `d_stop` bernilai `None`.

### 7.6 Memori kepatuhan `sigma_stop`

`sigma_stop` membedakan keadaan fisik yang sama tetapi memiliki riwayat stop
berbeda:

- `False`: kewajiban berhenti belum dipenuhi;
- `True`: kendaraan sudah melakukan full stop untuk sign aktif.

Komponen ini diperlukan agar state lebih mendekati sifat Markov.

### 7.7 Duck threat `duck`

```python
class DuckThreat(IntEnum):
    NONE = 0
    SIDE_FAR = 1
    SIDE_NEAR = 2
    CROSSING_FAR = 3
    CROSSING_NEAR = 4
```

Duckie harus berada di depan, berada di corridor ego, dan tidak melewati jarak
maksimum. `pedestrian_active` membedakan Duckie yang hanya berada di sisi jalan
dengan Duckie yang benar-benar menyeberang.

Config full-task:

```yaml
duck_max_distance: 1.20
duck_near_distance: 0.60
duck_corridor_width: 0.60
```

## 8. Pemicu pedestrian crossing

Sebelum setiap transition, wrapper memanggil:

```python
self.duck_controller.before_step()
```

Crossing hanya aktif jika titik crossing masih di depan dan jarak ego berada
dalam rentang:

```yaml
trigger_min_ego_distance: 0.35
trigger_max_ego_distance: 0.45
p_cross: 1.0
```

Karena `p_cross=1.0`, setiap kondisi eligible selalu memicu crossing.

## 9. Diskritisasi state

`src/discretizer.py` mengubah raw state menjadi:

\[
\bar{s}=(b_d(d),b_e(\phi+d),b_v(v),\kappa,
b_s(d_{stop}),\sigma_{stop},h_{duck})
\]

Shape state dan Q-table:

```python
STATE_SHAPE = (5, 5, 3, 3, 4, 2, 5)
Q_SHAPE = STATE_SHAPE + (7,)
```

Total state:

\[
5\times5\times3\times3\times4\times2\times5=9000
\]

Total state-action entry:

\[
9000\times7=63000
\]

### 9.1 Lateral bins

```python
D_BINS = [-0.15, -0.05, 0.05, 0.15]
```

### 9.2 Tracking-error bins

Tracking error menggunakan:

\[
e=\phi+d
\]

```python
TRACKING_ERROR_BINS = [-0.50, -0.10, 0.10, 0.50]
```

### 9.3 Velocity bins

```python
V_BINS = [0.04, 0.16]
```

Menghasilkan kategori diam, pelan, dan cepat.

### 9.4 Stop-distance bins

```python
if d_stop is None:
    stop = 0
elif d_stop > 1.0:
    stop = 1
elif d_stop >= 0.3:
    stop = 2
else:
    stop = 3
```

### 9.5 Contoh diskritisasi

Raw state:

```text
d=0.03
phi=0.06
v=0.17
tile=STRAIGHT
d_stop=0.20
sigma_stop=False
duck=CROSSING_NEAR
```

Karena \(e=\phi+d=0.09\), indeks state menjadi:

```python
(2, 2, 2, 0, 3, 0, 4)
```

Policy membaca tujuh nilai berikut:

```python
Q[2, 2, 2, 0, 3, 0, 4, :]
```

## 10. Action space

Action didefinisikan di `src/actions.py`:

| ID | Nama | \(v\) | \(\omega\) |
|---:|---|---:|---:|
| 0 | `fast_left` | 0.41 | +1.50 |
| 1 | `fast_straight` | 0.41 | 0 |
| 2 | `fast_right` | 0.41 | -1.50 |
| 3 | `slow_left` | 0.17 | +1.50 |
| 4 | `slow_straight` | 0.17 | 0 |
| 5 | `slow_right` | 0.17 | -1.50 |
| 6 | `brake` | 0 | 0 |

Duckiebot adalah differential drive. Command linear-angular dikonversi menjadi:

\[
u_L=v-\frac{L\omega}{2}
\]

\[
u_R=v+\frac{L\omega}{2}
\]

Command kemudian dibatasi ke interval \([-1,1]\).

## 11. Satu transition environment

`DuckieMDPEnv.step(action_id)` menjalankan urutan berikut:

1. Menyimpan state dan stop-sign identity sebelumnya.
2. Memperbarui controller Duckie.
3. Mengubah action ID menjadi command roda.
4. Menjalankan fisika selama `frame_skip`.
5. Mengekstrak raw state baru.
6. Memperbarui stop tracker.
7. Memeriksa collision dan goal.
8. Menentukan termination reason.
9. Menghitung custom reward.
10. Mengembalikan hasil transition.

```python
return current, reward.total, done, info
```

`info` berisi:

```text
raw_state
events
reward_terms
simulator_reward
action_id
wheel_commands
termination_reason
terminated
truncated
```

## 12. Reward

Reward full-task adalah:

\[
r_t=v_t\cos(\phi_t)-2d_t^2-0.5\phi_t^2-0.01+r_{event}
\]

Implementasinya berada di `src/reward.py`:

```python
progress = alpha_progress * state.v * cos(state.phi)
lateral = -alpha_lateral * state.d ** 2
heading = -alpha_heading * state.phi ** 2
time = -step_cost
```

Event reward full-task:

| Event | Reward |
|---|---:|
| Full stop | +15 |
| Stop violation | -40 |
| Duck collision | -200 |
| Other collision | -200 |
| Off-road | -200 |
| Goal | +50 |

Tidak ada bonus yield eksplisit. Perilaku pedestrian yielding dipelajari dari
state ancaman, penalti collision, pengalaman guided exploration, dan return
jangka panjang.

## 13. Stop tracker

Full stop terdeteksi ketika:

\[
d_{stop}\leq0.45,\qquad v<0.02
\]

Jika belum pernah berhenti untuk sign aktif:

```python
self.sigma_stop = True
events.full_stop = True
```

Syarat `not self.sigma_stop` memastikan bonus hanya diberikan satu kali. Saat
sign dilewati, code memeriksa apakah `sigma_stop` sudah terpenuhi. Jika belum:

```python
events.stop_violation = True
```

Setelah sign dilewati, tracker kembali ke `False` agar dapat menangani sign
berikutnya.

## 14. Terminal dan truncation

Terminal fisik:

```python
terminated = reason in {
    "duck_collision",
    "other_collision",
    "offroad",
    "goal",
}
```

Time limit:

```python
truncated = reason == "timeout"
```

Terminal sejati memutus bootstrap TD. Timeout hanya batas horizon eksperimen,
sehingga nilainya tetap di-bootstrap.

## 15. Teacher sebagai guided exploration

Teacher berada di `src/lane_teacher.py`. Prioritas action-nya:

1. `brake` jika Duckie sedang crossing;
2. `brake` jika stop sign dekat dan kewajiban belum dipenuhi;
3. koreksi tracking error;
4. mengikuti curvature;
5. `fast_straight` pada kondisi normal.

Schedule full-task:

```yaml
full_control_episodes: 100
decay_episodes: 200
min_probability: 0.0
```

Artinya:

- episode 1–100: teacher penuh;
- episode 101–300: probabilitas turun linear;
- episode 301–400: teacher nol.

Teacher tidak mengisi Q-table secara langsung dan bukan bagian dari MDP. Ia
hanya mengganti behavior action pada sebagian langkah training.

## 16. Q-learning

Entry point:

```bash
python -m src.train --config configs/small_loop_stop_duck_q.yaml
```

Training memuat policy lane-following terlebih dahulu:

```python
agent.load(initial_q_table)
```

Pada setiap step:

```python
state = discretize(raw)
action = agent.select_action(state)

if random < teacher_probability:
    action = select_lane_teacher_action(raw, teacher_cfg)

next_raw, reward, done, info = env.step(action)
agent.update(state, action, reward, next_state, terminated)
```

Update Q-learning:

\[
Q(s,a)\leftarrow Q(s,a)+\alpha
\left[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\right]
\]

Q-learning bersifat off-policy. Jika action berasal dari teacher, entry action
yang benar-benar dieksekusi tetap diperbarui, tetapi target menggunakan action
greedy terbaik pada next state.

## 17. SARSA

Entry point:

```bash
python -m src.train_sarsa --config configs/small_loop_stop_duck_sarsa.yaml
```

SARSA juga dimulai dari policy lane SARSA. Perbedaannya, action berikutnya
dipilih sebelum update:

```python
next_action, next_from_teacher = select_behavior_action(...)

agent.update(
    state,
    action,
    reward,
    next_state,
    next_action,
    terminated,
)
```

Update SARSA:

\[
Q(s,a)\leftarrow Q(s,a)+\alpha
\left[r+\gamma Q(s',a')-Q(s,a)\right]
\]

`next_action` adalah action behavior aktual, termasuk jika dipilih teacher.
Action itu kemudian dibawa ke iterasi berikutnya. Hal ini menjaga implementasi
tetap on-policy.

## 18. Epsilon dan random tie-breaking

Saat training, agent menggunakan epsilon-greedy:

```python
if random < epsilon:
    action = random allowed action
else:
    action = argmax Q(state, action)
```

Jika beberapa action memiliki nilai maksimum yang sama, salah satu dipilih
secara acak. Ini mencegah bias permanen ke action dengan indeks paling kecil.

Saat evaluation, `greedy=True` mematikan epsilon tetapi random tie-breaking
masih dapat memilih di antara action dengan Q-value identik.

## 19. Checkpoint dan training log

Trainer menyimpan:

```text
config.yaml
q_table_checkpoint.npy
training_partial.csv
q_table_ep100.npy
q_table_ep300.npy
q_table_ep400.npy
q_table.npy
training.csv
```

CSV mencatat:

- return dan moving average;
- jumlah decision dan physics step;
- termination reason;
- collision dan off-road;
- stop compliance;
- teacher probability dan teacher step ratio;
- epsilon.

Checkpoint ditulis melalui file temporary lalu di-rename agar lebih aman jika
training dihentikan.

## 20. Evaluation

Entry point:

```bash
python -m src.evaluate --config CONFIG --q-table Q_TABLE
```

`src/agents/factory.py` membaca field `algorithm` dan membuat Q-learning atau
SARSA agent. Evaluator tidak memanggil teacher:

```python
action = agent.select_action(discretize(raw), greedy=True)
```

Metrik yang dihitung:

- mean return;
- timeout, off-road, dan collision rate;
- stop compliance;
- jumlah Duckie crossing encounter;
- pedestrian yield-step rate;
- mean dan p95 lateral error;
- forward progress;
- moving dan brake ratio;
- tile transitions.

Yield-step didefinisikan sebagai kondisi crossing dengan kecepatan aktual
kurang dari 0.04. Ini adalah metrik evaluasi, bukan reward training.

## 21. Video multiview

`src/render_multiview_video.py` menampilkan:

- camera agent;
- bird's-eye view;
- vantage view;
- trajectory;
- raw state;
- discrete state;
- selected action;
- semua Q-values;
- cumulative reward;
- solver dan termination status.

Renderer juga menggunakan greedy policy tanpa teacher.

Untuk mempertahankan frekuensi keputusan policy sambil membuat video halus:

```python
policy_repeat = original_frame_skip
config["environment"]["frame_skip"] = 1
```

Policy memilih action sekali, lalu action yang sama dijalankan sebanyak enam
physics step. Setiap physics step dapat direkam, sehingga video 20 FPS tidak
mempercepat dinamika agent.

## 22. Transition model dan Value Iteration

Transition model opsional menyimpan count:

```text
(state, action, next_state, terminal)
```

Estimasi probabilitas:

\[
\hat P(s'\mid s,a)=
\frac{N(s,a,s')}{\sum_xN(s,a,x)}
\]

Value Iteration menggunakan:

\[
V_{k+1}(s)=\max_a\sum_{s'}
\hat P(s'\mid s,a)
\left[R(s,a,s')+\gamma V_k(s')\right]
\]

Full-task saat ini menetapkan:

```yaml
transition_model:
  enabled: false
```

Value Iteration belum menjadi solver utama karena coverage empirical
state-action model belum cukup tinggi.

## 23. DQN

`src/agents/dqn.py` adalah scaffold opsional. Isinya mencakup:

- encoder privileged state 13 dimensi;
- replay buffer;
- online Q-network;
- target Q-network;
- Huber loss dan satu training step.

DQN belum terhubung ke pipeline karena belum memiliki:

- `train_dqn.py`;
- exploration loop lengkap;
- config DQN;
- save/load model lengkap;
- evaluator dan renderer DQN.

DQN tersebut juga belum visuomotor karena menerima privileged state, bukan
gambar kamera.

## 24. Tests

Folder `tests/` mencakup pemeriksaan:

- konversi action ke roda;
- curvature ego-relative;
- diskritisasi dan tracking error;
- stop bonus one-shot dan stop violation;
- pemisahan posisi stop sign dan Duckie;
- action masking;
- terminal dan timeout bootstrap;
- target Q-learning;
- target on-policy SARSA;
- transition model;
- komposisi multiview renderer.

Status terakhir:

```text
32 passed
```

## 25. Contoh satu step lengkap

Misalkan raw state adalah:

```text
d=0.03
phi=0.06
v=0.17
tile=STRAIGHT
d_stop=0.20
sigma_stop=False
duck=CROSSING_NEAR
```

State diskrit:

```python
state = (2, 2, 2, 0, 3, 0, 4)
```

Policy membaca:

```python
q_values = Q[state]
action = argmax(q_values)
```

Jika action terbaik adalah:

```python
action = 6  # brake
```

command roda menjadi:

```python
[0.0, 0.0]
```

Simulator menjalankan action selama enam physics step. State berikutnya
memiliki kecepatan yang mendekati nol dan tidak terjadi collision. Q-learning
kemudian menggunakan:

\[
r+\gamma\max_{a'}Q(s',a')
\]

sedangkan SARSA menggunakan:

\[
r+\gamma Q(s',a')
\]

Setelah Duckie keluar dari conflict corridor, state ancaman berubah dan greedy
policy kembali memilih action bergerak.

## 26. Status solver

| Solver | Status |
|---|---|
| Q-learning lane | Selesai dan tervalidasi |
| Q-learning full-task | Selesai dan tervalidasi |
| SARSA lane | Selesai dan tervalidasi |
| SARSA full-task | Selesai dan tervalidasi |
| Value Iteration | Implementasi tersedia, coverage model belum memadai |
| DQN privileged-state | Scaffold, belum menjadi eksperimen lengkap |
| Visuomotor policy | Tahap pengembangan berikutnya |

