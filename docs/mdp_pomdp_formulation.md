# Formulasi MDP dan POMDP untuk Visuomotor Policy Gym-Duckietown

## 1. Tujuan tugas

Tugas penuh adalah mengendalikan Duckiebot agar:

1. mengikuti lajur;
2. berhenti pada stop line yang relevan;
3. menghindari duckie/pedestrian yang menyeberang;
4. melanjutkan perjalanan setelah aman; dan
5. mencapai tujuan atau bertahan sampai batas waktu tanpa pelanggaran.

Eksperimen sebaiknya dibangun bertahap: lane following, stop sign, pedestrian,
kemudian task gabungan.

## 2. Formulasi MDP

MDP ditulis sebagai tuple

\[
\mathcal{M}=(\mathcal{S},\mathcal{A},P,R,\gamma,\rho_0,\mathcal{T}).
\]

### 2.1 State sebenarnya

State kontinu yang cukup untuk simulator adalah

\[
s_t = (s_t^{ego},s_t^{lane},s_t^{route},s_t^{stop},s_t^{ped},s_t^{act}).
\]

- Ego: posisi global \((x_t,z_t)\), heading \(\psi_t\), kecepatan linear
  \(v_t\), dan kecepatan sudut \(\omega_t\).
- Lane: offset lateral \(d_t\), error heading \(\phi_t\), serta curvature
  ego-relative sekarang dan look-ahead \((\kappa_t,\kappa_{t+L})\).
- Route: tile/jalur yang sedang diikuti, arah keluar intersection, dan goal.
- Stop: jarak longitudinal ke stop line \(d_t^{stop}\), offset lateral,
  orientasi sign relatif terhadap ego, id stop line, dan flag
  \(\sigma_t^{stop}\) yang menyatakan kewajiban berhenti sudah dipenuhi.
- Pedestrian: posisi relatif longitudinal-lateral
  \((\Delta x_t^{ped},\Delta y_t^{ped})\), kecepatan relatif, apakah berada
  di koridor ego, serta mode gerak seperti diam, mendekat, menyeberang, atau
  menjauh.
- Aktuator: wheel speed aktual dan aksi sebelumnya. Komponen ini penting
  karena motor Duckiebot memiliki delay; pose saja tidak selalu Markov.

Posisi absolut \((x,z)\) boleh disimpan untuk simulator dan logging, tetapi
kurang baik sebagai satu-satunya input policy. Representasi lane-relative
\((d,\phi,\kappa)\) lebih mudah dipindahkan ke peta lain.

Untuk Q-learning lane-following saat ini, state diskrit minimal adalah

\[
\bar{s}_t=(bin(d_t),bin(\phi_t),bin(v_t),class(\kappa_{t+L})).
\]

Dengan 5 bin lateral, 5 bin heading, 3 bin speed, dan 3 kelas curvature,
terdapat \(5\times5\times3\times3=225\) state efektif. Kelas curvature harus
ego-relative: lurus, kiri, kanan. Nama tile `curve_left` pada file peta tidak
boleh langsung dianggap sebagai arah belok ego.

### 2.2 Action

Untuk differential-drive Duckiebot, action yang paling alami adalah

\[
a_t=(v_t^{cmd},\omega_t^{cmd})
\]

atau ekuivalennya wheel commands

\[
u_L=v^{cmd}-\frac{L}{2}\omega^{cmd},\qquad
u_R=v^{cmd}+\frac{L}{2}\omega^{cmd}.
\]

`theta` dapat dipakai sebagai sudut/arah kemudi konseptual, tetapi Duckiebot
tidak memiliki steering wheel seperti mobil Ackermann. Untuk simulator ini,
\((v,\omega)\) atau \((u_L,u_R)\) lebih tepat.

Pilihan action:

- kontinu: \(v^{cmd}\in[0,v_{max}]\),
  \(\omega^{cmd}\in[-\omega_{max},\omega_{max}]\);
- diskrit untuk Q-learning: fast-left, fast-straight, fast-right, slow-left,
  slow-straight, slow-right, brake.

Pada fase lane-following, brake dimask karena dapat menghasilkan policy diam
yang terlihat aman tetapi tidak menyelesaikan tugas. Brake diaktifkan kembali
pada fase stop-sign dan pedestrian.

### 2.3 Transition

\[
s_{t+1}\sim P(s_{t+1}\mid s_t,a_t).
\]

Transition mencakup dinamika differential drive, delay motor, collision,
pergerakan pedestrian, dan randomisasi simulator. Jika satu keputusan ditahan
selama beberapa physics-step, action-repeat tersebut harus dicatat sebagai
bagian definisi transition.

### 2.4 Reward

Reward per keputusan yang disarankan adalah

\[
r_t = w_p v_t\cos\phi_t
-w_d d_t^2-w_\phi\phi_t^2-w_\omega\omega_t^2-c_{step}
+r_t^{event}.
\]

Event reward:

- off-road: penalti besar;
- collision pedestrian: penalti terbesar;
- collision objek lain: penalti besar dan label terpisah;
- stop violation: penalti;
- full stop valid: bonus one-shot;
- goal: bonus;
- timeout tanpa pelanggaran pada lane-following: diperlakukan sebagai sukses,
  bukan kegagalan terminal.

Penalti off-road harus lebih buruk daripada akumulasi reward hidup agar agent
tidak belajar "bunuh diri cepat" untuk menghindari geometry penalty.

### 2.5 Terminal dan truncation

Terminal nyata:

- off-road;
- collision duckie;
- collision objek lain;
- goal tercapai.

Timeout adalah truncation. Dalam TD target, timeout masih melakukan bootstrap,
sedangkan terminal nyata tidak:

\[
y_t=\begin{cases}
r_t,&\text{terminal nyata},\\
r_t+\gamma\max_a Q(s_{t+1},a),&\text{selain itu}.
\end{cases}
\]

## 3. Formulasi POMDP

Untuk visuomotor policy, agent tidak memperoleh state simulator secara
langsung. Masalahnya menjadi

\[
\mathcal{P}=(\mathcal{S},\mathcal{A},P,R,\Omega,O,\gamma).
\]

Latent state \(s_t\) tetap state MDP di atas. Observation dapat berupa

\[
o_t=(I_t,v_t^{odom},\omega_t^{imu},a_{t-1}),
\]

dengan \(I_t\) citra kamera depan. Jika targetnya benar-benar end-to-end
visuomotor, input minimal adalah beberapa frame kamera dan aksi sebelumnya.
Odometry/IMU dapat ditambahkan sebagai proprioception.

### 3.1 Mengapa ini partially observable

- Kamera satu frame tidak memberikan velocity secara pasti.
- Duckie dapat tertutup objek dan niat menyeberangnya tidak terlihat langsung.
- Stop sign yang sudah dilewati dapat keluar dari frame, tetapi flag kepatuhan
  masih harus diingat.
- Delay motor berarti efek aksi lama masih memengaruhi gerak sekarang.
- Persimpangan yang tampak mirip dapat membutuhkan route intent berbeda.

Policy optimal bergantung pada belief

\[
b_t(s)=P(s_t=s\mid o_{0:t},a_{0:t-1}).
\]

Secara praktis belief tidak perlu dihitung eksplisit. Gunakan CNN/ViT untuk
encoder visual dan GRU/LSTM/temporal transformer untuk membentuk hidden state:

\[
z_t=f_{vision}(I_t),\qquad
h_t=f_{memory}(h_{t-1},z_t,a_{t-1}),\qquad
a_t\sim\pi(a\mid h_t).
\]

Alternatif modular adalah perception network yang mengestimasi
\((\hat d,\hat\phi,\hat\kappa,\hat d^{stop},\hat s^{ped})\), tracker temporal,
kemudian policy di atas estimasi state tersebut.

## 4. Hubungan eksperimen MDP ke visuomotor POMDP

Privileged MDP policy berguna sebagai teacher dan oracle:

1. selesaikan kontrol dari state simulator;
2. kumpulkan pasangan gambar, state privileged, dan action;
3. latih encoder visual untuk memprediksi state atau meniru action teacher;
4. fine-tune policy dengan RL menggunakan observation kamera;
5. evaluasi pada seed, spawn, pencahayaan, dan tekstur yang tidak terlihat saat
   training.

Dengan cara ini, kegagalan kontrol dipisahkan dari kegagalan perception.

## 5. Tahapan eksperimen yang disarankan

### Tahap A — lane following

- State MDP: \((d,\phi,v,\kappa_{lookahead})\).
- Action: enam aksi bergerak; brake dimask.
- Sukses: timeout tanpa off-road.
- Metrik: timeout rate, off-road rate, mean/p95 \(|d|\), progress, tile
  transitions, moving ratio.

### Tahap B — stop sign

- Tambah \(d^{stop}\), orientasi/relevansi stop, dan \(\sigma^{stop}\).
- Aktifkan brake.
- Metrik: stop compliance, false-stop rate, waktu berhenti, progress sesudah
  berhenti.

### Tahap C — pedestrian

- Tambah posisi dan velocity relatif, koridor konflik, dan mode crossing.
- Randomisasi waktu serta arah menyeberang.
- Metrik: collision rate, minimum distance, unnecessary-stop rate, dan waktu
  melanjutkan perjalanan setelah aman.

### Tahap D — POMDP visuomotor penuh

- Ganti privileged state dengan history kamera + proprioception.
- Gunakan recurrent policy/state estimator.
- Bandingkan oracle-state MDP, single-frame policy, dan recurrent POMDP policy.

## 6. Kriteria lulus lane-following saat ini

- greedy timeout rate minimal 90% pada held-out seeds;
- off-road rate maksimal 10%;
- mean physics steps minimal 1.350 dari 1.500;
- mean \(|d|\) maksimal sekitar 0,08--0,10 m;
- moving ratio mendekati 1 dan brake ratio 0;
- evaluasi dilakukan tanpa teacher dan tanpa epsilon exploration.
