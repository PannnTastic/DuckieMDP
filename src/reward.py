"""Reward function R(s, a, s') untuk MDP Duckietown.

Reward dense per decision:

    r_t = alpha_p*v_t*cos(phi_t)
          - alpha_d*d_t^2 - alpha_phi*phi_t^2 - c_step + r_event

Progress mendorong gerak maju sejajar lajur. Quadratic penalties menjaga posisi
dan heading. Event reward menangani keselamatan, kepatuhan stop, dan goal.
Semua koefisien berasal dari YAML agar eksperimen reproducible.
"""

from dataclasses import asdict, dataclass
from math import cos
from typing import Dict, Optional, Tuple

from .state import DuckThreat, RawState


@dataclass(frozen=True)
class RewardConfig:
    alpha_progress: float = 1.0
    alpha_lateral: float = 10.0
    alpha_heading: float = 2.0
    step_cost: float = 0.01
    collision_duck: float = -100.0
    other_collision: float = -50.0
    offroad: float = -50.0
    stop_violation: float = -20.0
    full_stop: float = 10.0
    # Default nol menjaga eksperimen lama reproducible. Config teacher-free
    # mengaktifkannya agar yielding dapat dipelajari tanpa demonstrasi.
    duck_yield: float = 0.0
    duck_unsafe: float = 0.0
    duck_yield_speed: float = 0.04
    unnecessary_stop: float = 0.0
    idle_speed: float = 0.04
    stop_exemption_distance: float = 0.45
    # Penalti action-conditioned khusus ruas lurus. Nilai nol menjaga seluruh
    # baseline lama identik; eksperimen warm-start mengaktifkannya dari YAML.
    straight_steer_penalty: float = 0.0
    straight_curvature_threshold: float = 0.05
    max_steer_command: float = 1.5
    goal: float = 50.0


@dataclass
class EventFlags:
    """Event diskrit yang menambahkan bonus atau penalti ke reward dense."""
    collision_duck: bool = False
    other_collision: bool = False
    offroad: bool = False
    timeout: bool = False
    stop_violation: bool = False
    full_stop: bool = False
    passed_stop: bool = False
    goal: bool = False


@dataclass(frozen=True)
class RewardBreakdown:
    progress: float
    lateral: float
    heading: float
    time: float
    pedestrian: float
    stagnation: float
    steering: float
    events: float
    total: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


class StopTracker:
    """Menjaga sigma_stop agar bonus full-stop hanya diberikan satu kali."""
    def __init__(
        self,
        zone: float = 0.45,
        speed: float = 0.02,
        pass_distance: float = 0.55,
        hold_steps: int = 1,
    ) -> None:
        self.zone = zone
        self.speed = speed
        self.pass_distance = max(zone, pass_distance)
        self.hold_steps_required = max(1, int(hold_steps))
        self.hold_steps = 0
        self.sigma_stop = False

    @property
    def hold_progress(self) -> float:
        """Progres dwell ternormalisasi; fitur ini membuat proses tetap Markov."""
        if self.sigma_stop:
            return 1.0
        return min(1.0, self.hold_steps / self.hold_steps_required)

    def reset(self) -> None:
        self.sigma_stop = False
        self.hold_steps = 0

    def update(
        self,
        previous: RawState,
        current: RawState,
        previous_stop_id: Optional[int] = None,
        current_stop_id: Optional[int] = None,
    ) -> Tuple[bool, EventFlags]:
        events = EventFlags()
        ids_available = previous_stop_id is not None or current_stop_id is not None
        if ids_available:
            stop_changed = previous_stop_id is not None and previous_stop_id != current_stop_id
            passed = stop_changed and previous.d_stop is not None
            passed = passed and previous.d_stop <= self.pass_distance
        else:
            passed = previous.d_stop is not None and previous.d_stop <= self.pass_distance
            passed = passed and (
                current.d_stop is None or current.d_stop > previous.d_stop + 0.5
            )

        if passed:
            events.passed_stop = True
            events.stop_violation = not self.sigma_stop
            self.sigma_stop = False
            self.hold_steps = 0
            return self.sigma_stop, events

        if ids_available and stop_changed:
            # Kandidat berubah tanpa melewati sign lama (misalnya sign hilang
            # dari filter); memori stop lama tidak boleh dibawa ke sign baru.
            self.sigma_stop = False
            self.hold_steps = 0

        near = current.d_stop is not None and current.d_stop <= self.zone
        slow = current.v < self.speed
        if not self.sigma_stop:
            if near and slow:
                self.hold_steps += 1
                if self.hold_steps >= self.hold_steps_required:
                    self.sigma_stop = True
                    self.hold_steps = self.hold_steps_required
                    events.full_stop = True
            else:
                # Dwell harus berurutan, bukan akumulasi beberapa brake singkat.
                self.hold_steps = 0
        return self.sigma_stop, events


def compute_reward(
    state: RawState,
    events: EventFlags,
    cfg: RewardConfig = RewardConfig(),
    action_omega: float = 0.0,
    curvature: Optional[float] = None,
) -> RewardBreakdown:
    """Menghitung komponen reward terpisah agar mudah diaudit di log."""
    # Kemajuan sepanjang tangent lajur; cos(phi) mengecil saat ego tidak sejajar.
    progress = cfg.alpha_progress * state.v * cos(state.phi)
    # Penalti kuadrat kecil dekat centerline dan cepat membesar saat menyimpang.
    lateral = -cfg.alpha_lateral * state.d ** 2
    heading = -cfg.alpha_heading * state.phi ** 2
    # Biaya waktu mencegah return gratis dari policy yang tidak bergerak.
    time = -cfg.step_cost
    # Collision adalah sinyal yang sangat jarang. Shaping ini membuat action
    # brake dapat ditemukan secara model-free: pelan saat crossing diberi
    # kredit, sedangkan tetap melaju menerima penalti langsung.
    crossing = state.duck in {
        DuckThreat.CROSSING_FAR,
        DuckThreat.CROSSING_NEAR,
    }
    pedestrian = 0.0
    if crossing:
        pedestrian = cfg.duck_yield if state.v < cfg.duck_yield_speed else cfg.duck_unsafe
    must_stop = (
        state.d_stop is not None
        and state.d_stop <= cfg.stop_exemption_distance
        and not state.sigma_stop
    )
    # Hilangkan reward hacking berupa brake di state normal. Diam tetap sah
    # selama crossing atau ketika kewajiban stop belum dipenuhi.
    unnecessary_idle = state.v < cfg.idle_speed and not crossing and not must_stop
    stagnation = cfg.unnecessary_stop if unnecessary_idle else 0.0
    # phi menghukum HASIL ketika heading sudah melenceng. Suku ini menghukum
    # PENYEBAB lebih awal: perintah steer besar pada geometri jalan lurus.
    # Curvature berasal dari s_t, sehingga reward tetap berbentuk R(s, a, s').
    steering = 0.0
    if (
        curvature is not None
        and abs(curvature) <= cfg.straight_curvature_threshold
        and cfg.straight_steer_penalty != 0.0
    ):
        steer_scale = max(abs(cfg.max_steer_command), 1e-9)
        normalized_steer = min(1.0, abs(action_omega) / steer_scale)
        steering = -abs(cfg.straight_steer_penalty) * normalized_steer ** 2
    # Event keselamatan/kepatuhan berskala lebih besar daripada shaping dense.
    event = (
        cfg.collision_duck * events.collision_duck
        + cfg.other_collision * events.other_collision
        + cfg.offroad * events.offroad
        + cfg.stop_violation * events.stop_violation
        + cfg.full_stop * events.full_stop
        + cfg.goal * events.goal
    )
    total = progress + lateral + heading + time + pedestrian + stagnation + steering + event
    return RewardBreakdown(
        progress=progress,
        lateral=lateral,
        heading=heading,
        time=time,
        pedestrian=pedestrian,
        stagnation=stagnation,
        steering=steering,
        events=event,
        total=total,
    )
