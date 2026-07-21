from types import SimpleNamespace

import numpy as np

from src.duck_controller import DuckController, DuckControllerConfig


class _Env:
    cur_pos = np.array([0.0, 0.0, 0.0])
    cur_angle = 0.0

    @staticmethod
    def closest_curve_point(_pos, _angle):
        return np.zeros(3), np.array([1.0, 0.0, 0.0])


def test_crossing_limit_prevents_immediate_second_crossing():
    duck = SimpleNamespace(
        pedestrian_active=False,
        pedestrian_wait_time=float("inf"),
        heading=np.array([1.0, 0.0, 0.0]),
        start=np.array([0.05, 0.0, 0.0]),
        walk_distance=0.90,
        vel=0.02,
        time=0.0,
    )
    controller = DuckController.__new__(DuckController)
    controller.env = _Env()
    controller.cfg = DuckControllerConfig(
        p_cross=1.0,
        trigger_min_ego_distance=0.40,
        trigger_max_ego_distance=0.60,
        max_crossings_per_episode=1,
    )
    controller.rng = np.random.RandomState(0)
    controller.ducks = [duck]
    controller.crossings_started = [0]
    controller.crossing_armed = [True]

    controller.before_step()
    assert duck.pedestrian_active
    assert controller.crossings_started == [1]

    # Meniru DuckieObj.finish_walk(). Ego masih di lokasi yang sama, tetapi
    # limit satu crossing memblokir crossing balik.
    duck.pedestrian_active = False
    controller.before_step()
    assert not duck.pedestrian_active
    assert controller.crossings_started == [1]


def test_repeat_crossing_rearms_only_after_ego_departs():
    env = _Env()
    env.cur_pos = np.array([0.0, 0.0, 0.0])
    duck = SimpleNamespace(
        pedestrian_active=False,
        pedestrian_wait_time=float("inf"),
        heading=np.array([1.0, 0.0, 0.0]),
        start=np.array([0.05, 0.0, 0.0]),
        walk_distance=0.90,
        vel=0.02,
        time=0.0,
    )
    controller = DuckController.__new__(DuckController)
    controller.env = env
    controller.cfg = DuckControllerConfig(
        p_cross=1.0,
        trigger_min_ego_distance=0.40,
        trigger_max_ego_distance=0.60,
        max_crossings_per_episode=0,
        repeat_rearm_distance=1.0,
    )
    controller.rng = np.random.RandomState(0)
    controller.ducks = [duck]
    controller.crossings_started = [0]
    controller.crossing_armed = [True]

    controller.before_step()
    assert duck.pedestrian_active
    assert controller.crossings_started == [1]
    assert not controller.crossing_armed[0]

    # Meniru finish_walk pada sisi seberang; ego masih dekat, jadi tidak balik.
    duck.pedestrian_active = False
    duck.start = np.array([0.95, 0.0, 0.0])
    duck.vel = -0.02
    controller.before_step()
    assert not duck.pedestrian_active
    assert controller.crossings_started == [1]

    # Ego meninggalkan crossing untuk meng-arm ulang.
    env.cur_pos = np.array([-1.0, 0.0, 0.0])
    controller.before_step()
    assert controller.crossing_armed[0]
    assert controller.crossings_started == [1]

    # Pada pendekatan lap berikutnya Duckie boleh menyeberang balik.
    env.cur_pos = np.array([0.0, 0.0, 0.0])
    controller.before_step()
    assert duck.pedestrian_active
    assert controller.crossings_started == [2]


def test_spawn_on_proximity_hides_duck_until_trigger_window():
    env = _Env()
    env.cur_pos = np.array([-1.0, 0.0, 0.0])
    duck = SimpleNamespace(
        visible=False,
        pedestrian_active=False,
        pedestrian_wait_time=float("inf"),
        heading=np.array([1.0, 0.0, 0.0]),
        start=np.array([0.05, 0.0, 0.0]),
        walk_distance=0.90,
        vel=0.02,
        time=0.0,
    )
    controller = DuckController.__new__(DuckController)
    controller.env = env
    controller.cfg = DuckControllerConfig(
        p_cross=1.0,
        trigger_min_ego_distance=0.40,
        trigger_max_ego_distance=0.60,
        spawn_on_ego_proximity=True,
    )
    controller.rng = np.random.RandomState(0)
    controller.ducks = [duck]
    controller.crossings_started = [0]
    controller.crossing_armed = [True]

    controller.before_step()
    assert not duck.visible
    assert not duck.pedestrian_active
    assert controller.crossings_started == [0]

    env.cur_pos = np.array([0.0, 0.0, 0.0])
    controller.before_step()
    assert duck.visible
    assert not duck.pedestrian_active
    assert controller.crossings_started == [0]

    # Crossing baru dimulai pada decision berikutnya sehingga policy sempat
    # menerima satu observation dengan duck_present=True.
    controller.before_step()
    assert duck.pedestrian_active
    assert controller.crossings_started == [1]
