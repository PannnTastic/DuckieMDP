from pathlib import Path

import numpy as np

from src.decorations import (
    TexturedQuadDecoration,
    attach_kfupm_small_loop_decorations,
    kfupm_small_loop_specs,
)


ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"


def test_kfupm_layout_contains_logo_billboard_and_two_poles():
    specs = kfupm_small_loop_specs(ASSET_DIR)
    assert [spec.name for spec in specs] == [
        "kfupm_center_logo",
        "jisr3_billboard",
        "jisr3_billboard_pole_0",
        "jisr3_billboard_pole_1",
    ]
    assert all(spec.texture_path.is_file() for spec in specs)


def test_logo_is_horizontal_and_centered_on_middle_tile():
    logo = kfupm_small_loop_specs(ASSET_DIR)[0]
    vertices = np.asarray(logo.vertices)
    expected_center = 1.5 * 0.585
    assert np.allclose(vertices[:, 1], 0.008)
    assert np.allclose(vertices[:, [0, 2]].mean(axis=0), expected_center)
    assert vertices[:, 0].min() >= 0.585
    assert vertices[:, 0].max() <= 1.170
    assert vertices[:, 2].min() >= 0.585
    assert vertices[:, 2].max() <= 1.170


def test_billboard_is_left_of_spawn_lane_and_decorations_are_non_collidable():
    billboard = kfupm_small_loop_specs(ASSET_DIR)[1]
    vertices = np.asarray(billboard.vertices)
    # Outer boundary lane pertama berada sekitar x=0.0585 m. Billboard boleh
    # sedikit masuk ke verge map, tetapi tidak ke area lajur ego.
    assert np.all(vertices[:, 0] < 0.0585)

    decoration = TexturedQuadDecoration(billboard)
    assert decoration.check_collision(None, None) is False
    assert decoration.proximity(None, None) == 0.0
    assert decoration.scale == 0.0


def test_attach_is_idempotent_and_does_not_touch_collision_arrays():
    class Simulator:
        road_tile_size = 0.585
        objects = []
        collidable_centers = np.array([[9.0, 9.0]])
        collidable_corners = np.array([[[9.0, 9.0]]])

    class Env:
        unwrapped = Simulator()

    centers_before = Env.unwrapped.collidable_centers.copy()
    corners_before = Env.unwrapped.collidable_corners.copy()
    assert len(attach_kfupm_small_loop_decorations(Env(), ASSET_DIR)) == 4
    assert len(attach_kfupm_small_loop_decorations(Env(), ASSET_DIR)) == 0
    assert np.array_equal(Env.unwrapped.collidable_centers, centers_before)
    assert np.array_equal(Env.unwrapped.collidable_corners, corners_before)
