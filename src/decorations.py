"""Dekorasi visual non-collidable untuk render small_loop.

Modul ini sengaja tidak menjadi bagian dari MDP. Objek hanya ditambahkan ke
daftar render simulator dan tidak dimasukkan ke collision arrays, state,
reward, atau observation policy. Dengan demikian checkpoint yang sama tetap
menjalankan policy yang sama; yang berubah hanya tampilan video.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class DecorationSpec:
    """Satu textured quad dalam koordinat world Duckietown."""

    name: str
    texture_path: Path
    vertices: Tuple[Tuple[float, float, float], ...]
    position: Tuple[float, float, float]


def _horizontal_quad(
    center_x: float,
    center_z: float,
    width: float,
    depth: float,
    height: float,
) -> Tuple[Tuple[float, float, float], ...]:
    half_width = 0.5 * width
    half_depth = 0.5 * depth
    return (
        (center_x - half_width, height, center_z - half_depth),
        (center_x + half_width, height, center_z - half_depth),
        (center_x + half_width, height, center_z + half_depth),
        (center_x - half_width, height, center_z + half_depth),
    )


def _vertical_quad(
    center_x: float,
    center_z: float,
    width: float,
    bottom: float,
    top: float,
    tangent_x: float,
    tangent_z: float,
) -> Tuple[Tuple[float, float, float], ...]:
    """Vertical quad dengan sumbu lebar ``tangent`` pada bidang x-z."""
    tangent = np.asarray([tangent_x, tangent_z], dtype=float)
    tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
    half_width = 0.5 * width
    offset_x, offset_z = half_width * tangent
    return (
        (center_x - offset_x, bottom, center_z - offset_z),
        (center_x + offset_x, bottom, center_z + offset_z),
        (center_x + offset_x, top, center_z + offset_z),
        (center_x - offset_x, top, center_z - offset_z),
    )


def kfupm_small_loop_specs(
    asset_dir: Path,
    tile_size: float = 0.585,
) -> Tuple[DecorationSpec, ...]:
    """Layout KFUPM untuk small_loop 3x3.

    Pusat tile asphalt adalah ``(1.5*tile_size, 1.5*tile_size)``. Billboard
    berada sedikit di luar batas barat map, sejajar ruas spawn tile ``(0, 1)``.
    Pada route clockwise, ego spawn menghadap +z sehingga billboard berada di
    sisi kirinya.
    """
    asset_dir = Path(asset_dir).resolve()
    center = 1.5 * float(tile_size)
    # Billboard dimiringkan menghadap nominal spawn di ruas barat. Jika bidang
    # dibuat sejajar ruas, kamera agen hanya melihat sisi tipisnya.
    billboard_x = -0.10
    billboard_z = 1.58
    # Tanda negatif menentukan sisi texture yang terbaca dari kamera ego.
    billboard_tangent = (-0.56, -0.8285)
    billboard_width = 0.36
    billboard_bottom = 0.04
    billboard_top = 0.22
    pole_width = 0.038
    tangent = np.asarray(billboard_tangent, dtype=float)
    tangent /= np.linalg.norm(tangent)
    pole_centers = tuple(
        (
            billboard_x + offset * float(tangent[0]),
            billboard_z + offset * float(tangent[1]),
        )
        for offset in (-0.40 * billboard_width, 0.40 * billboard_width)
    )

    specs = [
        DecorationSpec(
            name="kfupm_center_logo",
            texture_path=asset_dir / "kfupm_logo.png",
            vertices=_horizontal_quad(center, center, 0.48, 0.48, 0.008),
            position=(center, 0.008, center),
        ),
        DecorationSpec(
            name="jisr3_billboard",
            texture_path=asset_dir / "billboard_jisr3.png",
            vertices=_vertical_quad(
                billboard_x,
                billboard_z,
                billboard_width,
                billboard_bottom,
                billboard_top,
                *billboard_tangent,
            ),
            position=(billboard_x, 0.5 * (billboard_bottom + billboard_top), billboard_z),
        ),
    ]
    for index, (pole_x, pole_z) in enumerate(pole_centers):
        specs.append(
            DecorationSpec(
                name=f"jisr3_billboard_pole_{index}",
                texture_path=asset_dir / "pole.png",
                vertices=_vertical_quad(
                    pole_x,
                    pole_z,
                    pole_width,
                    0.0,
                    billboard_bottom + 0.025,
                    *billboard_tangent,
                ),
                position=(pole_x, 0.5 * billboard_bottom, pole_z),
            )
        )

    missing = [str(spec.texture_path) for spec in specs if not spec.texture_path.is_file()]
    if missing:
        raise FileNotFoundError("Missing decoration assets: " + ", ".join(missing))
    return tuple(specs)


class TexturedQuadDecoration:
    """Duckietown-renderable quad tanpa pengaruh collision atau dynamics."""

    visible = True
    static = True
    optional = False
    safety_radius = 0.0

    def __init__(self, spec: DecorationSpec) -> None:
        self.name = spec.name
        self.kind = f"decoration_{spec.name}"
        self.texture_path = str(spec.texture_path)
        self.vertices = np.asarray(spec.vertices, dtype=np.float32)
        self.pos = np.asarray(spec.position, dtype=np.float32)
        # Simulator memakai max_coords saat menyaring random spawn. Nilai nol
        # memastikan dekorasi visual tidak mengubah distribusi spawn.
        self.min_coords = np.zeros(3, dtype=np.float32)
        self.max_coords = np.zeros(3, dtype=np.float32)
        self.scale = 0.0
        self._texture = None

    def _ensure_texture(self):
        if self._texture is None:
            from gym_duckietown.graphics import load_texture

            self._texture = load_texture(self.texture_path)
        return self._texture

    def render(self, draw_bbox: bool, enable_leds: bool, segment: bool = False) -> None:
        del draw_bbox, enable_leds
        # Dekorasi tidak ikut semantic/segmentation render karena bukan bagian
        # observation atau konsep task.
        if not self.visible or segment:
            return

        from pyglet import gl

        texture = self._ensure_texture()
        gl.glPushAttrib(
            gl.GL_ENABLE_BIT
            | gl.GL_COLOR_BUFFER_BIT
            | gl.GL_TEXTURE_BIT
            | gl.GL_DEPTH_BUFFER_BIT
        )
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_ALPHA_TEST)
        gl.glAlphaFunc(gl.GL_GREATER, 0.01)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glColor4f(1.0, 1.0, 1.0, 1.0)
        gl.glBindTexture(texture.target, texture.id)
        gl.glBegin(gl.GL_QUADS)
        for (u, v), vertex in zip(
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            self.vertices,
        ):
            gl.glTexCoord2f(u, v)
            gl.glVertex3f(float(vertex[0]), float(vertex[1]), float(vertex[2]))
        gl.glEnd()
        gl.glPopAttrib()

    def check_collision(self, agent_corners, agent_norm) -> bool:
        del agent_corners, agent_norm
        return False

    def proximity(self, agent_pos, agent_safety_rad) -> float:
        del agent_pos, agent_safety_rad
        return 0.0

    def step(self, delta_time: float) -> None:
        del delta_time


def attach_kfupm_small_loop_decorations(
    env,
    asset_dir: Path,
) -> Sequence[TexturedQuadDecoration]:
    """Tambahkan dekorasi sekali saja dan kembalikan objek yang dibuat."""
    simulator = env.unwrapped
    existing = {
        getattr(obj, "name", "")
        for obj in simulator.objects
        if str(getattr(obj, "kind", "")).startswith("decoration_")
    }
    created: List[TexturedQuadDecoration] = []
    for spec in kfupm_small_loop_specs(asset_dir, simulator.road_tile_size):
        if spec.name in existing:
            continue
        decoration = TexturedQuadDecoration(spec)
        simulator.objects.append(decoration)
        created.append(decoration)
    return tuple(created)
