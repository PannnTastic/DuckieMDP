import numpy as np

from src.render_multiview_video import HEIGHT, WIDTH, compose_frame, letterbox, world_to_panel


def test_letterbox_and_composite_dimensions():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    assert letterbox(image, 640, 480).shape == (480, 640, 3)
    trajectory = np.zeros((600, 960, 3), dtype=np.uint8)
    dashboard = np.zeros((600, 960, 3), dtype=np.uint8)
    frame = compose_frame(image, image, image, trajectory, dashboard)
    assert frame.shape == (HEIGHT, WIDTH, 3)


def test_world_to_panel_preserves_map_corners():
    assert world_to_panel(0, 0, 3, 2, 960, 600) == (58, 542)
    assert world_to_panel(3, 2, 3, 2, 960, 600) == (902, 58)
