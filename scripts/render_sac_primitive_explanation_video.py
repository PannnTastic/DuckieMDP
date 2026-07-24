"""Headless entry point for the explanation-derived primitive video."""

import pyglet

pyglet.options["headless"] = True

from src.render_sac_primitive_explanation_video import main


if __name__ == "__main__":
    main()
