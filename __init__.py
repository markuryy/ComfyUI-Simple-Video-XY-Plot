"""Top-level package for video_xy_plot."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """Markury"""
__email__ = "comfy@markury.dev"
__version__ = "0.0.1"

from .src.video_xy_plot.nodes import NODE_CLASS_MAPPINGS
from .src.video_xy_plot.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
