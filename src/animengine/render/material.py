from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .color import to_rgba

DEFAULT_POINT_SIZE = 1.0
DEFAULT_LINE_WIDTH = 1.0


@dataclass
class Material:
    """How a mesh is shaded. `point_size` is in pixels of the render target.

    `line_width` above 1 is driver dependent in core-profile OpenGL.
    """

    kind: Literal["flat", "vertex_color", "points"] = "flat"
    color: np.ndarray = field(default_factory=lambda: to_rgba("#ffffff"))
    point_size: float = DEFAULT_POINT_SIZE
    line_width: float = DEFAULT_LINE_WIDTH

    def __post_init__(self) -> None:
        self.color = to_rgba(self.color)
