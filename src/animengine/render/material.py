from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .color import to_rgba
from .texture import Texture


@dataclass
class Material:
    """How a mesh is shaded. `point_size` is in pixels of the render target.

    `line_width` above 1 is driver dependent in core-profile OpenGL.
    """

    kind: Literal["flat", "vertex_color", "textured", "points"] = "flat"
    color: np.ndarray = field(default_factory=lambda: to_rgba("#ffffff"))
    texture: Texture | None = None
    point_size: float = 3.0
    line_width: float = 1.0

    def __post_init__(self) -> None:
        self.color = to_rgba(self.color)
