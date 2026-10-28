from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .color import to_rgba
from .texture import Texture

DEFAULT_POINT_SIZE = 1.0
DEFAULT_LINE_WIDTH = 1.0

# Keep in sync with `cutoff` in shaders/glow.frag.
_GLOW_CUTOFF = 1.0 / 255.0


@dataclass
class Material:
    """How a mesh is shaded. `point_size` is in pixels of the render target.

    `line_width` above 1 is driver dependent in core-profile OpenGL.

    `blend="additive"` adds the fragment color (scaled by alpha) to what is already drawn, so
    overlapping surfaces brighten each other, and the result does not depend on draw order.

    The `glow` kind draws every mesh vertex as a soft round sprite in the node's XY plane. Its
    brightness is 1 within `glow_core` of the vertex, then falls off as
    `1 / (1 + (x / glow_half_distance) ** glow_power)` where x is the distance beyond the core,
    so `glow_half_distance` is where it has fallen to half. All three are in local units and are
    not affected by resolution. The sprite is cut off where the falloff drops below 1/255.
    """

    kind: Literal["flat", "vertex_color", "textured", "points", "glow"] = "flat"
    color: np.ndarray = field(default_factory=lambda: to_rgba("#ffffff"))
    texture: Texture | None = None
    point_size: float = DEFAULT_POINT_SIZE
    line_width: float = DEFAULT_LINE_WIDTH
    blend: Literal["alpha", "additive"] = "alpha"
    glow_core: float = 0.0
    glow_half_distance: float = 0.01
    glow_power: float = 2.0

    def __post_init__(self) -> None:
        self.color = to_rgba(self.color)

    def glow_extent(self) -> float:
        """Distance from a vertex at which the glow is cut off."""
        if self.glow_half_distance <= 0.0 or self.glow_power <= 0.0:
            raise ValueError("glow_half_distance and glow_power must be positive")
        return self.glow_core + self.glow_half_distance * (_GLOW_CUTOFF**-1 - 1.0) ** (
            1.0 / self.glow_power
        )
