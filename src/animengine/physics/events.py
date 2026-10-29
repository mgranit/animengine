from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class WallCollision:
    """A particle hitting one wall, at an absolute simulation time in seconds."""

    time: float
    particle: int
    position: tuple[float, float]
    axis: Literal[0, 1]
    side: Literal["lower", "upper"]
    speed: float
