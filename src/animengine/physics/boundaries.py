import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _vectors(name: str, values) -> Array:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite array with shape (N, 2)")
    return array


class RectangularBoundary:
    """Elastic walls for point centers, with lower and upper corners in world coordinates.

    Folding a completed step is exact for free motion, even across multiple walls. With
    acceleration it is an approximation: the integrator does not resolve the impact time
    or integrate the remaining part of the step after impact.
    """

    def __init__(self, lower, upper) -> None:
        self.lower = np.array(lower, dtype=np.float64, copy=True)
        self.upper = np.array(upper, dtype=np.float64, copy=True)
        if self.lower.shape != (2,) or self.upper.shape != (2,):
            raise ValueError("boundary corners must have shape (2,)")
        if (not np.all(np.isfinite(self.lower)) or not np.all(np.isfinite(self.upper))
                or np.any(self.upper <= self.lower)):
            raise ValueError("boundary corners must be finite and upper must exceed lower")
        with np.errstate(over="ignore"):
            if not np.all(np.isfinite(2.0 * (self.upper - self.lower))):
                raise ValueError("boundary width is too large")

    def validate_positions(self, positions: Array) -> None:
        positions = _vectors("positions", positions)
        if np.any(positions < self.lower) or np.any(positions > self.upper):
            raise ValueError("particle positions must be inside the boundary")

    def resolve(self, positions: Array, velocities: Array) -> tuple[Array, Array]:
        """Fold unconstrained endpoints and reflect velocity, including exact wall hits."""
        positions = _vectors("positions", positions)
        velocities = _vectors("velocities", velocities)
        if velocities.shape != positions.shape:
            raise ValueError("velocities must match positions")
        width = self.upper - self.lower
        with np.errstate(over="ignore", invalid="ignore"):
            phase = np.remainder(positions - self.lower, 2.0 * width)
        if not np.all(np.isfinite(phase)):
            raise ValueError("particle displacement is too large to resolve")
        reflected = phase > width
        result_positions = self.lower + np.where(reflected, 2.0 * width - phase, phase)
        result_positions = np.clip(result_positions, self.lower, self.upper)
        result_velocities = np.where(reflected, -velocities, velocities)
        result_velocities = np.where(phase == 0, np.abs(result_velocities), result_velocities)
        result_velocities = np.where(phase == width, -np.abs(result_velocities), result_velocities)
        return result_positions, result_velocities
