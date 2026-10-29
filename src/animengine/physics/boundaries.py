import numpy as np
from numpy.typing import NDArray

from .events import WallCollision

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

    def collisions(
        self, positions: Array, endpoints: Array, dt: float, start_time: float,
    ) -> list[WallCollision]:
        """Find wall crossings along the straight segment of each unconstrained step."""
        positions = _vectors("positions", positions)
        endpoints = _vectors("endpoints", endpoints)
        if positions.shape != endpoints.shape:
            raise ValueError("endpoints must match positions")
        self.validate_positions(positions)
        if not np.isfinite(dt) or dt <= 0 or not np.isfinite(start_time):
            raise ValueError("dt must be positive and times must be finite")
        displacement = endpoints - positions
        if not np.all(np.isfinite(displacement)):
            raise ValueError("particle displacement is too large to resolve")

        width = self.upper - self.lower
        collisions = []
        for particle, (start, delta) in enumerate(zip(positions, displacement, strict=True)):
            for axis in (0, 1):
                distance = abs(delta[axis])
                if distance == 0:
                    continue
                first_upper = delta[axis] > 0
                first_distance = (
                    self.upper[axis] - start[axis] if first_upper
                    else start[axis] - self.lower[axis]
                )
                if first_distance > distance:
                    continue
                with np.errstate(over="ignore"):
                    crossings = (distance - first_distance) / width[axis]
                    speed = distance / dt
                if not np.isfinite(crossings) or crossings >= 100_000:
                    raise ValueError("too many wall collisions in one step")
                if not np.isfinite(speed):
                    raise ValueError("wall collision speed is too large")
                count = int(np.floor(crossings)) + 1
                for crossing in range(count):
                    fraction = (first_distance + crossing * width[axis]) / distance
                    phase = np.remainder(start + fraction * delta - self.lower, 2 * width)
                    impact = self.lower + np.where(phase > width, 2 * width - phase, phase)
                    is_upper = first_upper == (crossing % 2 == 0)
                    impact[axis] = self.upper[axis] if is_upper else self.lower[axis]
                    collisions.append(WallCollision(
                        time=float(start_time + fraction * dt),
                        particle=particle,
                        position=(float(impact[0]), float(impact[1])),
                        axis=axis,
                        side="upper" if is_upper else "lower",
                        speed=float(speed),
                    ))
        collisions.sort(key=lambda event: (event.time, event.particle, event.axis))
        return collisions
