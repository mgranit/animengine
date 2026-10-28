from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .boundaries import RectangularBoundary
from .integrators import Integrator, velocity_verlet

ParticleArray = NDArray[np.float64]
Acceleration = Callable[[ParticleArray], ParticleArray]


def _particle_array(name: str, values, shape: tuple[int, ...] | None = None) -> ParticleArray:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 2 or array.shape[1] != 2 or (shape is not None and array.shape != shape):
        raise ValueError(f"{name} must have shape {shape or '(N, 2)'}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


class ParticleState:
    """Positions, velocities and accelerations in world units, as float64 (N, 2) arrays."""

    def __init__(self, positions, velocities, accelerations=None) -> None:
        self.positions = _particle_array("positions", positions)
        self.velocities = _particle_array("velocities", velocities, self.positions.shape)
        if accelerations is None:
            self.accelerations = np.zeros_like(self.positions)
        else:
            self.accelerations = _particle_array(
                "accelerations", accelerations, self.positions.shape
            )


def zero_acceleration(positions: ParticleArray) -> ParticleArray:
    return np.zeros_like(positions)


class ParticleSimulation:
    """Advance a particle state with position-dependent acceleration and optional walls.

    The acceleration callback receives positions and returns an (N, 2) array. It may be
    evaluated outside the walls when a numerical step crosses one; see RectangularBoundary.
    """

    def __init__(
        self,
        state: ParticleState,
        *,
        acceleration: Acceleration = zero_acceleration,
        integrator: Integrator = velocity_verlet,
        boundary: RectangularBoundary | None = None,
    ) -> None:
        if not callable(acceleration) or not callable(integrator):
            raise TypeError("acceleration and integrator must be callable")
        self.state = state
        self.acceleration = acceleration
        self.integrator = integrator
        self.boundary = boundary
        if boundary is not None:
            boundary.validate_positions(state.positions)
        self.state.accelerations[:] = self._acceleration_at(state.positions)

    def _acceleration_at(self, positions: ParticleArray) -> ParticleArray:
        return _particle_array("acceleration", self.acceleration(positions.copy()), positions.shape)

    def step(self, dt: float) -> None:
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and non-negative")
        if dt == 0:
            return

        positions = _particle_array("positions", self.state.positions)
        velocities = _particle_array("velocities", self.state.velocities, positions.shape)
        if self.state.accelerations.shape != positions.shape:
            raise ValueError("state accelerations must match positions")
        if self.boundary is not None:
            self.boundary.validate_positions(positions)

        acceleration = self._acceleration_at(positions)
        new_positions, new_velocities, new_accelerations = self.integrator(
            positions, velocities, acceleration, dt, self._acceleration_at
        )
        new_positions = _particle_array("integrated positions", new_positions, positions.shape)
        new_velocities = _particle_array("integrated velocities", new_velocities, positions.shape)
        new_accelerations = _particle_array(
            "integrated accelerations", new_accelerations, positions.shape
        )
        if self.boundary is not None:
            new_positions, new_velocities = self.boundary.resolve(new_positions, new_velocities)
            new_accelerations = self._acceleration_at(new_positions)

        self.state.positions[:] = new_positions
        self.state.velocities[:] = new_velocities
        self.state.accelerations[:] = new_accelerations
