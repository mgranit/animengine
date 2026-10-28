from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Acceleration = Callable[[Array], Array]
Integrator = Callable[[Array, Array, Array, float, Acceleration], tuple[Array, Array, Array]]


def forward_euler(
    positions: Array, velocities: Array, accelerations: Array, dt: float,
    acceleration: Acceleration,
) -> tuple[Array, Array, Array]:
    """Explicit Euler, using acceleration at the beginning of the step."""
    new_positions = positions + velocities * dt
    new_velocities = velocities + accelerations * dt
    return new_positions, new_velocities, acceleration(new_positions)


def velocity_verlet(
    positions: Array, velocities: Array, accelerations: Array, dt: float,
    acceleration: Acceleration,
) -> tuple[Array, Array, Array]:
    """Velocity Verlet for acceleration that depends on position, not velocity."""
    new_positions = positions + velocities * dt + 0.5 * accelerations * dt**2
    new_accelerations = acceleration(new_positions)
    new_velocities = velocities + 0.5 * (accelerations + new_accelerations) * dt
    return new_positions, new_velocities, new_accelerations
