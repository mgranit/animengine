from collections.abc import Callable
import numpy as np

def forward_euler_step(
    dt: float,
    pos: np.ndarray,
    vel: np.ndarray,
    acc: np.ndarray,
    mass: np.ndarray,
    force_func: Callable[[np.ndarray], np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    if pos.ndim != 2 or vel.ndim != 2 or acc.ndim != 2:
        raise ValueError("pos, vel, and acc must be 2D arrays with shape (N, 2)")
    
    if mass.ndim == 1:
        mass = mass[:, np.newaxis]
    
    pos_new = pos + vel * dt
    vel_new = vel + acc * dt
    acc_new = force_func(pos_new) / mass
    
    return pos_new, vel_new, acc_new

def symplectic_euler_step(
    dt: float,
    pos: np.ndarray,
    vel: np.ndarray,
    acc: np.ndarray,
    mass: np.ndarray,
    force_func: Callable[[np.ndarray], np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    if pos.ndim != 2 or vel.ndim != 2 or acc.ndim != 2:
        raise ValueError("pos, vel, and acc must be 2D arrays with shape (N, 2)")
    
    if mass.ndim == 1:
        mass = mass[:, np.newaxis]
    
    vel_new = vel + acc * dt
    pos_new = pos + vel_new * dt
    acc_new = force_func(pos_new) / mass
    
    return pos_new, vel_new, acc_new

def velocity_verlet_step(
    dt: float,
    pos: np.ndarray,
    vel: np.ndarray,
    acc: np.ndarray,
    mass: np.ndarray,
    force_func: Callable[[np.ndarray], np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    if pos.ndim != 2 or vel.ndim != 2 or acc.ndim != 2:
        raise ValueError("pos, vel, and acc must be 2D arrays with shape (N, 2)")
    
    if mass.ndim == 1:
        mass = mass[:, np.newaxis]
    
    pos_new = pos + vel * dt + 0.5 * acc * dt**2
    acc_new = force_func(pos_new) / mass
    vel_new = vel + 0.5 * (acc + acc_new) * dt
    
    return pos_new, vel_new, acc_new
