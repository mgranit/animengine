import numpy as np


def linear(t: np.ndarray | float) -> np.ndarray | float:
    return t


def smoothstep(t: np.ndarray | float) -> np.ndarray | float:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ease_in_out_cubic(t: np.ndarray | float) -> np.ndarray | float:
    t = np.clip(t, 0.0, 1.0)
    return np.where(t < 0.5, 4.0 * t**3, 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0)
