import numpy as np

from .frame import Frame


def perspective(fov_deg: float, aspect: float, near: float, far: float,
                dtype=np.float32) -> np.ndarray:
    f = 1.0 / np.tan(np.deg2rad(fov_deg) / 2.0)
    return np.array([
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far)],
        [0.0, 0.0, -1.0, 0.0],
    ], dtype=dtype)


def orthographic(left: float, right: float, bottom: float, top: float,
                 near: float, far: float, dtype=np.float32) -> np.ndarray:
    return np.array([
        [2.0 / (right - left), 0.0, 0.0, -(right + left) / (right - left)],
        [0.0, 2.0 / (top - bottom), 0.0, -(top + bottom) / (top - bottom)],
        [0.0, 0.0, -2.0 / (far - near), -(far + near) / (far - near)],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=dtype)


def look_at(eye, target, up, dtype=np.float32) -> np.ndarray:
    return Frame.look_at(eye, target, up).inverse_matrix(dtype)
