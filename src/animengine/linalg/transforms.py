
import numpy as np
from numpy.typing import ArrayLike


def get_translation_matrix(tx: float, ty: float, tz: float, dtype: type[np.floating] = np.float64):
    return np.array([
        [1.0, 0.0, 0.0,  tx],
        [0.0, 1.0, 0.0,  ty],
        [0.0, 0.0, 1.0,  tz],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=dtype)

def get_scaling_matrix(sx: float, sy: float, sz: float, dtype: type[np.floating] = np.float64):
    return np.array([
        [ sx, 0.0, 0.0, 0.0],
        [0.0,  sy, 0.0, 0.0],
        [0.0, 0.0,  sz, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=dtype)

def get_rotation_matrix_x(angle_deg: float, dtype: type[np.floating] = np.float64) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    cos = np.cos(angle_rad)
    sin = np.sin(angle_rad)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cos,-sin, 0.0],
        [0.0, sin, cos, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=dtype)

def get_rotation_matrix_y(angle_deg: float, dtype: type[np.floating] = np.float64) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    cos = np.cos(angle_rad)
    sin = np.sin(angle_rad)
    return np.array([
        [ cos, 0.0, sin, 0.0],
        [ 0.0, 1.0, 0.0, 0.0],
        [-sin, 0.0, cos, 0.0],
        [ 0.0, 0.0, 0.0, 1.0],
    ], dtype=dtype)

def get_rotation_matrix_z(angle_deg: float, dtype: type[np.floating] = np.float64) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    cos = np.cos(angle_rad)
    sin = np.sin(angle_rad)
    return np.array([
        [ cos,-sin, 0.0, 0.0],
        [ sin, cos, 0.0, 0.0],
        [ 0.0, 0.0, 1.0, 0.0],
        [ 0.0, 0.0, 0.0, 1.0],
    ], dtype=dtype)

def get_rotation_matrix(
    angle_deg: float, axis: ArrayLike, dtype: type[np.floating] = np.float64
) -> np.ndarray:
    """Rodrigues rotation about an arbitrary axis as a 4x4 homogeneous matrix."""
    angle_rad = np.deg2rad(angle_deg)
    cos = np.cos(angle_rad)
    sin = np.sin(angle_rad)
    axis = np.asarray(axis, dtype=np.float64)
    x, y, z = axis / np.linalg.norm(axis)
    return np.array([
        [cos + x*x*(1-cos),   x*y*(1-cos) - z*sin, x*z*(1-cos) + y*sin, 0.0],
        [y*x*(1-cos) + z*sin, cos + y*y*(1-cos),   y*z*(1-cos) - x*sin, 0.0],
        [z*x*(1-cos) - y*sin, z*y*(1-cos) + x*sin, cos + z*z*(1-cos),   0.0],
        [0.0,                 0.0,                 0.0,                 1.0],
    ], dtype=dtype)