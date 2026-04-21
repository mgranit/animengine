from typing import Type
import numpy as np

def get_translation_matrix(tx, ty, tz, dtype: Type[np.floating] = np.float64):
    return np.array([
        [1.0, 0.0, 0.0,  tx],
        [0.0, 1.0, 0.0,  ty],
        [0.0, 0.0, 1.0,  tz],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=dtype)

def get_scaling_matrix(sx, sy, sz, dtype: Type[np.floating] = np.float64):
    return np.array([
        [ sx, 0.0, 0.0, 0.0],
        [0.0,  sy, 0.0, 0.0],
        [0.0, 0.0,  sz, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=dtype)

def get_rotation_matrix_x(angle_deg, dtype: Type[np.floating] = np.float64):
    angle_rad = np.deg2rad(angle_deg)
    cos = np.cos(angle_rad)
    sin = np.sin(angle_rad)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cos,-sin, 0.0],
        [0.0, sin, cos, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=dtype)

def get_rotation_matrix_y(angle_deg, dtype: Type[np.floating] = np.float64):
    angle_rad = np.deg2rad(angle_deg)
    cos = np.cos(angle_rad)
    sin = np.sin(angle_rad)
    return np.array([
        [ cos, 0.0, sin, 0.0],
        [ 0.0, 1.0, 0.0, 0.0],
        [-sin, 0.0, cos, 0.0],
        [ 0.0, 0.0, 0.0, 1.0],
    ], dtype=dtype)

def get_rotation_matrix_z(angle_deg, dtype: Type[np.floating] = np.float64):
    angle_rad = np.deg2rad(angle_deg)
    cos = np.cos(angle_rad)
    sin = np.sin(angle_rad)
    return np.array([
        [ cos,-sin, 0.0, 0.0],
        [ sin, cos, 0.0, 0.0],
        [ 0.0, 0.0, 1.0, 0.0],
        [ 0.0, 0.0, 0.0, 1.0],
    ], dtype=dtype)
