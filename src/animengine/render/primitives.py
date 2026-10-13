from functools import cache
from itertools import product

import numpy as np

from .mesh import DrawMode, Mesh


def _cube_faces() -> tuple[np.ndarray, np.ndarray]:
    positions, uvs = [], []
    corners = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
    for axis, sign in product(range(3), (-1.0, 1.0)):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        order = [0, 1, 2, 0, 2, 3] if sign > 0 else [0, 2, 1, 0, 3, 2]  # outward-facing CCW
        for i in order:
            vertex = np.zeros(3)
            vertex[axis] = sign * 0.5
            vertex[u], vertex[v] = corners[i]
            positions.append(vertex)
            uvs.append((corners[i][0] + 0.5, corners[i][1] + 0.5))
    return np.array(positions, dtype=np.float32), np.array(uvs, dtype=np.float32)


@cache
def unit_cube() -> Mesh:
    return Mesh(_cube_faces()[0])


@cache
def unit_textured_cube() -> Mesh:
    positions, uvs = _cube_faces()
    return Mesh(positions, uvs=uvs)


@cache
def unit_cube_edges() -> Mesh:
    corners = np.array(list(product((-0.5, 0.5), repeat=3)), dtype=np.float32)
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
             if np.count_nonzero(corners[i] != corners[j]) == 1]
    return Mesh(corners[np.array(edges).ravel()], mode=DrawMode.LINES)


@cache
def unit_axes() -> Mesh:
    positions = np.array([
        [0, 0, 0], [1, 0, 0],
        [0, 0, 0], [0, 1, 0],
        [0, 0, 0], [0, 0, 1],
    ], dtype=np.float32)
    colors = np.array([
        [1, 0, 0], [1, 0, 0],
        [0, 1, 0], [0, 1, 0],
        [0, 0, 1], [0, 0, 1],
    ], dtype=np.float32)
    return Mesh(positions, mode=DrawMode.LINES, colors=colors)


def make_grid(size: float = 10.0, step: float = 1.0) -> Mesh:
    """Lines on the XY plane covering [-size/2, size/2] in both directions."""
    half = size / 2.0
    ticks = np.arange(-half, half + step / 2.0, step)
    segments = []
    for t in ticks:
        segments += [[t, -half, 0.0], [t, half, 0.0], [-half, t, 0.0], [half, t, 0.0]]
    return Mesh(np.array(segments, dtype=np.float32), mode=DrawMode.LINES)
