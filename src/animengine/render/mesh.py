from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class DrawMode(Enum):
    POINTS = "points"
    LINES = "lines"
    LINE_STRIP = "line_strip"
    TRIANGLES = "triangles"


_ATTRIBUTES = ("positions", "colors", "uvs", "indices")


@dataclass(eq=False)
class Mesh:
    """CPU-side geometry; each Renderer uploads it and re-uploads whatever changed.

    positions are (N, 2) or (N, 3); colors (N, 3); uvs (N, 2); indices (M,).
    """

    positions: np.ndarray
    mode: DrawMode = DrawMode.TRIANGLES
    colors: np.ndarray | None = None
    uvs: np.ndarray | None = None
    indices: np.ndarray | None = None
    dynamic: bool = False
    _versions: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.set_positions(self.positions)
        if self.colors is not None:
            self.set_colors(self.colors)
        if self.uvs is not None:
            self.set_uvs(self.uvs)
        if self.indices is not None:
            self.set_indices(self.indices)

    def _touch(self, name: str) -> None:
        self._versions[name] = self._versions.get(name, 0) + 1

    @staticmethod
    def _check(name: str, array, widths: tuple[int, ...]) -> np.ndarray:
        array = np.ascontiguousarray(array, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] not in widths:
            raise ValueError(f"{name} must have shape (N, {' or '.join(map(str, widths))}), "
                             f"got {array.shape}")
        return array

    def set_positions(self, positions) -> None:
        self.positions = self._check("positions", positions, (2, 3))
        self._touch("positions")

    def set_colors(self, colors) -> None:
        self.colors = self._check("colors", colors, (3,))
        self._touch("colors")

    def set_uvs(self, uvs) -> None:
        self.uvs = self._check("uvs", uvs, (2,))
        self._touch("uvs")

    def set_indices(self, indices) -> None:
        self.indices = np.ascontiguousarray(indices, dtype=np.uint32).ravel()
        self._touch("indices")

    @property
    def vertex_count(self) -> int:
        return len(self.positions)

    def changed_since(self, seen: dict[str, int]) -> list[str]:
        """Attributes whose version differs from the versions in `seen`."""
        return [name for name in _ATTRIBUTES
                if name in self._versions and seen.get(name) != self._versions[name]]

    def version(self, name: str) -> int:
        return self._versions[name]
