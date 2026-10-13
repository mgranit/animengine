from collections.abc import Iterator

import numpy as np

from animengine.linalg import (
    get_rotation_matrix_x,
    get_rotation_matrix_y,
    get_rotation_matrix_z,
    get_scaling_matrix,
    get_translation_matrix,
)
from animengine.render import DrawItem, Material, Mesh


class Node:
    """A transform in the scene graph, optionally drawing a mesh.

    `rotation` is in degrees about x, y, z and is applied in z-y-x order.
    """

    def __init__(
        self,
        mesh: Mesh | None = None,
        material: Material | None = None,
        *,
        position=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0),
        scale=1.0,
        visible: bool = True,
    ) -> None:
        self.mesh = mesh
        self.material = material if material is not None else Material()
        self.position = np.array(position, dtype=np.float64)
        self.rotation = np.array(rotation, dtype=np.float64)
        self.scale = scale
        self.visible = visible
        self.children: list[Node] = []

    @property
    def scale(self) -> np.ndarray:
        return self._scale

    @scale.setter
    def scale(self, value) -> None:
        self._scale = np.broadcast_to(np.asarray(value, dtype=np.float64), (3,)).copy()

    def add(self, child: "Node") -> "Node":
        self.children.append(child)
        return child

    def remove(self, child: "Node") -> None:
        self.children.remove(child)

    @property
    def local_matrix(self) -> np.ndarray:
        rx, ry, rz = self.rotation
        return (
            get_translation_matrix(*self.position)
            @ get_rotation_matrix_z(rz)
            @ get_rotation_matrix_y(ry)
            @ get_rotation_matrix_x(rx)
            @ get_scaling_matrix(*self.scale)
        )

    def draw_items(self, parent_matrix: np.ndarray | None = None) -> Iterator[DrawItem]:
        if not self.visible:
            return
        world = self.local_matrix if parent_matrix is None else parent_matrix @ self.local_matrix
        if self.mesh is not None:
            yield DrawItem(self.mesh, self.material, world)
        for child in self.children:
            yield from child.draw_items(world)
