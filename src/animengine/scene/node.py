from collections.abc import Iterator

import numpy as np

from animengine.linalg import Frame, get_scaling_matrix
from animengine.render import DrawItem, Material, Mesh


class Node:
    """A pose in the scene graph, optionally drawing a mesh.

    The pose is a `Frame` relative to the parent node, plus a separate per-axis `scale` that
    applies to the node and its children but is not part of the frame. `position` and `rotation`
    (degrees about x, y, z, applied in z-y-x order) are conveniences over `frame`; they are
    read-only views, so assign a new value rather than editing them in place. Pass either `frame`
    or `position`/`rotation` to the constructor, not both.
    """

    def __init__(
        self,
        mesh: Mesh | None = None,
        material: Material | None = None,
        *,
        position=None,
        rotation=None,
        frame: Frame | None = None,
        scale=1.0,
        visible: bool = True,
    ) -> None:
        if frame is not None:
            if position is not None or rotation is not None:
                raise ValueError("pass either frame or position/rotation, not both")
        else:
            frame = Frame.from_euler(
                (0.0, 0.0, 0.0) if rotation is None else rotation,
                (0.0, 0.0, 0.0) if position is None else position,
            )
        self.mesh = mesh
        self.material = material if material is not None else Material()
        self.frame = frame
        self.scale = scale
        self.visible = visible
        self.parent: Node | None = None
        self.children: list[Node] = []

    @property
    def position(self) -> np.ndarray:
        return self.frame.origin

    @position.setter
    def position(self, value) -> None:
        self.frame = Frame(value, self.frame.basis)

    @property
    def rotation(self) -> np.ndarray:
        euler = self.frame.to_euler()
        euler.setflags(write=False)
        return euler

    @rotation.setter
    def rotation(self, value) -> None:
        self.frame = Frame.from_euler(value, self.frame.origin)

    @property
    def scale(self) -> np.ndarray:
        return self._scale

    @scale.setter
    def scale(self, value) -> None:
        self._scale = np.broadcast_to(np.asarray(value, dtype=np.float64), (3,)).copy()

    def add(self, child: "Node") -> "Node":
        if child.parent is not None:
            raise ValueError("node already has a parent; remove it from there first")
        ancestor: Node | None = self
        while ancestor is not None:
            if ancestor is child:
                raise ValueError("a node cannot be added beneath itself")
            ancestor = ancestor.parent
        self.children.append(child)
        child.parent = self
        return child

    def remove(self, child: "Node") -> None:
        self.children.remove(child)
        child.parent = None

    @property
    def local_matrix(self) -> np.ndarray:
        return self.frame.matrix() @ get_scaling_matrix(*self.scale)

    @property
    def world_matrix(self) -> np.ndarray:
        """Local-to-world matrix including scale, relative to the topmost ancestor."""
        matrix = self.local_matrix
        node = self.parent
        while node is not None:
            matrix = node.local_matrix @ matrix
            node = node.parent
        return matrix

    @property
    def world_frame(self) -> Frame:
        """The node's rigid pose in the topmost ancestor's frame.

        Scale is discarded, so this is exact when every scale along the chain is uniform and the
        nearest rotation otherwise.
        """
        return Frame.from_matrix(self.world_matrix, orthonormalize=True)

    def draw_items(self, parent_matrix: np.ndarray | None = None) -> Iterator[DrawItem]:
        if not self.visible:
            return
        world = self.local_matrix if parent_matrix is None else parent_matrix @ self.local_matrix
        if self.mesh is not None:
            yield DrawItem(self.mesh, self.material, world)
        for child in self.children:
            yield from child.draw_items(world)
