import numpy as np

from animengine.render import DrawMode, Material, Mesh, Texture, primitives

from .node import Node


class Cube(Node):
    def __init__(self, position=(0.0, 0.0, 0.0), size: float = 1.0, color="#ffffff",
                 wireframe: bool = False, **kwargs) -> None:
        mesh = primitives.unit_cube_edges() if wireframe else primitives.unit_cube()
        super().__init__(mesh, Material(color=color), position=position, scale=size, **kwargs)


class TexturedCube(Node):
    def __init__(self, texture: Texture, position=(0.0, 0.0, 0.0), size: float = 1.0,
                 **kwargs) -> None:
        material = Material(kind="textured", texture=texture)
        super().__init__(primitives.unit_textured_cube(), material, position=position,
                         scale=size, **kwargs)


class Axes(Node):
    """RGB axes for x, y, z."""

    def __init__(self, length: float = 1.0, position=(0.0, 0.0, 0.0), line_width: float = 1.0,
                 **kwargs) -> None:
        material = Material(kind="vertex_color", line_width=line_width)
        super().__init__(primitives.unit_axes(), material, position=position, scale=length,
                         **kwargs)


class Grid(Node):
    """Grid on the XY plane."""

    def __init__(self, size: float = 10.0, step: float = 1.0, color="#808080", **kwargs) -> None:
        super().__init__(primitives.make_grid(size, step), Material(color=color), **kwargs)


class Lines(Node):
    """Line segments (consecutive point pairs), or a connected strip with `strip=True`."""

    def __init__(self, points=None, color="#ffffff", width: float = 1.0, strip: bool = False,
                 **kwargs) -> None:
        mode = DrawMode.LINE_STRIP if strip else DrawMode.LINES
        mesh = Mesh(np.zeros((0, 3)) if points is None else points, mode=mode, dynamic=True)
        super().__init__(mesh, Material(color=color, line_width=width), **kwargs)

    def set_points(self, points) -> None:
        self.mesh.set_positions(points)


class Points(Node):
    """Round points; `size` is in pixels of the render target."""

    def __init__(self, points=None, color="#ffffff", size: float = 3.0, **kwargs) -> None:
        mesh = Mesh(np.zeros((0, 3)) if points is None else points, mode=DrawMode.POINTS,
                    dynamic=True)
        super().__init__(mesh, Material(kind="points", color=color, point_size=size), **kwargs)

    def set_points(self, points) -> None:
        self.mesh.set_positions(points)
