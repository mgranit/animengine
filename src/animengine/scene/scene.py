
from animengine.gpu import Framebuffer
from animengine.render import Camera, PerspectiveCamera, Renderer

from .node import Node


class Scene:
    def __init__(
        self,
        size: tuple[int, int] = (1280, 720),
        background="#000000",
        camera: Camera | None = None,
    ) -> None:
        self.size = size
        self.background = background
        self.camera = camera if camera is not None else PerspectiveCamera()
        self.root = Node()

    def add(self, node: Node) -> Node:
        return self.root.add(node)

    def remove(self, node: Node) -> None:
        self.root.remove(node)

    def render(
        self,
        renderer: Renderer,
        *,
        target: Framebuffer | None = None,
        size: tuple[int, int] | None = None,
    ) -> None:
        renderer.render(
            self.root.draw_items(),
            self.camera,
            background=self.background,
            target=target,
            size=size,
        )
