import numpy as np

from animengine.linalg import look_at, orthographic, perspective


class Camera:
    """Position, orientation and projection only; input handling lives in controllers."""

    def __init__(self, position, target, up) -> None:
        self.position = np.array(position, dtype=np.float64)
        self.target = np.array(target, dtype=np.float64)
        self.up = np.array(up, dtype=np.float64)

    @property
    def view_matrix(self) -> np.ndarray:
        return look_at(self.position, self.target, self.up)

    def projection_matrix(self, aspect: float) -> np.ndarray:
        raise NotImplementedError


class PerspectiveCamera(Camera):
    """3D camera; the world is z-up."""

    def __init__(
        self,
        position=(3.0, 3.0, 3.0),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
        fov: float = 45.0,
        near: float = 0.1,
        far: float = 100.0,
    ) -> None:
        super().__init__(position, target, up)
        self.fov = fov
        self.near = near
        self.far = far

    def projection_matrix(self, aspect: float) -> np.ndarray:
        return perspective(self.fov, aspect, self.near, self.far)


class OrthographicCamera(Camera):
    """2D camera looking down -z; `height` is the visible world height."""

    def __init__(
        self,
        height: float = 2.0,
        position=(0.0, 0.0, 1.0),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        near: float = 0.1,
        far: float = 100.0,
    ) -> None:
        super().__init__(position, target, up)
        self.height = height
        self.near = near
        self.far = far

    def projection_matrix(self, aspect: float) -> np.ndarray:
        half_h = self.height / 2.0
        half_w = half_h * aspect
        return orthographic(-half_w, half_w, -half_h, half_h, self.near, self.far)
