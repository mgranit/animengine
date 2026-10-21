import numpy as np

from animengine.linalg import Frame, orthographic, perspective


class Camera:
    """Pose and projection only; input handling lives in controllers.

    The pose is a `Frame` that looks down its local -z with +y up. `position`, `target` and `up`
    are a look-at view of it: setting `position` or `target` re-aims the camera at the other,
    using `up` as the up hint. `target` is a point `look_distance` in front of the camera.

    Set `parent` to a scene node (anything with a `world_frame`) to mount the camera on it. The
    pose is then expressed in that node's frame, and the view follows the node as it moves.
    """

    def __init__(self, position, target, up) -> None:
        self._up = np.array(up, dtype=np.float64)
        self._frame = Frame.look_at(position, target, self._up)
        self._look_distance = float(np.linalg.norm(np.subtract(target, position)))
        self.parent = None

    @property
    def frame(self) -> Frame:
        return self._frame

    @frame.setter
    def frame(self, value: Frame) -> None:
        self._frame = value

    @property
    def world_frame(self) -> Frame:
        if self.parent is None:
            return self._frame
        return self.parent.world_frame @ self._frame

    @property
    def position(self) -> np.ndarray:
        return self._frame.origin

    @position.setter
    def position(self, value) -> None:
        self._aim(value, self.target)

    @property
    def target(self) -> np.ndarray:
        return self._frame.origin - self._look_distance * self._frame.z_axis

    @target.setter
    def target(self, value) -> None:
        self._aim(self.position, value)

    @property
    def up(self) -> np.ndarray:
        return self._up

    @up.setter
    def up(self, value) -> None:
        self._up = np.array(value, dtype=np.float64)
        self._aim(self.position, self.target)

    def _aim(self, position, target) -> None:
        self._frame = Frame.look_at(position, target, self._up)
        self._look_distance = float(np.linalg.norm(np.subtract(target, position)))

    @property
    def view_matrix(self) -> np.ndarray:
        return self.world_frame.inverse_matrix(np.float32)

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
        near: float = 0.0,
        far: float = 2.0,
    ) -> None:
        super().__init__(position, target, up)
        self.height = height
        self.near = near
        self.far = far

    def projection_matrix(self, aspect: float) -> np.ndarray:
        height = self.height
        width = height * aspect
        return orthographic(width, height, self.near, self.far)
