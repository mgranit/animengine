import glfw
import numpy as np

from animengine.linalg import Frame
from animengine.platform import Input

from .camera import PerspectiveCamera


class FlyController:
    """WASD/mouse free-fly control for a z-up PerspectiveCamera (preview only)."""

    def __init__(
        self,
        camera: PerspectiveCamera,
        input: Input,
        speed: float = 2.5,
        sensitivity: float = 0.1,
    ) -> None:
        self.camera = camera
        self.input = input
        self.speed = speed
        self.sensitivity = sensitivity

        front = -camera.frame.z_axis
        self.polar = float(np.degrees(np.arcsin(front[2])))
        self.azimuth = float(np.degrees(np.arctan2(front[1], front[0])))

        self._last_cursor: tuple[float, float] | None = None
        input.on_cursor(self._look)
        input.on_scroll(self._zoom)

    def _front(self) -> np.ndarray:
        polar = np.radians(self.polar)
        azimuth = np.radians(self.azimuth)
        return np.array([
            np.cos(polar) * np.cos(azimuth),
            np.cos(polar) * np.sin(azimuth),
            np.sin(polar),
        ])

    def _frame(self) -> Frame:
        azimuth = np.radians(self.azimuth)
        front = self._front()
        right = np.array([np.sin(azimuth), -np.cos(azimuth), 0.0])
        up = np.cross(right, front)
        return Frame(self.camera.position, np.column_stack((right, up, -front)))

    def _look(self, x: float, y: float) -> None:
        if self._last_cursor is not None:
            dx = x - self._last_cursor[0]
            dy = self._last_cursor[1] - y
            self.azimuth -= dx * self.sensitivity
            self.polar = float(np.clip(self.polar + dy * self.sensitivity, -90.0, 90.0))
        self._last_cursor = (x, y)

    def _zoom(self, _dx: float, dy: float) -> None:
        self.camera.fov = float(np.clip(self.camera.fov - dy, 15.0, 45.0))

    def update(self, dt: float) -> None:
        frame = self._frame()
        front, right, up = -frame.z_axis, frame.x_axis, frame.y_axis

        step = self.speed * dt
        move = np.zeros(3)
        if self.input.is_key_down(glfw.KEY_W):
            move += front
        if self.input.is_key_down(glfw.KEY_S):
            move -= front
        if self.input.is_key_down(glfw.KEY_D):
            move += right
        if self.input.is_key_down(glfw.KEY_A):
            move -= right
        if self.input.is_key_down(glfw.KEY_SPACE):
            move += up
        if self.input.is_key_down(glfw.KEY_LEFT_CONTROL):
            move -= up

        self.camera.frame = Frame(frame.origin + step * move, frame.basis)
