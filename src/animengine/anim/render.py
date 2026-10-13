import sys
from collections.abc import Callable

import numpy as np
import OpenGL.GL as gl

from animengine.gpu import Framebuffer
from animengine.platform import FixedClock, Window
from animengine.render import Renderer
from animengine.scene import Scene

from .exporter import VideoWriter

UpdateFn = Callable[[float, float], None]


class OfflineSession:
    """Hidden GL context plus a framebuffer of the scene's size."""

    def __init__(self, size: tuple[int, int], samples: int = 4) -> None:
        self.size = size
        self.window = Window(*size, "animengine offline", visible=False, vsync=False)
        try:
            samples = min(samples, int(gl.glGetIntegerv(gl.GL_MAX_SAMPLES)))
            self.framebuffer = Framebuffer(*size, samples=samples)
            self.renderer = Renderer()
        except Exception:
            self.window.close()
            raise

    def render(self, scene: Scene) -> np.ndarray:
        scene.render(self.renderer, target=self.framebuffer)
        return self.framebuffer.read_pixels()

    def close(self) -> None:
        self.window.close()

    def __enter__(self) -> "OfflineSession":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def render_frame(scene: Scene, samples: int = 4) -> np.ndarray:
    """Render the scene's current state to a top-down HxWx3 uint8 array."""
    with OfflineSession(scene.size, samples) as session:
        return session.render(scene)


def render_video(
    scene: Scene,
    update: UpdateFn | None,
    duration: float,
    fps: float,
    out,
    *,
    steps_per_frame: int = 1,
    samples: int = 4,
    crf: int = 17,
    progress: bool = True,
) -> None:
    """Render `duration` seconds of the scene to a video, independent of wall-clock speed.

    `update(t, dt)` brings the scene to time `t` after `dt` seconds have passed since the previous
    call. It is called with (0, 0) before the first frame, and `steps_per_frame` times per later
    frame with dt = 1 / (fps * steps_per_frame).
    """
    n_frames = round(duration * fps)
    if n_frames < 1:
        raise ValueError("duration * fps must be at least one frame")

    clock = FixedClock(fps)
    sub_dt = clock.dt / steps_per_frame

    with (
        OfflineSession(scene.size, samples) as session,
        VideoWriter(out, scene.size, fps, crf) as writer,
    ):
        for i in range(n_frames):
            if update is not None:
                if i == 0:
                    update(0.0, 0.0)
                else:
                    for k in range(steps_per_frame):
                        update((i - 1) * clock.dt + (k + 1) * sub_dt, sub_dt)
            writer.write(session.render(scene))
            clock.tick()

            if progress:
                print(f"\rframe {i + 1}/{n_frames}", end="", file=sys.stderr, flush=True)
    if progress:
        print(file=sys.stderr)
