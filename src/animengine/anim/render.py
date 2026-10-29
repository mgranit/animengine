import sys
import tempfile
import warnings
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import OpenGL.GL as gl

from animengine.audio import AudioMixer
from animengine.audio.export import WaveWriter
from animengine.gpu import Framebuffer
from animengine.platform import FixedClock, Window
from animengine.render import Renderer
from animengine.scene import Scene

from .exporter import VideoWriter, mux_audio

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
    audio: AudioMixer | None = None,
    audio_fade: float = 0.05,
) -> None:
    """Render `duration` seconds of the scene to a video, independent of wall-clock speed.

    `update(t, dt)` brings the scene to time `t` after `dt` seconds have passed since the previous
    call. It is called with (0, 0) before the first frame, and `steps_per_frame` times per later
    frame with dt = 1 / (fps * steps_per_frame). With audio, update is also called for the
    final frame interval to generate events through the end of the video.
    """
    n_frames = round(duration * fps)
    if n_frames < 1:
        raise ValueError("duration * fps must be at least one frame")

    clock = FixedClock(fps)
    sub_dt = clock.dt / steps_per_frame

    with ExitStack() as workspace:
        video_path = out
        audio_path = None
        if audio is not None:
            directory = Path(workspace.enter_context(tempfile.TemporaryDirectory()))
            video_path = directory / "video.mp4"
            audio_path = directory / "audio.wav"

        with (
            OfflineSession(scene.size, samples) as session,
            VideoWriter(video_path, scene.size, fps, crf) as writer,
            ExitStack() as audio_resources,
        ):
            wave = None
            if audio is not None:
                total_samples = round(n_frames * clock.dt * audio.sample_rate)
                wave = audio_resources.enter_context(
                    WaveWriter(audio_path, audio, total_samples, audio_fade)
                )
            for i in range(n_frames):
                if update is not None:
                    if i == 0:
                        update(0.0, 0.0)
                    else:
                        for k in range(steps_per_frame):
                            update((i - 1) * clock.dt + (k + 1) * sub_dt, sub_dt)
                if wave is not None:
                    wave.write_until(round(i * clock.dt * wave.mixer.sample_rate))
                writer.write(session.render(scene))
                clock.tick()

                if progress:
                    print(f"\rframe {i + 1}/{n_frames}", end="", file=sys.stderr, flush=True)
            if wave is not None:
                if update is not None:
                    for k in range(steps_per_frame):
                        update((n_frames - 1) * clock.dt + (k + 1) * sub_dt, sub_dt)
                wave.write_until(wave.total_samples)

        if audio is not None:
            mux_audio(video_path, audio_path, out)
            if audio.dropped_voices:
                warnings.warn(f"Audio voice limit dropped {audio.dropped_voices} sounds.",
                              RuntimeWarning, stacklevel=2)
    if progress:
        print(file=sys.stderr)
