from contextlib import ExitStack

import glfw

from animengine.audio import AudioMixer
from animengine.audio.playback import DEFAULT_AUDIO_DELAY, AudioPlayer
from animengine.platform import RealtimeClock, Window
from animengine.render import FlyController, PerspectiveCamera, Renderer
from animengine.scene import Scene

from .render import UpdateFn


def preview(
    scene: Scene,
    update: UpdateFn | None = None,
    *,
    title: str = "animengine",
    fly_camera: bool = False,
    samples: int = 4,
    audio: AudioMixer | None = None,
    audio_delay: float = DEFAULT_AUDIO_DELAY,
    audio_device: int | str | None = None,
) -> None:
    """Play the scene in a window in real time; ESC closes it.

    With `fly_camera` (perspective cameras only) WASD, space/ctrl, mouse and scroll move the camera.
    Optional audio is delayed to give the simulation time to schedule collision sounds.
    """
    with Window(*scene.size, title, samples=samples, cursor_disabled=fly_camera) as window:
        window.input.on_key(glfw.KEY_ESCAPE, glfw.PRESS, window.set_should_close)

        controller = None
        if fly_camera:
            if not isinstance(scene.camera, PerspectiveCamera):
                raise ValueError("fly_camera needs a PerspectiveCamera")
            controller = FlyController(scene.camera, window.input)

        renderer = Renderer()
        with ExitStack() as stack:
            player = (
                stack.enter_context(AudioPlayer(audio, delay=audio_delay, device=audio_device))
                if audio else None
            )
            clock = RealtimeClock()
            while not window.should_close():
                clock.tick()
                if controller is not None:
                    controller.update(clock.dt)
                if update is not None:
                    update(clock.t, clock.dt)
                if player is not None:
                    player.check()

                scene.render(renderer, size=window.framebuffer_size)
                window.swap_buffers()
                window.poll_events()
            if player is not None:
                player.check()
