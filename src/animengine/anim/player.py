import glfw

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
) -> None:
    """Play the scene in a window in real time; ESC closes it.

    With `fly_camera` (perspective cameras only) WASD, space/ctrl, mouse and scroll move the camera.
    """
    with Window(*scene.size, title, samples=samples, cursor_disabled=fly_camera) as window:
        window.input.on_key(glfw.KEY_ESCAPE, glfw.PRESS, window.set_should_close)

        controller = None
        if fly_camera:
            if not isinstance(scene.camera, PerspectiveCamera):
                raise ValueError("fly_camera needs a PerspectiveCamera")
            controller = FlyController(scene.camera, window.input)

        renderer = Renderer()
        clock = RealtimeClock()
        while not window.should_close():
            clock.tick()
            if controller is not None:
                controller.update(clock.dt)
            if update is not None:
                update(clock.t, clock.dt)

            scene.render(renderer, size=window.framebuffer_size)
            window.swap_buffers()
            window.poll_events()
