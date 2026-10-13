import glfw
import OpenGL.GL as gl

from .input import Input


class Window:
    """A GLFW window with an OpenGL context; hidden windows are used for offline rendering."""

    _open_windows = 0

    def __init__(
        self,
        width: int,
        height: int,
        title: str = "animengine",
        *,
        visible: bool = True,
        version: tuple[int, int] = (3, 3),
        samples: int = 0,
        vsync: bool = True,
        cursor_disabled: bool = False,
    ) -> None:
        if Window._open_windows == 0 and not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.default_window_hints()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, version[0])
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, version[1])
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, samples)
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE if visible else glfw.FALSE)

        handle = glfw.create_window(width, height, title, None, None)
        if not handle:
            if Window._open_windows == 0:
                glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        self._handle = handle
        Window._open_windows += 1

        glfw.make_context_current(handle)
        glfw.swap_interval(1 if vsync else 0)
        if cursor_disabled:
            glfw.set_input_mode(handle, glfw.CURSOR, glfw.CURSOR_DISABLED)

        self.input = Input(handle)
        gl.glViewport(0, 0, *self.framebuffer_size)

    @property
    def handle(self):
        return self._handle

    @property
    def framebuffer_size(self) -> tuple[int, int]:
        return glfw.get_framebuffer_size(self._handle)

    @property
    def aspect_ratio(self) -> float:
        width, height = self.framebuffer_size
        return width / height

    def make_current(self) -> None:
        glfw.make_context_current(self._handle)

    def should_close(self) -> bool:
        return glfw.window_should_close(self._handle)

    def set_should_close(self, value: bool = True) -> None:
        glfw.set_window_should_close(self._handle, value)

    def swap_buffers(self) -> None:
        glfw.swap_buffers(self._handle)

    def poll_events(self) -> None:
        glfw.poll_events()

    @staticmethod
    def time() -> float:
        return glfw.get_time()

    def close(self) -> None:
        if self._handle is None:
            return
        glfw.destroy_window(self._handle)
        self._handle = None
        Window._open_windows -= 1
        if Window._open_windows == 0:
            glfw.terminate()

    def __enter__(self) -> "Window":
        return self

    def __exit__(self, *_) -> None:
        self.close()
