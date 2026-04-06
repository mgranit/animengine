import glfw
import OpenGL.GL as gl
from typing import Callable

class Window:
    _glfw_initialized = False
    
    def __init__(
        self,
        width: int,
        height: int,
        title: str = "Window",
        version_major: int = 3,
        version_minor: int = 3,
        profile: int = glfw.OPENGL_CORE_PROFILE
        ) -> None:
        
        if not Window._glfw_initialized:
            if not glfw.init():
                raise Exception("Failed to initialize GLFW")
            Window._glfw_initialized = True
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, version_major)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, version_minor)
        glfw.window_hint(glfw.OPENGL_PROFILE, profile)
        
        self._window = glfw.create_window(width, height, title, None, None)
        if not self._window:
            glfw.terminate()
            raise Exception("Failed to create GLFW window")
        
        glfw.make_context_current(self._window)
        glfw.set_framebuffer_size_callback(self._window, self._framebuffer_size_callback)
        glfw.set_key_callback(self._window, self._key_callback)
        gl.glViewport(0, 0, width, height)
        
        self.key_callbacks = []
    
    def _framebuffer_size_callback(self, window: glfw._GLFWwindow, width: int, height: int) -> None:
        gl.glViewport(0, 0, width, height)

    def _key_callback(self, window: glfw._GLFWwindow, key: int, scancode: int, action: int, mods: int) -> None:
        for callback in self.key_callbacks:
            if (
                callback["key"] == key
                and callback["action"] == action
                and callback["mods"] == mods
            ):
                callback["callback"]()
    
    def add_key_callback(self, key: int, action: int, callback: Callable, mods: int = 0) -> None:
        self.key_callbacks.append({
            "key": key,
            "action": action,
            "mods": mods,
            "callback": callback
        })
    
    def should_close(self) -> bool:
        return glfw.window_should_close(self._window)
    
    def set_should_close(self, value: bool) -> None:
        glfw.set_window_should_close(self._window, value)

    def is_key_down(self, key: int) -> bool:
        return (glfw.get_key(self._window, key) == glfw.PRESS)
        
    @property
    def resolution(self) -> tuple[int, int]:
        return glfw.get_window_size(self._window)
    
    @property
    def width(self) -> int:
        return self.resolution[0]
    
    @property
    def height(self) -> int:
        return self.resolution[1]
    
    @property
    def aspect_ratio(self) -> float:
        width, height = self.resolution
        return width / height
