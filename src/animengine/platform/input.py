from collections.abc import Callable

import glfw


class Input:
    """Fans GLFW input events out to any number of listeners."""

    def __init__(self, handle) -> None:
        self._handle = handle
        self._key_listeners: list[tuple[int, int, int, Callable[[], None]]] = []
        self._cursor_listeners: list[Callable[[float, float], None]] = []
        self._scroll_listeners: list[Callable[[float, float], None]] = []

        glfw.set_key_callback(handle, self._on_key)
        glfw.set_cursor_pos_callback(handle, self._on_cursor)
        glfw.set_scroll_callback(handle, self._on_scroll)

    def on_key(self, key: int, action: int, callback: Callable[[], None], mods: int = 0) -> None:
        self._key_listeners.append((key, action, mods, callback))

    def on_cursor(self, callback: Callable[[float, float], None]) -> None:
        self._cursor_listeners.append(callback)

    def on_scroll(self, callback: Callable[[float, float], None]) -> None:
        self._scroll_listeners.append(callback)

    def is_key_down(self, key: int) -> bool:
        return glfw.get_key(self._handle, key) == glfw.PRESS

    def _on_key(self, _window, key, _scancode, action, mods) -> None:
        for k, a, m, callback in self._key_listeners:
            if k == key and a == action and m == mods:
                callback()

    def _on_cursor(self, _window, x, y) -> None:
        for callback in self._cursor_listeners:
            callback(x, y)

    def _on_scroll(self, _window, dx, dy) -> None:
        for callback in self._scroll_listeners:
            callback(dx, dy)
