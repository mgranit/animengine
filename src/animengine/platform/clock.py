from .window import Window


class FixedClock:
    """Deterministic time: frame i is always at t = i / fps, regardless of wall-clock speed."""

    def __init__(self, fps: float) -> None:
        self.dt = 1.0 / fps
        self.frame = 0

    @property
    def t(self) -> float:
        return self.frame * self.dt

    def tick(self) -> None:
        self.frame += 1


class RealtimeClock:
    """Wall-clock time for interactive preview."""

    def __init__(self) -> None:
        self._start = Window.time()
        self._last = self._start
        self.t = 0.0
        self.dt = 0.0

    def tick(self) -> None:
        now = Window.time()
        self.dt = now - self._last
        self.t = now - self._start
        self._last = now
