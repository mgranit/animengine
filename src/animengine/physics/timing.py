import math
from collections.abc import Callable


class FixedStepper:
    """Accumulate elapsed time and call step with a fixed dt, retaining fractional time."""

    def __init__(self, step: Callable[[float], None], dt: float) -> None:
        if not callable(step):
            raise TypeError("step must be callable")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("fixed dt must be finite and positive")
        self.step = step
        self.dt = dt
        self.remainder = 0.0

    def advance(self, elapsed: float) -> int:
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("elapsed time must be finite and non-negative")
        if elapsed == 0:
            return 0
        total = self.remainder + elapsed
        if not math.isfinite(total):
            raise ValueError("accumulated time is too large")
        self.remainder = total
        count = 0
        while self.remainder >= self.dt or math.isclose(
            self.remainder, self.dt, rel_tol=1e-12, abs_tol=0.0
        ):
            if self.remainder - self.dt == self.remainder:
                raise ValueError("fixed dt is too small for accumulated time")
            self.step(self.dt)
            self.remainder = max(0.0, self.remainder - self.dt)
            count += 1
        return count
