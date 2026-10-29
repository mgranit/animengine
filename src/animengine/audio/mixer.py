import heapq
import math
import threading
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Samples = NDArray[np.float32]


class Sound:
    """A mono, normalized sound at a fixed sample rate."""

    def __init__(self, samples, sample_rate: int) -> None:
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        data = np.array(samples, dtype=np.float32, copy=True)
        if data.ndim != 1 or len(data) == 0 or not np.all(np.isfinite(data)):
            raise ValueError("sound must be a non-empty, finite mono array")
        if np.any(np.abs(data) > 1):
            raise ValueError("sound samples must be between -1 and 1")
        data.setflags(write=False)
        self.samples = data
        self.sample_rate = sample_rate


def synthesize_thump(
    sample_rate: int = 48_000, frequency: float = 90.0, duration: float = 0.18,
) -> Sound:
    """Create a short low-frequency thump with a smooth attack and decay."""
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if not math.isfinite(frequency) or not 0 < frequency < sample_rate / 4:
        raise ValueError("frequency must be positive and below one quarter of sample_rate")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    count = round(sample_rate * duration)
    if count < 2:
        raise ValueError("duration must span at least two samples")
    t = np.arange(count, dtype=np.float64) / sample_rate
    wave = np.sin(2 * np.pi * frequency * t) + 0.35 * np.sin(4 * np.pi * frequency * t)
    attack = 1 - np.exp(-t / 0.004)
    decay = np.exp(-5 * t / duration)
    fade = np.minimum(1.0, (count - 1 - np.arange(count)) / max(1, sample_rate * 0.01))
    samples = wave * attack * decay * fade
    peak = np.max(np.abs(samples))
    if peak == 0:
        raise ValueError("thump parameters produced silence")
    return Sound(samples / peak, sample_rate)


@dataclass(frozen=True)
class _Voice:
    start: int
    sound: Sound
    gain: float

    @property
    def end(self) -> int:
        return self.start + len(self.sound.samples)


class AudioMixer:
    """Mix timestamped mono sounds into bounded float32 sample blocks.

    Scheduling and rendering can run on different threads. Over-capacity voices are counted in
    dropped_voices; late events are played at the next available sample and issue a warning.
    """

    def __init__(
        self, sample_rate: int = 48_000, *, max_voices: int = 64,
        ceiling: float = 0.85, knee: float = 0.6,
    ) -> None:
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        if not isinstance(max_voices, int) or max_voices <= 0:
            raise ValueError("max_voices must be a positive integer")
        if not math.isfinite(ceiling) or not math.isfinite(knee) or not 0 <= knee < ceiling <= 1:
            raise ValueError("require 0 <= knee < ceiling <= 1")
        self.sample_rate = sample_rate
        self.max_voices = max_voices
        self.ceiling = ceiling
        self.knee = knee
        self._cursor = 0
        self._sequence = 0
        self._pending: list[tuple[int, int, _Voice]] = []
        self._active: list[_Voice] = []
        self._lock = threading.Lock()
        self._limiter_gain = 1.0
        self._release = math.exp(-1.0 / (0.1 * sample_rate))
        self.dropped_voices = 0
        self.late_events = 0

    @property
    def sample_position(self) -> int:
        with self._lock:
            return self._cursor

    def schedule(self, sound: Sound, time: float, gain: float = 1.0) -> None:
        """Schedule a sound at a simulation time, quantized to the nearest sample."""
        if sound.sample_rate != self.sample_rate:
            raise ValueError("sound and mixer sample rates must match")
        if not math.isfinite(time) or time < 0:
            raise ValueError("sound time must be finite and non-negative")
        if not math.isfinite(gain) or not 0 <= gain <= 1:
            raise ValueError("gain must be finite and between 0 and 1")
        sample_time = time * self.sample_rate
        if not math.isfinite(sample_time):
            raise ValueError("sound time is too large")
        start = round(sample_time)
        with self._lock:
            late = start < self._cursor
            if late:
                self.late_events += 1
                start = self._cursor
            voice = _Voice(start, sound, gain)
            heapq.heappush(self._pending, (start, self._sequence, voice))
            self._sequence += 1
        if late:
            warnings.warn("Audio event arrived late; playing it at the next sample.",
                          RuntimeWarning, stacklevel=2)

    def render(self, frames: int) -> Samples:
        """Render the next block, advancing the mixer timeline."""
        if not isinstance(frames, int) or frames < 0:
            raise ValueError("frames must be a non-negative integer")
        with self._lock:
            start = self._cursor
            end = start + frames
            while self._pending and self._pending[0][0] < end:
                _, _, voice = heapq.heappop(self._pending)
                overlapping = sum(other.start <= voice.start < other.end for other in self._active)
                if overlapping >= self.max_voices:
                    self.dropped_voices += 1
                else:
                    self._active.append(voice)

            mixed = np.zeros(frames, dtype=np.float64)
            for voice in self._active:
                first = max(start, voice.start)
                last = min(end, voice.end)
                if first < last:
                    mixed[first - start:last - start] += (
                        voice.gain * voice.sound.samples[first - voice.start:last - voice.start]
                    )
            self._active = [voice for voice in self._active if voice.end > end]
            self._cursor = end

            magnitude = np.abs(mixed)
            over = magnitude > self.knee
            if self._limiter_gain < 1.0 or np.any(over):
                targets = np.ones(frames)
                span = self.ceiling - self.knee
                targets[over] = (
                    self.knee + span * np.tanh((magnitude[over] - self.knee) / span)
                ) / magnitude[over]
                gain = self._limiter_gain
                for index, target in enumerate(targets):
                    gain = min(float(target), 1.0 - (1.0 - gain) * self._release)
                    mixed[index] *= gain
                self._limiter_gain = gain
            return mixed.astype(np.float32)
