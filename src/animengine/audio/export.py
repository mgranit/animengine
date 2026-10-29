import wave

import numpy as np

from .mixer import AudioMixer


class WaveWriter:
    """Write a mix incrementally as mono PCM, fading the end of the video."""

    def __init__(self, path, mixer: AudioMixer, total_samples: int, fade_seconds: float = 0.05):
        if total_samples < 1:
            raise ValueError("audio duration must contain at least one sample")
        if mixer.sample_position != 0:
            raise ValueError("audio mixer must be at the start of its timeline")
        if not np.isfinite(fade_seconds) or fade_seconds < 0:
            raise ValueError("fade_seconds must be finite and non-negative")
        self.mixer = mixer
        self.total_samples = total_samples
        self.fade_samples = min(total_samples, round(fade_seconds * mixer.sample_rate))
        self.position = 0
        self._file = wave.open(str(path), "wb")
        self._file.setnchannels(1)
        self._file.setsampwidth(2)
        self._file.setframerate(mixer.sample_rate)

    def write_until(self, target: int) -> None:
        if not self.position <= target <= self.total_samples:
            raise ValueError("audio target must be within the remaining video duration")
        while self.position < target:
            count = min(4096, target - self.position)
            samples = self.mixer.render(count)
            if self.fade_samples:
                indices = np.arange(self.position, self.position + count)
                fade = np.clip(
                    (self.total_samples - 1 - indices) / self.fade_samples, 0.0, 1.0,
                )
                samples = samples * fade
            pcm = np.rint(samples * 32767).astype("<i2")
            self._file.writeframesraw(pcm.tobytes())
            self.position += count

    def __enter__(self) -> "WaveWriter":
        return self

    def __exit__(self, *_) -> None:
        self._file.close()
