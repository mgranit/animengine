import sys
import threading
import warnings

import numpy as np

from .mixer import AudioMixer

DEFAULT_AUDIO_DELAY = 0.06


def _preferred_output_device(sounddevice) -> int | None:
    if sys.platform != "win32":
        return None
    apis = {api["name"]: api for api in sounddevice.query_hostapis()}
    mme = apis.get("MME")
    wasapi = apis.get("Windows WASAPI")
    if mme is None or wasapi is None:
        return None
    default_mme = mme["default_output_device"]
    default_wasapi = wasapi["default_output_device"]
    if (default_mme >= 0 and default_wasapi >= 0
            and sounddevice.default.device[1] == default_mme):
        return default_wasapi
    return None


class AudioPlayer:
    """Play a mixer through an optional sounddevice output stream."""

    def __init__(
        self, mixer: AudioMixer, *, delay: float = DEFAULT_AUDIO_DELAY, blocksize: int = 512,
        device: int | str | None = None,
    ) -> None:
        if not np.isfinite(delay) or delay < 0:
            raise ValueError("audio delay must be finite and non-negative")
        if not isinstance(blocksize, int) or blocksize <= 0:
            raise ValueError("blocksize must be a positive integer")
        if mixer.sample_position != 0:
            raise ValueError("audio mixer must be at the start of its timeline")
        self.mixer = mixer
        self.delay = delay
        self.blocksize = blocksize
        self.device = device
        self._leading_samples = round(delay * mixer.sample_rate)
        self._lock = threading.Lock()
        self._statuses: list[str] = []
        self._error: Exception | None = None
        self._reported_drops = 0
        self._stream = None

    def _callback(self, outdata, frames, _, status) -> None:
        try:
            if status:
                with self._lock:
                    self._statuses.append(str(status))
            outdata.fill(0)
            leading = min(self._leading_samples, frames)
            self._leading_samples -= leading
            if frames > leading:
                outdata[leading:, 0] = self.mixer.render(frames - leading)
        except Exception as error:
            from sounddevice import CallbackAbort

            outdata.fill(0)
            with self._lock:
                self._error = error
            raise CallbackAbort from error

    def check(self) -> None:
        with self._lock:
            error = self._error
            statuses = self._statuses
            self._statuses = []
        if error is not None:
            raise RuntimeError("Audio playback failed") from error
        if statuses:
            warnings.warn(f"Audio playback reported: {', '.join(set(statuses))}",
                          RuntimeWarning, stacklevel=2)
        dropped = self.mixer.dropped_voices
        if dropped > self._reported_drops:
            warnings.warn(f"Audio voice limit dropped {dropped - self._reported_drops} sounds.",
                          RuntimeWarning, stacklevel=2)
            self._reported_drops = dropped

    def __enter__(self) -> "AudioPlayer":
        try:
            import sounddevice
        except ImportError as error:
            raise RuntimeError(
                'Live audio requires sounddevice; install animengine with the "audio" extra.'
            ) from error
        preferred = _preferred_output_device(sounddevice) if self.device is None else None

        def open_stream(device):
            stream = sounddevice.OutputStream(
                samplerate=self.mixer.sample_rate, channels=1, dtype="float32",
                blocksize=self.blocksize, latency="low", callback=self._callback, device=device,
            )
            try:
                stream.start()
            except Exception:
                stream.close()
                raise
            return stream

        try:
            self._stream = open_stream(preferred if preferred is not None else self.device)
        except sounddevice.PortAudioError as error:
            if preferred is None:
                raise
            warnings.warn(f"WASAPI output unavailable ({error}); using the default audio device.",
                          RuntimeWarning, stacklevel=2)
            self._stream = open_stream(None)
        return self

    def __exit__(self, exc_type, *_) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
        if exc_type is None:
            self.check()
