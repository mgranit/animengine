import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
import numpy as np


class VideoWriter:
    """Encodes top-down RGB uint8 frames to H.264 by piping them to the bundled ffmpeg."""

    def __init__(self, path, size: tuple[int, int], fps: float, crf: int = 17) -> None:
        width, height = size
        if width % 2 or height % 2:
            raise ValueError(f"yuv420p needs even dimensions, got {width}x{height}")

        self.size = size
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._stderr = tempfile.TemporaryFile()
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
            "-pix_fmt", "yuv420p", str(self.path),
        ]
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=self._stderr)

    def _ffmpeg_errors(self) -> str:
        self._stderr.seek(0)
        return self._stderr.read().decode(errors="replace")

    def write(self, frame: np.ndarray) -> None:
        width, height = self.size
        if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
            raise ValueError(f"Expected uint8 frame of shape {(height, width, 3)}, "
                             f"got {frame.dtype} {frame.shape}")
        try:
            self._process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            raise RuntimeError(f"ffmpeg stopped unexpectedly:\n{self._ffmpeg_errors()}") from None

    def close(self) -> None:
        self._process.stdin.close()
        returncode = self._process.wait()
        errors = self._ffmpeg_errors()
        self._stderr.close()
        if returncode != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {returncode}:\n{errors}")

    def abort(self) -> None:
        self._process.kill()
        self._process.wait()
        try:
            self._process.stdin.close()
        except OSError:
            pass  # pipe already broken
        self._stderr.close()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, *_) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()
