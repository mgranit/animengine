import imageio_ffmpeg
import numpy as np
import pytest

from animengine.anim import VideoWriter


def test_writes_all_frames(tmp_path):
    path = tmp_path / "clip.mp4"
    with VideoWriter(path, (64, 48), fps=10) as writer:
        for i in range(10):
            writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))

    frames, _ = imageio_ffmpeg.count_frames_and_secs(str(path))
    assert frames == 10


def test_rejects_odd_dimensions(tmp_path):
    with pytest.raises(ValueError, match="even"):
        VideoWriter(tmp_path / "bad.mp4", (63, 48), fps=10)


def test_rejects_wrong_frame_shape(tmp_path):
    with pytest.raises(ValueError):
        with VideoWriter(tmp_path / "bad.mp4", (64, 48), fps=10) as writer:
            writer.write(np.zeros((64, 48, 3), dtype=np.uint8))
