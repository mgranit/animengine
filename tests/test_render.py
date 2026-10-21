import imageio_ffmpeg
import numpy as np
import pytest

from animengine import (
    Cube,
    OrthographicCamera,
    Points,
    Scene,
    Texture,
    TexturedCube,
    render_frame,
    render_video,
)

pytestmark = [pytest.mark.gl, pytest.mark.usefixtures("require_gl")]


def test_background_fills_frame():
    frame = render_frame(Scene(size=(64, 48), background="#ff0000"))
    assert frame.shape == (48, 64, 3)
    assert frame.dtype == np.uint8
    assert np.all(frame == [255, 0, 0])


def test_fixed_orthographic_camera_fits_viewport_with_bar_color():
    scene = Scene(
        size=(80, 40),
        background="#00ff00",
        bar_color="#ff0000",
        camera=OrthographicCamera(width=2.0, height=2.0),
    )
    frame = render_frame(scene, samples=0)
    assert np.all(frame[20, 0] == [255, 0, 0])
    assert np.all(frame[20, 20] == [0, 255, 0])


def test_fixed_orthographic_camera_defaults_bar_color_to_background():
    scene = Scene(
        size=(80, 40),
        background="#123456",
        camera=OrthographicCamera(width=2.0, height=2.0),
    )
    frame = render_frame(scene, samples=0)
    assert np.all(frame == [0x12, 0x34, 0x56])


def test_cube_is_drawn_at_screen_center():
    scene = Scene(size=(64, 64))
    scene.add(Cube())
    assert np.all(render_frame(scene)[32, 32] == 255)


def test_image_is_top_down():
    scene = Scene(size=(64, 64), camera=OrthographicCamera(height=2.0))
    scene.add(Points([[0.0, 0.8, 0.0]], size=9.0))
    frame = render_frame(scene, samples=0)
    assert frame[:16].max() == 255
    assert frame[48:].max() == 0


def test_textured_cube_samples_its_texture():
    texture = Texture(np.full((4, 4, 3), [0, 255, 0], dtype=np.uint8), mipmaps=False)
    scene = Scene(size=(64, 64))
    scene.add(TexturedCube(texture))
    assert np.all(render_frame(scene)[32, 32] == [0, 255, 0])


def test_rendering_is_deterministic():
    scene = Scene(size=(64, 64))
    scene.add(Cube(wireframe=True))
    assert np.array_equal(render_frame(scene), render_frame(scene))


def test_render_video_calls_update_with_fixed_steps_and_writes_frames(tmp_path):
    scene = Scene(size=(64, 48))
    calls = []
    out = tmp_path / "out.mp4"

    render_video(scene, lambda t, dt: calls.append((t, dt)), duration=0.5, fps=10, out=out,
                 steps_per_frame=2, progress=False)

    assert calls[0] == (0.0, 0.0)
    assert len(calls) == 1 + 2 * 4
    assert calls[-1][0] == pytest.approx(0.4)
    assert all(dt == pytest.approx(0.05) for _, dt in calls[1:])
    assert imageio_ffmpeg.count_frames_and_secs(str(out))[0] == 5
