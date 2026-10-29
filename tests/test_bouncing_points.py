import sys

import numpy as np
import pytest
from examples import bouncing_points
from examples.bouncing_points import (
    PHYSICS_DT,
    VIEW_HEIGHT,
    VIEW_WIDTH,
    build_animation,
)

from animengine import OrthographicCamera, render_frame
from animengine.audio import AudioMixer


def test_seeded_example_and_fixed_step_preview_match_offline():
    scene, points, offline, offline_stepper = build_animation(count=50, seed=42)
    _, _, preview, preview_stepper = build_animation(count=50, seed=42)
    _, _, different, _ = build_animation(count=50, seed=43)
    assert np.array_equal(offline.state.positions, preview.state.positions)
    assert not np.array_equal(offline.state.positions, different.state.positions)
    assert isinstance(scene.camera, OrthographicCamera)
    assert scene.camera.width == VIEW_WIDTH
    assert scene.camera.height == VIEW_HEIGHT
    assert np.allclose(np.linalg.norm(offline.state.velocities, axis=1), 0.5)
    for _ in range(100):
        offline_stepper.advance(PHYSICS_DT)
    for elapsed in (0.03, 0.1, 0.04, 100 * PHYSICS_DT - 0.17):
        preview_stepper.advance(elapsed)
    assert np.allclose(offline.state.positions, preview.state.positions)
    assert np.allclose(offline.state.velocities, preview.state.velocities)
    points.set_points(offline.state.positions)
    assert points.mesh is not None
    assert np.allclose(points.mesh.positions, offline.state.positions)


@pytest.mark.gl
@pytest.mark.usefixtures("require_gl")
def test_example_draws_and_moves_points():
    scene, points, simulation, stepper = build_animation(count=30, seed=42)
    scene.size = (128, 72)
    first = render_frame(scene, samples=0)
    stepper.advance(0.5)
    points.set_points(simulation.state.positions)
    second = render_frame(scene, samples=0)
    assert first.max() > 0
    assert second.max() > 0
    assert not np.array_equal(first, second)


def test_example_schedules_sound_for_a_wall_collision():
    audio = AudioMixer()
    _, _, simulation, stepper = build_animation(count=1, audio=audio)
    simulation.state.positions[:] = [[VIEW_WIDTH / 2 - 0.01, 0]]
    simulation.state.velocities[:] = [[1, 0]]
    stepper.advance(4 * PHYSICS_DT)
    samples = audio.render(round(4 * PHYSICS_DT * audio.sample_rate))
    assert np.max(np.abs(samples)) > 0.01


def test_example_passes_preview_audio_options_and_displays_current_positions(monkeypatch):
    _, _, reference, reference_stepper = build_animation(count=1, seed=42)
    reference_stepper.advance(0.1)
    calls = []

    def fake_preview(scene, update, **kwargs):
        update(0.1, 0.1)
        calls.append((scene, kwargs))

    monkeypatch.setattr(bouncing_points, "preview", fake_preview)
    monkeypatch.setattr(
        sys, "argv",
        ["bouncing_points.py", "--preview", "--sound", "--count", "1", "--seed", "42",
         "--audio-delay", "0.05", "--audio-device", "10"],
    )
    bouncing_points.main()
    scene, kwargs = calls[0]
    assert kwargs["audio_delay"] == 0.05
    assert kwargs["audio_device"] == 10
    assert kwargs["audio"] is not None
    points = scene.root.children[0]
    assert points.mesh is not None
    assert np.allclose(points.mesh.positions, reference.state.positions)
