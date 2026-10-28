import numpy as np
import pytest
from examples.bouncing_points import PHYSICS_DT, VIEW_HEIGHT, VIEW_WIDTH, build_animation

from animengine import OrthographicCamera, render_frame


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
