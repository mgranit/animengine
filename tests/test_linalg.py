import numpy as np
import pytest

from animengine.linalg import (
    get_rotation_matrix,
    get_rotation_matrix_z,
    get_translation_matrix,
    look_at,
    orthographic,
    perspective,
)


def apply(matrix, point):
    return (matrix @ np.append(point, 1.0))[:3]


def test_translation():
    assert np.allclose(apply(get_translation_matrix(1, 2, 3), [1, 1, 1]), [2, 3, 4])


def test_rotation_z_quarter_turn():
    assert np.allclose(apply(get_rotation_matrix_z(90), [1, 0, 0]), [0, 1, 0])


def test_axis_rotation_matches_z_rotation_and_keeps_axis():
    axis = np.array([0.0, 0.0, 2.0])
    matrix = get_rotation_matrix(37, axis)
    assert matrix.shape == (4, 4)
    assert np.allclose(matrix, get_rotation_matrix_z(37))
    assert np.array_equal(axis, [0.0, 0.0, 2.0])


def test_look_at_moves_eye_to_origin_and_target_to_negative_z():
    view = look_at([0, 0, 5], [0, 0, 0], [0, 1, 0])
    assert np.allclose(apply(view, [0, 0, 5]), [0, 0, 0])
    assert np.allclose(apply(view, [0, 0, 0]), [0, 0, -5])


def test_look_at_with_up_parallel_to_view_direction_is_finite():
    view = look_at([0, 0, 5], [0, 0, 0], [0, 0, 1])
    assert np.all(np.isfinite(view))
    assert np.allclose(apply(view, [0, 0, 0]), [0, 0, -5])


def test_look_at_rejects_coincident_eye_and_target():
    with pytest.raises(ValueError):
        look_at([1, 2, 3], [1, 2, 3], [0, 0, 1])


def test_perspective_maps_near_and_far_to_ndc_limits():
    proj = perspective(60, 1.5, 0.1, 50.0, dtype=np.float64)
    near = proj @ [0, 0, -0.1, 1]
    far = proj @ [0, 0, -50.0, 1]
    assert near[2] / near[3] == pytest.approx(-1.0)
    assert far[2] / far[3] == pytest.approx(1.0)


def test_orthographic_maps_box_corners_to_ndc_corners():
    proj = orthographic(-2, 2, -1, 1, 0.1, 10, dtype=np.float64)
    assert np.allclose(apply(proj, [2, 1, -0.1]), [1, 1, -1])
    assert np.allclose(apply(proj, [-2, -1, -10]), [-1, -1, 1])
