import numpy as np
import pytest

from animengine.linalg import (
    Frame,
    get_rotation_matrix_x,
    get_rotation_matrix_y,
    get_rotation_matrix_z,
)


def test_default_frame_is_identity():
    frame = Frame()
    assert np.array_equal(frame.origin, [0, 0, 0])
    assert np.array_equal(frame.matrix(), np.eye(4))


def test_rejects_non_orthonormal_and_left_handed_bases():
    with pytest.raises(ValueError):
        Frame(basis=np.diag([2.0, 1.0, 1.0]))
    with pytest.raises(ValueError):
        Frame(basis=np.diag([1.0, 1.0, -1.0]))
    with pytest.raises(ValueError):
        Frame(origin=(0, 0))


def test_frame_is_immutable_and_copies_input():
    origin = np.array([1.0, 2.0, 3.0])
    frame = Frame(origin)
    origin[0] = 99.0
    assert frame.origin[0] == 1.0
    with pytest.raises(ValueError):
        frame.origin[0] = 5.0
    with pytest.raises(ValueError):
        frame.basis[0, 0] = 5.0


def test_axes_are_basis_columns():
    frame = Frame.from_axis_angle(90, [0, 0, 1])
    assert np.allclose(frame.x_axis, [0, 1, 0])
    assert np.allclose(frame.y_axis, [-1, 0, 0])
    assert np.allclose(frame.z_axis, [0, 0, 1])


def test_transform_point_and_vector():
    frame = Frame.from_axis_angle(90, [0, 0, 1], origin=(1, 2, 3))
    assert np.allclose(frame.transform_point([1, 0, 0]), [1, 3, 3])
    assert np.allclose(frame.transform_vector([1, 0, 0]), [0, 1, 0])
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert np.allclose(frame.transform_point(points), [[1, 3, 3], [0, 2, 3]])


def test_matrix_agrees_with_transform_point_and_inverse_round_trips():
    frame = Frame.from_euler((10, 20, 30), origin=(1, -2, 3))
    point = np.array([0.3, -0.7, 1.1])
    assert np.allclose((frame.matrix() @ np.append(point, 1.0))[:3], frame.transform_point(point))
    assert np.allclose(frame.matrix() @ frame.inverse_matrix(), np.eye(4))
    assert np.allclose(frame.to_local_point(frame.transform_point(point)), point)
    assert np.allclose(frame.to_local_vector(frame.transform_vector(point)), point)
    assert np.allclose(frame.inverse().matrix(), frame.inverse_matrix())


def test_relative_to_sibling_frames():
    a = Frame.from_euler((0, 0, 90), origin=(1, 0, 0))
    b = Frame.from_euler((10, 20, 30), origin=(0, 2, 1))
    relative = b.relative_to(a)
    assert np.allclose((a @ relative).matrix(), b.matrix())
    assert np.allclose(a.relative_to(a).matrix(), np.eye(4))


def test_transform_point_and_vector_between_siblings():
    a = Frame.from_euler((0, 0, 90), origin=(1, 0, 0))
    b = Frame.from_euler((10, 20, 30), origin=(0, 2, 1))
    point = np.array([0.3, -0.7, 1.1])
    world = a.transform_point(point)
    assert np.allclose(a.transform_point_to(b, point), b.to_local_point(world))
    expected = b.relative_to(a).inverse().transform_point(point)
    assert np.allclose(a.transform_point_to(b, point), expected)
    assert np.allclose(b.transform_point_to(b, point), point)
    assert np.allclose(
        a.transform_vector_to(b, point), b.to_local_vector(a.transform_vector(point))
    )


def test_from_matrix_orthonormalize_discards_scale_and_reflection():
    rotation = Frame.from_euler((10, 20, 30), origin=(1, 2, 3))
    scaled = rotation.matrix() @ np.diag([2.0, 2.0, 2.0, 1.0])
    assert np.allclose(Frame.from_matrix(scaled, orthonormalize=True).matrix(), rotation.matrix())
    mirrored = rotation.matrix() @ np.diag([1.0, 1.0, -1.0, 1.0])
    fixed = Frame.from_matrix(mirrored, orthonormalize=True)
    assert np.linalg.det(fixed.basis) == pytest.approx(1.0)
    assert np.allclose(fixed.origin, [1, 2, 3])


@pytest.mark.parametrize("angles", [
    (10, 20, 30), (-170, 45, 120), (0, 0, 0), (33, -89, 12), (90, 0, -90),
    (25, 90, 0), (25, -90, 0), (25, 90, 40), (-60, -90, 70),
])
def test_to_euler_round_trips_to_the_same_rotation(angles):
    frame = Frame.from_euler(angles)
    again = Frame.from_euler(frame.to_euler())
    assert np.allclose(again.basis, frame.basis, atol=1e-9)


def test_to_euler_recovers_ordinary_angles():
    assert np.allclose(Frame.from_euler((10, 20, 30)).to_euler(), [10, 20, 30])


def test_matrix_dtype():
    assert Frame().matrix(np.float32).dtype == np.float32
    assert Frame().inverse_matrix(np.float32).dtype == np.float32


def test_compose_matches_matrix_product():
    parent = Frame.from_euler((0, 0, 90), origin=(1, 0, 0))
    child = Frame.from_euler((45, 0, 0), origin=(0, 2, 0))
    assert np.allclose((parent @ child).matrix(), parent.matrix() @ child.matrix())
    assert np.allclose(parent.compose(child).origin, [-1, 0, 0])


def test_from_euler_matches_node_rotation_order():
    expected = (
        get_rotation_matrix_z(30) @ get_rotation_matrix_y(20) @ get_rotation_matrix_x(10)
    )
    assert np.allclose(Frame.from_euler((10, 20, 30)).matrix(), expected)


def test_from_matrix_round_trip_and_rejects_scale():
    frame = Frame.from_euler((10, 20, 30), origin=(1, 2, 3))
    restored = Frame.from_matrix(frame.matrix())
    assert np.allclose(restored.matrix(), frame.matrix())
    with pytest.raises(ValueError):
        Frame.from_matrix(np.diag([2.0, 2.0, 2.0, 1.0]))
    with pytest.raises(ValueError):
        Frame.from_matrix(np.eye(3))


def test_look_at_points_negative_z_at_target():
    frame = Frame.look_at((0, 0, 5), (0, 0, 0), (0, 1, 0))
    assert np.allclose(frame.z_axis, [0, 0, 1])
    assert np.allclose(frame.to_local_point([0, 0, 0]), [0, 0, -5])


def test_look_at_degenerate_up_falls_back():
    frame = Frame.look_at((0, 0, 5), (0, 0, 0), (0, 0, 1))
    assert np.allclose(frame.z_axis, [0, 0, 1])
    assert np.allclose(frame.basis.T @ frame.basis, np.eye(3))


def test_look_at_rejects_coincident_points():
    with pytest.raises(ValueError):
        Frame.look_at((1, 1, 1), (1, 1, 1), (0, 0, 1))


def test_orthonormalized_repairs_accumulated_drift():
    step = Frame.from_axis_angle(1.0, [1, 2, 3])
    frame = Frame()
    for _ in range(20000):
        frame = frame @ step
    fixed = frame.orthonormalized()
    assert np.allclose(fixed.basis.T @ fixed.basis, np.eye(3), atol=1e-12)
    assert np.linalg.det(fixed.basis) == pytest.approx(1.0)
    assert np.allclose(fixed.basis, frame.basis, atol=1e-6)
