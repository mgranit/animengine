import numpy as np

from animengine import OrthographicCamera, PerspectiveCamera


def to_ndc(camera, aspect, point):
    clip = camera.projection_matrix(aspect) @ camera.view_matrix @ np.append(point, 1.0)
    return clip[:3] / clip[3]


def test_perspective_camera_puts_target_at_screen_center():
    camera = PerspectiveCamera(position=(3, 3, 3), target=(1, 0, 0))
    assert np.allclose(to_ndc(camera, 16 / 9, [1, 0, 0])[:2], [0, 0], atol=1e-6)


def test_orthographic_camera_visible_height_and_aspect():
    camera = OrthographicCamera(height=4.0)
    assert np.allclose(to_ndc(camera, 2.0, [4.0, 2.0, 0.0])[:2], [1, 1], atol=1e-6)
    assert np.allclose(to_ndc(camera, 2.0, [-4.0, -2.0, 0.0])[:2], [-1, -1], atol=1e-6)
