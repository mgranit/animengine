import glfw
import numpy as np

from animengine import OrthographicCamera, PerspectiveCamera
from animengine.linalg import Frame, look_at
from animengine.render import FlyController


def to_ndc(camera, aspect, point):
    clip = camera.projection_matrix(aspect) @ camera.view_matrix @ np.append(point, 1.0)
    return clip[:3] / clip[3]


class FakeInput:
    def __init__(self, *keys):
        self.keys = set(keys)

    def on_cursor(self, callback):
        pass

    def on_scroll(self, callback):
        pass

    def is_key_down(self, key):
        return key in self.keys


def test_view_matrix_matches_look_at():
    camera = PerspectiveCamera(position=(3, 2, 1), target=(1, 0, 0))
    assert np.allclose(camera.view_matrix, look_at((3, 2, 1), (1, 0, 0), (0, 0, 1)))
    assert camera.view_matrix.dtype == np.float32


def test_position_and_target_properties_round_trip():
    camera = PerspectiveCamera(position=(3, 2, 1), target=(1, 0, 0))
    assert np.allclose(camera.position, [3, 2, 1])
    assert np.allclose(camera.target, [1, 0, 0])
    assert np.allclose(camera.world_up, [0, 0, 1])


def test_setting_position_keeps_aiming_at_target():
    camera = PerspectiveCamera(position=(3, 0, 1), target=(0, 0, 0))
    camera.position = (0, 3, 1)
    assert np.allclose(camera.target, [0, 0, 0])
    assert np.allclose(to_ndc(camera, 1.0, [0, 0, 0])[:2], [0, 0], atol=1e-6)


def test_setting_target_keeps_position():
    camera = PerspectiveCamera(position=(3, 0, 1), target=(0, 0, 0))
    camera.target = (0, 5, 1)
    assert np.allclose(camera.position, [3, 0, 1])
    assert np.allclose(to_ndc(camera, 1.0, [0, 5, 1])[:2], [0, 0], atol=1e-6)


def test_assigning_frame_drives_view_matrix():
    camera = PerspectiveCamera()
    frame = Frame.from_euler((90, 0, 0), origin=(1, 2, 3))
    camera.frame = frame
    assert np.allclose(camera.position, [1, 2, 3])
    assert np.allclose(camera.view_matrix, frame.inverse_matrix(np.float32))


def test_fly_controller_moves_along_camera_axes():
    camera = PerspectiveCamera(position=(0, 0, 0), target=(1, 0, 0))
    controller = FlyController(camera, FakeInput(glfw.KEY_W, glfw.KEY_D), speed=1.0)
    controller.update(1.0)
    assert np.allclose(camera.position, [1, -1, 0])
    assert np.allclose(-camera.frame.z_axis, [1, 0, 0])
    assert np.allclose(camera.frame.y_axis, [0, 0, 1])


def test_fly_controller_vertical_motion_is_along_camera_up():
    camera = PerspectiveCamera(position=(0, 0, 0), target=(1, 0, 1))
    controller = FlyController(camera, FakeInput(glfw.KEY_SPACE), speed=1.0)
    controller.update(1.0)
    assert np.allclose(camera.position, np.array([-1, 0, 1]) / np.sqrt(2))


def test_fly_controller_is_continuous_through_straight_up_and_down():
    camera = PerspectiveCamera(position=(0, 0, 0), target=(1, 0, 0))
    controller = FlyController(camera, FakeInput())
    for polar in (89.0, 90.0, -89.0, -90.0):
        controller.polar = polar
        controller.update(0.0)
        frame = camera.frame
        assert np.allclose(-frame.z_axis, controller._front())
        assert np.allclose(frame.x_axis, [0, -1, 0])
        assert np.allclose(frame.y_axis, [-np.sin(np.radians(polar)), 0, np.cos(np.radians(polar))])


def test_perspective_camera_puts_target_at_screen_center():
    camera = PerspectiveCamera(position=(3, 3, 3), target=(1, 0, 0))
    assert np.allclose(to_ndc(camera, 16 / 9, [1, 0, 0])[:2], [0, 0], atol=1e-6)


def test_fixed_orthographic_camera_keeps_projection_and_fits_viewport():
    camera = OrthographicCamera(width=8.0, height=4.0)
    assert np.allclose(camera.projection_matrix(1.0), camera.projection_matrix(2.0))
    assert np.allclose(to_ndc(camera, 2.0, [4.0, 2.0, 0.0])[:2], [1, 1], atol=1e-6)
    assert np.allclose(to_ndc(camera, 2.0, [-4.0, -2.0, 0.0])[:2], [-1, -1], atol=1e-6)
    assert camera.viewport_rect((100, 100)) == (0, 25, 100, 50)
    assert camera.viewport_rect((300, 100)) == (50, 0, 200, 100)
