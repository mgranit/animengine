import numpy as np
import pytest

from animengine import Cube, Lines, Node, PerspectiveCamera, Points, Scene
from animengine.linalg import Frame
from animengine.render import to_rgba


def test_child_inherits_parent_transform():
    parent = Node(position=(1, 0, 0))
    child = parent.add(Cube(position=(0, 2, 0)))
    (item,) = list(parent.draw_items())
    assert item.mesh is child.mesh
    assert np.allclose(item.model[:3, 3], [1, 2, 0])


def test_node_pose_is_a_frame_and_position_rotation_are_views():
    node = Node(position=(1, 2, 3), rotation=(10, 20, 30))
    assert isinstance(node.frame, Frame)
    assert np.allclose(node.frame.origin, [1, 2, 3])
    assert np.allclose(node.rotation, [10, 20, 30])
    node.position = (4, 5, 6)
    node.rotation = (0, 0, 90)
    assert np.allclose(node.frame.origin, [4, 5, 6])
    assert np.allclose(node.frame.x_axis, [0, 1, 0])
    with pytest.raises(ValueError):
        node.position[0] = 1.0


def test_node_local_matrix_is_frame_times_scale():
    node = Node(position=(1, 2, 3), rotation=(10, 20, 30), scale=(1, 2, 3))
    expected = node.frame.matrix() @ np.diag([1.0, 2.0, 3.0, 1.0])
    assert np.allclose(node.local_matrix, expected)


def test_node_accepts_frame_but_not_frame_and_position():
    frame = Frame.from_euler((0, 0, 90), origin=(1, 0, 0))
    assert Cube(frame=frame).frame is frame
    with pytest.raises(ValueError):
        Node(frame=frame, position=(0, 0, 0))


def test_parent_links_and_reparenting_rules():
    parent, child = Node(), Node()
    parent.add(child)
    assert child.parent is parent
    with pytest.raises(ValueError):
        Node().add(child)
    with pytest.raises(ValueError):
        child.add(parent)
    with pytest.raises(ValueError):
        parent.add(parent)
    parent.remove(child)
    assert child.parent is None


def test_world_frame_composes_parent_chain_and_matches_draw_matrix():
    root = Node(frame=Frame.from_euler((0, 0, 90), origin=(1, 0, 0)))
    child = root.add(Cube(position=(0, 2, 0), rotation=(0, 0, 90)))
    (item,) = list(root.draw_items())
    world = child.world_frame
    assert np.allclose(world.origin, [-1, 0, 0])
    assert np.allclose(world.x_axis, [-1, 0, 0])
    assert np.allclose(world.matrix(), child.world_matrix)
    assert np.allclose(item.model, child.world_matrix)


def test_world_frame_ignores_uniform_scale_of_ancestors():
    parent = Node(position=(1, 0, 0), scale=2.0)
    child = parent.add(Node(position=(1, 0, 0)))
    world = child.world_frame
    assert np.allclose(world.origin, [3, 0, 0])
    assert np.allclose(world.basis, np.eye(3))


def test_camera_mounted_on_node_follows_it():
    bob = Node(position=(0, 0, 0))
    camera = PerspectiveCamera(position=(0, 0, 0), target=(0, 1, 0))
    camera.parent = bob
    before = camera.view_matrix
    bob.position = (5, 0, 0)
    assert not np.allclose(camera.view_matrix, before)
    assert np.allclose(camera.world_frame.origin, [5, 0, 0])
    ndc = camera.projection_matrix(1.0) @ camera.view_matrix @ [5, 1, 0, 1]
    assert np.allclose(ndc[:2] / ndc[3], [0, 0], atol=1e-6)


def test_invisible_nodes_are_not_drawn():
    scene = Scene()
    cube = scene.add(Cube())
    cube.visible = False
    assert list(scene.root.draw_items()) == []


def test_scale_accepts_scalar_and_vector():
    node = Node()
    node.scale = 2.0
    assert np.array_equal(node.scale, [2, 2, 2])
    node.scale = (1, 2, 3)
    assert np.array_equal(node.scale, [1, 2, 3])


def test_cubes_share_one_mesh():
    assert Cube().mesh is Cube().mesh


def test_set_points_bumps_only_the_positions_version():
    points = Points()
    seen = {"positions": points.mesh.version("positions")}
    assert points.mesh.changed_since(seen) == []
    points.set_points([[0, 0], [1, 1]])
    assert points.mesh.changed_since(seen) == ["positions"]
    assert points.mesh.positions.dtype == np.float32


def test_mesh_attribute_setters_track_indices():
    mesh = Lines().mesh
    seen = {"positions": mesh.version("positions")}
    mesh.set_indices([0, 1])

    assert mesh.changed_since(seen) == ["indices"]
    assert mesh.indices.dtype == np.uint32
    assert mesh.indices.shape == (2,)


def test_mesh_rejects_bad_shapes():
    with pytest.raises(ValueError):
        Lines().set_points(np.zeros((4, 5)))


def test_to_rgba():
    assert np.allclose(to_rgba("#ff0000"), [1, 0, 0, 1])
    assert np.allclose(to_rgba("#f008"), [1, 0, 0, 0x88 / 255])
    assert np.allclose(to_rgba((0.1, 0.2, 0.3)), [0.1, 0.2, 0.3, 1.0])
    with pytest.raises(ValueError):
        to_rgba("red")
