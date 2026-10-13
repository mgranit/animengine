import numpy as np
import pytest

from animengine import Cube, Lines, Node, Points, Scene
from animengine.render import to_rgba


def test_child_inherits_parent_transform():
    parent = Node(position=(1, 0, 0))
    child = parent.add(Cube(position=(0, 2, 0)))
    (item,) = list(parent.draw_items())
    assert item.mesh is child.mesh
    assert np.allclose(item.model[:3, 3], [1, 2, 0])


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


def test_mesh_rejects_bad_shapes():
    with pytest.raises(ValueError):
        Lines().set_points(np.zeros((4, 5)))


def test_to_rgba():
    assert np.allclose(to_rgba("#ff0000"), [1, 0, 0, 1])
    assert np.allclose(to_rgba("#f008"), [1, 0, 0, 0x88 / 255])
    assert np.allclose(to_rgba((0.1, 0.2, 0.3)), [0.1, 0.2, 0.3, 1.0])
    with pytest.raises(ValueError):
        to_rgba("red")
