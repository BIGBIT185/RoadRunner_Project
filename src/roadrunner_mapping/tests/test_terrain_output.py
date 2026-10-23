"""Exercise real ROS GridMap messages on an asymmetric metric terrain."""
import ast
from pathlib import Path

import numpy as np
import pytest
import rospy
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout

from roadrunner_mapping.geometry.coordinates import body_xy_to_grid
from roadrunner_mapping.geometry.schema import ONLINE_MAP_SPEC, MapSpec
from roadrunner_mapping.online.online_fusion import grid_map_column_major_data


def publisher():
    # Load the production serializer without constructing its GPU/ROS node.
    path = Path(__file__).resolve().parents[1] / "scripts/mapping_node.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "probability_grid_map_message")
    namespace = dict(np=np, GridMap=GridMap, ONLINE_MAP_SPEC=ONLINE_MAP_SPEC,
                     Float32MultiArray=Float32MultiArray,
                     MultiArrayDimension=MultiArrayDimension,
                     MultiArrayLayout=MultiArrayLayout,
                     grid_map_column_major_data=grid_map_column_major_data)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["probability_grid_map_message"]


def test_terrain_retains_metric_height_probability_and_cell_direction():
    spec = MapSpec(-1., 1., -1.5, 1.5, 1.)
    probability = np.array([[.1, .2, np.nan], [.4, .5, .6]], np.float32)
    elevation = np.array([[-2., -1., np.nan], [1., 2., 3.]], np.float32)
    original = probability.copy()
    message = publisher()({"final_probability": probability}, rospy.Time(12, 34),
                          "base_link", map_spec=spec, elevation=elevation)
    assert message.info.header.frame_id == "base_link"
    assert message.info.header.stamp == rospy.Time(12, 34)
    assert (message.info.length_x, message.info.length_y, message.info.resolution) == (2., 3., 1.)
    assert message.info.pose.orientation.w == 1.
    assert message.layers == ["final_probability", "elevation"]
    assert "elevation" in message.basic_layers
    data = {name: np.array(layer.data).reshape(3, 2).T
            for name, layer in zip(message.layers, message.data)}
    np.testing.assert_allclose(data["elevation"], elevation, equal_nan=True)
    np.testing.assert_allclose(data["final_probability"], original, equal_nan=True)
    np.testing.assert_allclose(probability, original, equal_nan=True)
    # (+x,+y) is the first cell; (-x,-y) is the last cell in GridMap.
    rows, cols, inside = body_xy_to_grid(np.array([[.5, 1.], [-.5, -1.]]), spec)
    assert inside.all()
    assert data["elevation"][rows, cols].tolist() == [-2., 3.]
    assert len(message.data[0].layout.dim) == 2
    assert message.data[0].layout.dim[0].size == 3
    assert message.data[0].layout.dim[1].size == 2
    assert message.outer_start_index == message.inner_start_index == 0


def test_elevation_shape_must_match_probability_geometry():
    spec = MapSpec(-1., 1., -1.5, 1.5, 1.)
    with pytest.raises(ValueError, match="elevation.*shape"):
        publisher()({"final_probability": np.ones((2, 3), np.float32)},
                    rospy.Time(1), "base_link", map_spec=spec,
                    elevation=np.ones((3, 2), np.float32))


def test_unobserved_height_placeholder_is_not_published_as_real_terrain():
    spec = MapSpec(-1., 1., -1.5, 1.5, 1.)
    geometry_probability = np.array([[1., np.nan, 1.], [1., 1., 1.]], np.float32)
    message = publisher()({"final_probability": np.ones((2, 3), np.float32),
                           "geometry_probability": geometry_probability},
                          rospy.Time(1), "base_link", map_spec=spec,
                          elevation=np.zeros((2, 3), np.float32))
    data = np.array(message.data[message.layers.index("elevation")].data).reshape(3, 2).T
    assert np.isnan(data[0, 1])
    assert data[0, 0] == 0.  # Observed zero-metre terrain stays valid.
