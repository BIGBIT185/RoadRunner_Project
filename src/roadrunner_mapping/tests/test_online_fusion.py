from types import SimpleNamespace

import numpy as np
import pytest

from roadrunner_mapping.online import online_fusion
from roadrunner_mapping.online.online_fusion import (
    CompactSemanticFrame,
    RecentFrameWindow,
    map_to_occupancy_order,
    rasterize_semantic_window,
)
from roadrunner_mapping.geometry.schema import MapSpec


def test_recent_window_strictly_expires_old_frames():
    window = RecentFrameWindow(60.0)
    assert not window.append(SimpleNamespace(timestamp=1.0))
    assert not window.append(SimpleNamespace(timestamp=61.0))
    expired = window.append(SimpleNamespace(timestamp=61.001))
    assert [frame.timestamp for frame in expired] == [1.0]
    assert [frame.timestamp for frame in window] == [61.0, 61.001]


def test_semantic_window_rasterizes_risk_in_current_body():
    frame = CompactSemanticFrame(
        timestamp=10.0,
        xyz_world=np.asarray([[0.5, 0.5, 0.0]], dtype=np.float32),
        labels=np.asarray([4], dtype=np.uint8),
        confidence=np.asarray([1.0], dtype=np.float32),
        valid=np.asarray([True]),
        dynamic=np.asarray([False]),
    )
    result = rasterize_semantic_window(
        [frame],
        target_timestamp=10.0,
        target_world_from_body=np.eye(4),
        map_spec=MapSpec(-1.0, 1.0, -1.0, 1.0, 1.0),
        semantic_risk={4: 1.0},
    )
    assert result["semantic_hindsight_valid_mask"][0, 0]
    assert result["semantic_hindsight_target"][0, 0] == 1.0


def test_semantic_window_accumulates_both_goose_grass_classes():
    frame = CompactSemanticFrame(
        timestamp=10.0,
        xyz_world=np.asarray(
            [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.5, 0.5, 0.0]],
            dtype=np.float32,
        ),
        labels=np.asarray([50, 51, 31], dtype=np.uint8),
        confidence=np.asarray([1.0, 0.5, 0.5], dtype=np.float32),
        valid=np.asarray([True, True, True]),
        dynamic=np.asarray([False, False, False]),
    )

    result = rasterize_semantic_window(
        [frame],
        target_timestamp=10.0,
        target_world_from_body=np.eye(4),
        map_spec=MapSpec(-1.0, 1.0, -1.0, 1.0, 1.0),
        semantic_risk={31: 0.3, 50: 0.25, 51: 0.55},
        grass_class_ids={50, 51},
    )

    assert result["grass_probability"][0, 0] == 0.75


def test_incremental_semantic_grid_aggregates_same_cell_before_history_update():
    grid = online_fusion.IncrementalSemanticGrid(
        map_spec=MapSpec(-1.0, 1.0, -1.0, 1.0, 1.0),
        semantic_risk={1: 0.0, 2: 1.0},
        grass_class_ids=set(),
        decay_tau_sec=30.0,
        stale_sec=90.0,
    )

    result = grid.advance(
        timestamp=10.0,
        world_from_body=np.eye(4),
        xyz_world=np.asarray(
            [[0.25, 0.25, 0.0], [0.40, 0.40, 0.0], [0.49, 0.49, 0.0]],
            dtype=np.float32,
        ),
        labels=np.asarray([1, 2, 2], dtype=np.uint8),
        confidence=np.asarray([1.0, 1.0, 0.5], dtype=np.float32),
        valid=np.ones(3, dtype=bool),
        dynamic=np.zeros(3, dtype=bool),
    )

    assert result["semantic_input_points"] == 3
    assert result["semantic_aggregated_cells"] == 1
    assert result["semantic_hindsight_observation_count"][0, 0] == 3
    assert result["semantic_hindsight_target"][0, 0] == np.float32(0.6)


def test_incremental_semantic_grid_reports_conservative_cvar():
    grid = online_fusion.IncrementalSemanticGrid(
        map_spec=MapSpec(-1.0, 1.0, -1.0, 1.0, 1.0),
        semantic_risk={1: 0.1, 2: 0.35},
        grass_class_ids=set(),
        decay_tau_sec=30.0,
        stale_sec=90.0,
        risk_histogram_bins=4,
        risk_cvar_alpha=0.5,
    )
    advance_semantic_grid(grid, timestamp=0.0, labels=(1,))
    result = advance_semantic_grid(grid, timestamp=0.1, labels=(2,))

    assert result["semantic_hindsight_target"][0, 0] < 0.35
    assert result["semantic_hindsight_risk_std"][0, 0] > 0.0
    assert result["semantic_hindsight_risk_cvar"][0, 0] == pytest.approx(0.5)
    assert result["semantic_hindsight_risk_distribution_valid"][0, 0]


def semantic_grid(
    risks,
    *,
    grass_class_ids=(),
):
    return online_fusion.IncrementalSemanticGrid(
        map_spec=MapSpec(-1.0, 1.0, -1.0, 1.0, 1.0),
        semantic_risk=risks,
        grass_class_ids=grass_class_ids,
        decay_tau_sec=30.0,
        stale_sec=90.0,
    )


def advance_semantic_grid(
    grid,
    *,
    timestamp,
    pose=None,
    xyz=((0.25, 0.25, 0.0),),
    labels=(1,),
    confidence=(1.0,),
    dynamic=(False,),
):
    points = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    return grid.advance(
        timestamp=timestamp,
        world_from_body=np.eye(4) if pose is None else pose,
        xyz_world=points,
        labels=np.asarray(labels, dtype=np.uint8),
        confidence=np.asarray(confidence, dtype=np.float32),
        valid=np.ones(points.shape[0], dtype=bool),
        dynamic=np.asarray(dynamic, dtype=bool),
    )


def test_incremental_semantic_grid_decays_previous_frame_before_accumulating():
    grid = semantic_grid({1: 0.2, 2: 0.4})
    advance_semantic_grid(grid, timestamp=0.0, labels=(1,))
    result = advance_semantic_grid(grid, timestamp=30.0, labels=(2,))

    decay = np.exp(-1.0)
    expected = (0.2 * decay + 0.4) / (decay + 1.0)
    assert result["semantic_hindsight_target"][0, 0] == pytest.approx(expected)


def test_incremental_semantic_grid_accumulates_grass_weight():
    grid = semantic_grid(
        {31: 0.3, 50: 0.25, 51: 0.55},
        grass_class_ids={50, 51},
    )
    result = advance_semantic_grid(
        grid,
        timestamp=10.0,
        xyz=((0.25, 0.25, 0.0),) * 3,
        labels=(50, 51, 31),
        confidence=(1.0, 0.5, 0.5),
        dynamic=(False, False, False),
    )

    assert result["grass_probability"][0, 0] == pytest.approx(0.75)


def test_incremental_semantic_grid_uses_latest_risk_after_conflict():
    grid = semantic_grid({1: 0.0, 2: 1.0})
    advance_semantic_grid(grid, timestamp=0.0, labels=(1,))
    result = advance_semantic_grid(grid, timestamp=1.0, labels=(2,))

    assert result["semantic_hindsight_target"][0, 0] == 1.0


def test_incremental_semantic_grid_uses_latest_risk_after_dynamic_observation():
    grid = semantic_grid({1: 0.2, 2: 0.3})
    advance_semantic_grid(
        grid,
        timestamp=0.0,
        labels=(1,),
        dynamic=(True,),
    )
    result = advance_semantic_grid(grid, timestamp=1.0, labels=(2,))

    assert result["semantic_hindsight_target"][0, 0] == pytest.approx(0.3)


def test_incremental_semantic_grid_samples_fixed_world_cell_after_translation():
    grid = semantic_grid({1: 0.8})
    advance_semantic_grid(grid, timestamp=0.0)
    translated = np.eye(4)
    translated[0, 3] = 1.0
    result = advance_semantic_grid(
        grid,
        timestamp=1.0,
        pose=translated,
        xyz=(),
        labels=(),
        confidence=(),
        dynamic=(),
    )

    assert result["semantic_hindsight_valid_mask"][1, 0]
    assert result["semantic_hindsight_target"][1, 0] == pytest.approx(0.8)


def test_incremental_semantic_grid_samples_fixed_world_cell_after_rotation():
    grid = semantic_grid({1: 0.8})
    advance_semantic_grid(
        grid,
        timestamp=0.0,
        xyz=((0.25, 0.25, 0.0),),
    )
    rotated = np.eye(4)
    rotated[:2, :2] = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    result = advance_semantic_grid(
        grid,
        timestamp=1.0,
        pose=rotated,
        xyz=(),
        labels=(),
        confidence=(),
        dynamic=(),
    )

    assert result["semantic_hindsight_valid_mask"][0, 1]
    assert result["semantic_hindsight_target"][0, 1] == pytest.approx(0.8)


def test_incremental_semantic_grid_clears_stale_cell_before_reobservation():
    grid = semantic_grid({1: 0.2, 2: 0.4})
    advance_semantic_grid(grid, timestamp=0.0, labels=(1,))
    stale = advance_semantic_grid(
        grid,
        timestamp=91.0,
        xyz=(),
        labels=(),
        confidence=(),
        dynamic=(),
    )
    assert not stale["semantic_hindsight_valid_mask"].any()

    result = advance_semantic_grid(grid, timestamp=92.0, labels=(2,))

    assert result["semantic_hindsight_target"][0, 0] == pytest.approx(0.4)
    assert result["semantic_hindsight_observation_count"][0, 0] == 1


def test_incremental_semantic_grid_sampling_does_not_advance_observation_time():
    grid = semantic_grid({1: 0.2, 2: 0.4})
    grid.update(
        timestamp=10.0,
        world_from_body=np.eye(4),
        xyz_world=np.asarray([[0.25, 0.25, 0.0]], dtype=np.float32),
        labels=np.asarray([1], dtype=np.uint8),
        confidence=np.asarray([1.0], dtype=np.float32),
        valid=np.asarray([True]),
        dynamic=np.asarray([False]),
    )
    sampled = grid.sample(
        timestamp=20.0,
        world_from_body=np.eye(4),
    )
    assert sampled["semantic_hindsight_valid_mask"][0, 0]

    grid.update(
        timestamp=11.0,
        world_from_body=np.eye(4),
        xyz_world=np.asarray([[0.25, 0.25, 0.0]], dtype=np.float32),
        labels=np.asarray([2], dtype=np.uint8),
        confidence=np.asarray([1.0], dtype=np.float32),
        valid=np.asarray([True]),
        dynamic=np.asarray([False]),
    )
    result = grid.sample(timestamp=11.0, world_from_body=np.eye(4))

    assert result["semantic_hindsight_observation_count"][0, 0] >= 2


def test_map_to_occupancy_order_preserves_xy_coordinates():
    road_runner = np.asarray([[1, 2], [3, 4]])
    assert np.array_equal(
        map_to_occupancy_order(road_runner),
        np.asarray([[4, 2], [3, 1]]),
    )


def test_probability_elevation_layer_clips_values_and_marks_unknown_as_nan():
    map_spec = MapSpec(-1.0, 1.0, -1.0, 1.0, 1.0)
    probability = np.asarray([[-0.2, 0.25], [0.75, 1.4]], dtype=np.float32)
    valid = np.asarray([[True, False], [True, True]])

    elevation = online_fusion.probability_elevation_layer(
        probability,
        valid,
        map_spec,
    )

    assert elevation.dtype == np.float32
    assert elevation.shape == map_spec.shape
    assert elevation[0, 0] == 0.0
    assert np.isnan(elevation[0, 1])
    assert elevation[1, 0] == 0.75
    assert elevation[1, 1] == 1.0


def test_probability_elevation_layer_rejects_wrong_map_shape():
    with np.testing.assert_raises_regex(ValueError, "map_spec.shape"):
        online_fusion.probability_elevation_layer(
            np.zeros((1, 2), dtype=np.float32),
            np.ones((1, 2), dtype=bool),
            MapSpec(-1.0, 1.0, -1.0, 1.0, 1.0),
        )


def test_probability_grid_map_layers_invert_each_risk_and_apply_its_valid_mask():
    map_spec = MapSpec(-1.0, 1.0, -1.0, 1.0, 1.0)
    layers = online_fusion.traversability_probability_layers(
        geometry_risk=np.asarray([[-0.2, 0.2], [0.8, 1.2]], dtype=np.float32),
        geometry_valid=np.asarray([[True, True], [False, True]]),
        semantic_risk=np.asarray([[0.1, 0.3], [0.7, 0.9]], dtype=np.float32),
        semantic_valid=np.asarray([[False, True], [True, False]]),
        final_risk=np.asarray([[0.0, 0.4], [0.6, 1.4]], dtype=np.float32),
        final_valid=np.asarray([[True, True], [True, True]]),
        map_spec=map_spec,
    )

    assert tuple(layers) == (
        "geometry_probability",
        "semantic_probability",
        "final_probability",
    )
    assert layers["geometry_probability"][0, 0] == 1.0
    assert layers["geometry_probability"][0, 1] == np.float32(0.8)
    assert np.isnan(layers["geometry_probability"][1, 0])
    assert layers["geometry_probability"][1, 1] == 0.0
    assert np.isnan(layers["semantic_probability"][0, 0])
    assert layers["semantic_probability"][1, 0] == np.float32(0.3)
    assert layers["final_probability"][1, 1] == 0.0


def test_grid_map_column_major_data_preserves_xy_index_order():
    layer = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    data = online_fusion.grid_map_column_major_data(layer)

    assert data.dtype == np.float32
    assert np.array_equal(data, np.asarray([1.0, 3.0, 2.0, 4.0]))
