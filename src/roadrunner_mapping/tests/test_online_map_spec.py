import numpy as np
import pytest

from roadrunner_mapping.geometry.schema import (
    DEFAULT_MAP_SPEC,
    ONLINE_MAP_SPEC,
    make_online_map_spec,
)
from roadrunner_mapping.online.online_fusion import traversability_probability_layers
from roadrunner_mapping.online.incremental_semantic_grid import IncrementalSemanticGrid


def test_online_map_is_40m_without_changing_offline_contract():
    assert ONLINE_MAP_SPEC.shape == (200, 200)
    assert ONLINE_MAP_SPEC.x_max - ONLINE_MAP_SPEC.x_min == 40.0
    assert ONLINE_MAP_SPEC.y_max - ONLINE_MAP_SPEC.y_min == 40.0
    assert DEFAULT_MAP_SPEC.shape == (200, 200)


def test_online_map_spec_accepts_external_dimensions_and_resolution():
    spec = make_online_map_spec(
        length_x_m=30.0,
        length_y_m=20.0,
        resolution_m=0.1,
    )
    assert spec.x_min == -15.0
    assert spec.x_max == 15.0
    assert spec.y_min == -10.0
    assert spec.y_max == 10.0
    assert spec.resolution == 0.1
    assert spec.shape == (300, 200)


def test_online_map_spec_square_shortcut_and_validation():
    square = make_online_map_spec(length_x_m=12.0, resolution_m=0.5)
    assert square.x_max - square.x_min == 12.0
    assert square.y_max - square.y_min == 12.0
    with pytest.raises(ValueError):
        make_online_map_spec(length_x_m=12.0, length_y_m=10.0, resolution_m=0.0)


def test_online_world_grid_contains_rotated_output_map():
    output_diagonal = np.hypot(
        ONLINE_MAP_SPEC.x_max - ONLINE_MAP_SPEC.x_min,
        ONLINE_MAP_SPEC.y_max - ONLINE_MAP_SPEC.y_min,
    )
    assert output_diagonal < 80.0
    assert 80.0 / ONLINE_MAP_SPEC.resolution == 400


def test_cpu_semantic_fallback_and_probability_layers_keep_online_shape():
    semantic = IncrementalSemanticGrid(
        map_spec=ONLINE_MAP_SPEC,
        semantic_risk={},
    )
    assert semantic._output_body_xy.shape == (200 * 200, 2)
    zeros = np.zeros(ONLINE_MAP_SPEC.shape, dtype=np.float32)
    valid = np.ones(ONLINE_MAP_SPEC.shape, dtype=bool)
    layers = traversability_probability_layers(
        geometry_risk=zeros,
        geometry_valid=valid,
        semantic_risk=zeros,
        semantic_valid=valid,
        final_risk=zeros,
        final_valid=valid,
        map_spec=ONLINE_MAP_SPEC,
    )
    assert all(layer.shape == (200, 200) for layer in layers.values())


def test_cpu_world_semantic_snapshot_uses_rolling_map_shape():
    semantic = IncrementalSemanticGrid(
        map_spec=ONLINE_MAP_SPEC,
        semantic_risk={0: 0.2},
    )
    semantic.update(
        timestamp=1.0,
        world_from_body=np.eye(4),
        xyz_world=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        labels=np.asarray([0], dtype=np.uint8),
        confidence=np.asarray([1.0], dtype=np.float32),
        valid=np.asarray([True]),
        dynamic=np.asarray([False]),
    )
    snapshot = semantic.sample_world(
        timestamp=1.0,
        center=np.zeros(2),
        world_map_length=80.0,
        resolution=ONLINE_MAP_SPEC.resolution,
    )
    assert snapshot["semantic_hindsight_target"].shape == (400, 400)
    assert snapshot["semantic_hindsight_valid_mask"].sum() == 1
