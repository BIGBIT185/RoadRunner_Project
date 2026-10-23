import numpy as np
import pytest


def test_bilinear_world_xy_samples_returns_four_valid_neighbors():
    from roadrunner_mapping.geometry.gpu_rolling_grid import (
        bilinear_world_xy_samples,
    )

    rows, columns, weights, valid_weight = bilinear_world_xy_samples(
        np.asarray([0.0]),
        np.asarray([0.0]),
        center=np.asarray([0.0, 0.0]),
        world_map_length=4.0,
        resolution=1.0,
        cell_n=4,
        xp=np,
    )

    # The world origin lies at the boundary between the four central cells.
    np.testing.assert_array_equal(rows[:, 0], [1, 1, 2, 2])
    np.testing.assert_array_equal(columns[:, 0], [1, 2, 1, 2])
    np.testing.assert_allclose(weights[:, 0], 0.25)
    np.testing.assert_allclose(valid_weight, [1.0])


def test_bilinear_world_xy_samples_renormalizes_at_boundary():
    from roadrunner_mapping.geometry.gpu_rolling_grid import (
        bilinear_world_xy_samples,
    )

    rows, columns, weights, valid_weight = bilinear_world_xy_samples(
        np.asarray([2.0 - 1e-6]),
        np.asarray([2.0 - 1e-6]),
        center=np.asarray([0.0, 0.0]),
        world_map_length=4.0,
        resolution=1.0,
        cell_n=4,
        xp=np,
    )

    assert rows.shape == (4, 1)
    assert columns.shape == (4, 1)
    assert np.isfinite(weights).all()
    assert float(valid_weight[0]) == pytest.approx(0.250001, abs=2e-6)


def test_bilinear_world_xy_samples_changes_continuously_for_small_yaw():
    from roadrunner_mapping.geometry.gpu_rolling_grid import (
        bilinear_world_xy_samples,
    )

    rows, columns = np.meshgrid(
        np.arange(4, dtype=np.float64),
        np.arange(4, dtype=np.float64),
        indexing="ij",
    )
    body_x = 2.0 - (rows + 0.5)
    body_y = 2.0 - (columns + 0.5)
    angle = np.deg2rad(0.5)
    world_x = np.cos(angle) * body_x - np.sin(angle) * body_y
    world_y = np.sin(angle) * body_x + np.cos(angle) * body_y
    _, _, weights, valid_weight = bilinear_world_xy_samples(
        world_x.reshape(-1),
        world_y.reshape(-1),
        center=np.asarray([0.0, 0.0]),
        world_map_length=4.0,
        resolution=1.0,
        cell_n=4,
        xp=np,
    )

    # A small rotation should retain most interpolation mass instead of
    # turning every output cell into an unrelated nearest-neighbor lookup.
    assert np.count_nonzero(valid_weight >= 0.5) >= 8
    assert np.all(weights >= 0.0)


def test_weighted_neighbor_sample_ignores_unobserved_cells_and_normalizes():
    from roadrunner_mapping.geometry.gpu_rolling_grid import (
        weighted_neighbor_sample,
    )

    values = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    rows = np.asarray([[0], [0], [1], [1]])
    columns = np.asarray([[0], [1], [0], [1]])
    weights = np.asarray([[0.25], [0.25], [0.25], [0.25]])
    support = np.asarray([[True, False], [True, True]])
    sampled, valid_weight, valid = weighted_neighbor_sample(
        values,
        rows,
        columns,
        weights,
        support=support,
        min_valid_weight=0.5,
        xp=np,
    )

    np.testing.assert_allclose(sampled, [8.0 / 3.0])
    np.testing.assert_allclose(valid_weight, [0.75])
    np.testing.assert_array_equal(valid, [True])


def test_shared_gpu_grid_rolls_all_registered_layers_and_reuses_indices():
    cp = pytest.importorskip("cupy")
    from roadrunner_mapping.geometry.gpu_rolling_grid import (
        GpuRollingGrid,
    )

    grid = GpuRollingGrid(
        world_map_length=4.0,
        resolution=1.0,
        device=cp.cuda.Device(0),
        cp=cp,
    )
    geometry_layer = cp.zeros((4, 4), dtype=cp.float32)
    semantic_layer = cp.zeros((4, 4), dtype=cp.float32)
    grid.register_layer(geometry_layer, 0.0)
    grid.register_layer(semantic_layer, 0.0)

    grid.move_center(np.asarray([0.0, 0.0]))
    geometry_layer[2, 2] = 3.0
    semantic_layer[2, 2] = 7.0
    grid.move_center(np.asarray([1.0, 0.0]))

    assert float(geometry_layer[3, 2].get()) == 3.0
    assert float(semantic_layer[3, 2].get()) == 7.0
    assert np.allclose(geometry_layer[0].get(), 0.0)
    assert np.allclose(semantic_layer[0].get(), 0.0)

    rows, columns, valid = grid.world_indices(
        cp.asarray([[1.0, 0.0, 0.0]], dtype=cp.float32)
    )
    assert int(rows[0].get()) == 2
    assert int(columns[0].get()) == 2
    assert bool(valid[0].get())


def test_gpu_semantic_layer_matches_cpu_before_and_after_shared_roll():
    cp = pytest.importorskip("cupy")
    from roadrunner_mapping.online.gpu_incremental_semantic_grid import (
        GpuIncrementalSemanticGrid,
    )
    from roadrunner_mapping.online.online_fusion import IncrementalSemanticGrid
    from roadrunner_mapping.geometry.gpu_rolling_grid import (
        GpuRollingGrid,
    )
    from roadrunner_mapping.geometry.schema import MapSpec

    spec = MapSpec(-1.0, 1.0, -1.0, 1.0, 1.0)
    rolling = GpuRollingGrid(
        world_map_length=4.0,
        resolution=1.0,
        device=cp.cuda.Device(0),
        cp=cp,
    )
    rolling.move_center(np.zeros(2, dtype=np.float64))
    gpu = GpuIncrementalSemanticGrid(
        map_spec=spec,
        semantic_risk={1: 0.2, 2: 0.8},
        grass_class_ids=set(),
        stale_sec=30.0,
        rolling_grid=rolling,
    )
    cpu = IncrementalSemanticGrid(
        map_spec=spec,
        semantic_risk={1: 0.2, 2: 0.8},
        grass_class_ids=set(),
    )
    kwargs = dict(
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
    gpu_result = gpu.advance(**kwargs)
    cpu_result = cpu.advance(**kwargs)
    for key in (
        "semantic_hindsight_target",
        "semantic_hindsight_valid_mask",
        "semantic_hindsight_confidence",
        "semantic_hindsight_observation_count",
        "semantic_hindsight_risk_std",
        "semantic_hindsight_risk_cvar",
        "semantic_hindsight_risk_distribution_valid",
        "grass_probability",
        "grass_observation_weight",
        "grass_mask",
    ):
        np.testing.assert_allclose(
            gpu_result[key], cpu_result[key], rtol=1e-5, atol=1e-5
        )

    rolling.move_center(np.asarray([1.0, 0.0]))
    shifted_pose = np.eye(4)
    shifted_pose[0, 3] = 1.0
    gpu_shifted = gpu.sample(timestamp=10.0, world_from_body=shifted_pose)
    cpu_shifted = cpu.sample(timestamp=10.0, world_from_body=shifted_pose)
    np.testing.assert_allclose(
        gpu_shifted["semantic_hindsight_target"],
        cpu_shifted["semantic_hindsight_target"],
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_array_equal(
        gpu_shifted["semantic_hindsight_valid_mask"],
        cpu_shifted["semantic_hindsight_valid_mask"],
    )
    np.testing.assert_allclose(
        gpu_shifted["semantic_hindsight_risk_std"],
        cpu_shifted["semantic_hindsight_risk_std"],
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        gpu_shifted["semantic_hindsight_risk_cvar"],
        cpu_shifted["semantic_hindsight_risk_cvar"],
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_array_equal(
        gpu_shifted["semantic_hindsight_risk_distribution_valid"],
        cpu_shifted["semantic_hindsight_risk_distribution_valid"],
    )
