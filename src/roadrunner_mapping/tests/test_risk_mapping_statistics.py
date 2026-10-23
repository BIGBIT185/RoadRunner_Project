"""Independent numerical counterexamples for map and tail-risk statistics."""
import numpy as np
import pytest
from scipy import ndimage

from roadrunner_mapping.geometry.gpu_rolling_elevation import (
    _compute_world_geometry_fields,
)
from roadrunner_mapping.geometry.gaussian_elevation import (
    GaussianElevationConfig,
)
from roadrunner_mapping.geometry.schema import MapSpec
from roadrunner_mapping.online.incremental_semantic_grid import IncrementalSemanticGrid


def world_fields(
    support_count,
    mean,
    variance=0.0004,
    *,
    valid=None,
    last_seen=None,
    accepted_frame_count=3,
    conflict=0.0,
    conflict_updated_at=None,
    latest_timestamp=10.0,
    max_geometry_age_sec=5.0,
    gaussian_config=None,
):
    support_count = np.asarray(support_count, dtype=np.int32)
    shape = support_count.shape
    mean = np.broadcast_to(mean, shape).astype(np.float32).copy()
    variance = np.broadcast_to(variance, shape).astype(np.float32).copy()
    if valid is None:
        valid = support_count > 0
    if last_seen is None:
        last_seen = np.full(shape, 10.0, dtype=np.float32)
    if conflict_updated_at is None:
        conflict_updated_at = np.asarray(last_seen, dtype=np.float32)
    return _compute_world_geometry_fields(
        cp=np,
        ndimage=ndimage,
        elevation_mean=mean,
        elevation_variance=variance,
        elevation_valid=np.broadcast_to(valid, shape).astype(bool).copy(),
        geometry_support_count=support_count,
        accepted_frame_count=np.broadcast_to(
            accepted_frame_count, shape
        ).astype(np.int32).copy(),
        ground_sum=support_count.astype(np.float32),
        min_z=mean.copy(),
        max_z=mean.copy(),
        last_seen=np.asarray(last_seen, dtype=np.float32),
        variance_updated_at=np.asarray(last_seen, dtype=np.float32),
        conflict_score=np.broadcast_to(conflict, shape).astype(np.float32).copy(),
        conflict_updated_at=np.asarray(conflict_updated_at, dtype=np.float32),
        latest_timestamp=latest_timestamp,
        time_origin=0.0,
        max_geometry_age_sec=max_geometry_age_sec,
        resolution=0.2,
        gaussian_config=(
            GaussianElevationConfig()
            if gaussian_config is None
            else gaussian_config
        ),
    )


@pytest.mark.parametrize('height', [-2., 0., 1., 3.])
@pytest.mark.parametrize('density', [1, 2, 10])
def test_flat_ground_roughness_is_zero_at_any_height_and_density(height, density):
    count=np.full((15,15),density,dtype=np.int32)
    fields=world_fields(count,height)
    np.testing.assert_allclose(fields['roughness'][3:-3,3:-3],0,atol=1e-5)


def test_flat_ground_with_nonuniform_point_density_is_not_rough():
    count=np.arange(1,226,dtype=np.int32).reshape(15,15)
    fields=world_fields(count,2.)
    np.testing.assert_allclose(fields['roughness'][3:-3,3:-3],0,atol=1e-5)


def test_tilted_plane_has_low_plane_fit_roughness():
    rows, columns = np.indices((15, 15), dtype=np.float32)
    plane = 0.03 * rows - 0.02 * columns + 1.2
    fields = world_fields(np.ones((15, 15), dtype=np.int32), plane)
    np.testing.assert_allclose(fields['roughness'][4:-4,4:-4],0,atol=2e-5)


def _plane_fit_validity(local_count, window_area, determinant):
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module

    predicate = getattr(module, '_plane_fit_is_valid', None)
    assert predicate is not None, 'roughness validity predicates must be shared'
    return predicate(
        np,
        np.asarray(local_count, dtype=np.float64),
        float(window_area),
        np.asarray(determinant, dtype=np.float64),
    )


def test_plane_fit_rejects_fewer_than_four_cells_with_sufficient_coverage():
    assert not bool(_plane_fit_validity(3.0, 9.0, 1.0))


def test_plane_fit_rejects_full_rank_support_below_quarter_coverage():
    assert not bool(_plane_fit_validity(4.0, 25.0, 1.0))


def test_plane_fit_rejects_rank_degenerate_support_with_sufficient_coverage():
    assert not bool(_plane_fit_validity(13.0, 49.0, 0.0))


def test_map_boundary_uses_constant_invalid_support_not_nearest_replication():
    support = np.ones((15, 15), dtype=np.int32)
    fields = world_fields(support, 0.0)
    expected_corner_coverage = 16.0 / 49.0
    expected_corner_confidence = (
        lambda value: value * value * (3.0 - 2.0 * value)
    )((expected_corner_coverage - 0.15) / (0.60 - 0.15))
    assert fields['ground_fit_valid'][0, 0]
    assert fields['roughness'][0, 0] == pytest.approx(0.0, abs=1e-6)
    assert fields['geometry_confidence'][0, 0] == pytest.approx(
        expected_corner_confidence, abs=1e-5
    )
    assert fields['geometry_confidence'][0, 0] < fields['geometry_confidence'][7, 7]


def test_single_frame_uncertainty_is_conservative_and_confidence_is_capped():
    support = np.ones((15, 15), dtype=np.int32)
    fields = world_fields(
        support,
        0.0,
        variance=0.01,
        accepted_frame_count=1,
    )
    interior = np.s_[4:-4, 4:-4]
    assert np.min(fields['uncertainty_risk'][interior]) >= 0.8
    assert np.max(fields['geometry_confidence'][interior]) <= 0.25 + 1e-6
    np.testing.assert_allclose(fields['elevation_std'][interior],0.1,atol=1e-6)


def test_confidence_variance_factor_changes_with_only_variance():
    support = np.ones((15, 15), dtype=np.int32)
    low = world_fields(support, 0.0, variance=0.0004)
    high = world_fields(support, 0.0, variance=0.04)
    assert low['geometry_confidence'][7, 7] == pytest.approx(1.0)
    assert high['geometry_confidence'][7, 7] == pytest.approx(0.0)


def test_confidence_freshness_factor_changes_with_only_age():
    support = np.ones((15, 15), dtype=np.int32)
    config = GaussianElevationConfig(process_variance_per_sec=0.0)
    fresh = world_fields(
        support, 0.0, last_seen=np.full((15, 15), 10.0),
        gaussian_config=config,
    )
    stale = world_fields(
        support, 0.0, last_seen=np.full((15, 15), 8.0),
        conflict_updated_at=np.full((15, 15), 10.0),
        gaussian_config=config,
    )
    assert fresh['geometry_confidence'][7, 7] == pytest.approx(1.0)
    assert stale['geometry_confidence'][7, 7] == pytest.approx(np.exp(-2.0 / 5.0))


def test_confidence_local_coverage_factor_changes_with_only_coverage():
    sparse = np.zeros((15, 15), dtype=np.int32)
    sparse[6:9, 6:9] = 1
    dense = np.zeros((15, 15), dtype=np.int32)
    dense[5:10, 5:10] = 1
    sparse_fields = world_fields(sparse, 0.0)
    dense_fields = world_fields(dense, 0.0)
    sparse_coverage = 9.0 / 49.0
    dense_coverage = 25.0 / 49.0

    def smoothstep_coverage(value):
        scaled = np.clip((value - 0.15) / (0.60 - 0.15), 0.0, 1.0)
        return scaled * scaled * (3.0 - 2.0 * scaled)

    assert sparse_fields['geometry_confidence'][7, 7] == pytest.approx(
        smoothstep_coverage(sparse_coverage), abs=1e-6
    )
    assert dense_fields['geometry_confidence'][7, 7] == pytest.approx(
        smoothstep_coverage(dense_coverage), abs=1e-6
    )
    assert sparse_fields['geometry_confidence'][7, 7] < dense_fields['geometry_confidence'][7, 7]


def test_confidence_conflict_factor_changes_with_only_conflict():
    support = np.ones((15, 15), dtype=np.int32)
    clear = world_fields(support, 0.0, conflict=0.0)
    conflicted = world_fields(support, 0.0, conflict=0.6)
    assert clear['geometry_confidence'][7, 7] == pytest.approx(1.0)
    assert conflicted['geometry_confidence'][7, 7] == pytest.approx(0.4)


def test_confidence_frame_factor_is_quarter_for_one_and_one_for_three_frames():
    support = np.ones((15, 15), dtype=np.int32)
    one_frame = world_fields(support, 0.0, accepted_frame_count=1)
    three_frames = world_fields(support, 0.0, accepted_frame_count=3)
    assert one_frame['geometry_confidence'][7, 7] == pytest.approx(0.25)
    assert three_frames['geometry_confidence'][7, 7] == pytest.approx(1.0)


def test_roughness_risk_is_diagnostic_only():
    support = np.ones((15, 15), dtype=np.int32)
    rows, columns = np.indices((15, 15))
    corrugated = ((rows + columns) % 2).astype(np.float32) * 0.08
    fields = world_fields(support, corrugated, variance=0.0004)
    interior = (7, 7)
    assert fields['roughness_risk'][interior] > 0.0
    expected = max(
        fields['slope_risk'][interior],
        fields['positive_step_risk'][interior],
        fields['negative_step_risk'][interior],
        fields['uncertainty_risk'][interior],
    )
    assert fields['geometry_risk'][interior] == pytest.approx(expected)


def test_expired_height_has_no_influence_on_fresh_neighborhood():
    count=np.ones((15,15),dtype=np.int32)
    height=np.zeros((15,15),dtype=np.float32)
    height[:,7:]=3.
    last_seen=np.full((15,15),10.)
    last_seen[:,7:]=0.
    fields=world_fields(count,height,last_seen=last_seen)
    reference_count=count.copy();reference_count[:,7:]=0
    reference=world_fields(reference_count,height,last_seen=last_seen)
    for name in ['slope','roughness','positive_step_height','negative_step_height','geometry_risk']:
        np.testing.assert_allclose(fields[name][:,:7],reference[name][:,:7],atol=1e-5)
    assert not fields['observed'][:,7:].any()


def semantic_grid(backend):
    spec=MapSpec(-1.,1.,-1.,1.,1.)
    kwargs=dict(map_spec=spec,semantic_risk={1:0.,2:1.},grass_class_ids=set(),
                decay_tau_sec=30.,stale_sec=90.,risk_histogram_bins=16,risk_cvar_alpha=.8)
    if backend=='cpu':return IncrementalSemanticGrid(**kwargs)
    cp=pytest.importorskip('cupy')
    from roadrunner_mapping.geometry.gpu_rolling_grid import GpuRollingGrid
    from roadrunner_mapping.online.gpu_incremental_semantic_grid import GpuIncrementalSemanticGrid
    rolling=GpuRollingGrid(world_map_length=4.,resolution=1.,device=cp.cuda.Device(0),cp=cp)
    rolling.move_center(np.zeros(2))
    return GpuIncrementalSemanticGrid(rolling_grid=rolling,**kwargs)


def advance_semantic(grid, labels, confidence=None, timestamp=1.):
    n=len(labels)
    return grid.advance(timestamp=timestamp,world_from_body=np.eye(4),
        xyz_world=np.tile([.25,.25,0.],(n,1)).astype(np.float32),
        labels=np.asarray(labels,dtype=np.uint8),
        confidence=np.ones(n,dtype=np.float32) if confidence is None else np.asarray(confidence,dtype=np.float32),
        valid=np.ones(n,dtype=bool),dynamic=np.zeros(n,dtype=bool))


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_semantic_cvar_preserves_minority_hazard_in_a_single_frame(backend):
    result=advance_semantic(semantic_grid(backend),[1]*9+[2])
    assert result['semantic_hindsight_target'][0,0]==pytest.approx(.1)
    assert result['semantic_hindsight_risk_std'][0,0]==pytest.approx(.3)
    # Top 20% mass = 1 hazardous + 1 safe. Quantized safe risk is 1/16.
    assert result['semantic_hindsight_risk_cvar'][0,0]==pytest.approx((1+1/16)/2)


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_semantic_histogram_mass_and_mean_use_the_same_confidence_weights(backend):
    grid=semantic_grid(backend)
    result=advance_semantic(grid,[1,2],[1.,.1])
    # Top mass=.22, containing .1 at 1 and .12 at the safe-bin upper edge.
    assert result['semantic_hindsight_risk_cvar'][0,0]==pytest.approx((.1+.12/16)/.22,abs=1e-6)
    xp=np if backend=='cpu' else grid.cp
    np.testing.assert_allclose(float(xp.sum(sum(grid._risk_histogram))),1.1,rtol=1e-6)


def gpu_geometry():
    pytest.importorskip('cupy')
    from roadrunner_mapping.geometry.gpu_rolling_elevation import GpuRollingElevationMap
    return GpuRollingElevationMap(map_spec=MapSpec(-1.,1.,-1.,1.,.2),world_map_length=4.)


def geometry_points():
    x,y=np.meshgrid(np.arange(-1.5,1.5,.1),np.arange(-1.5,1.5,.1))
    return np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size))).astype(np.float32)


def ingest_geometry(grid, frame_id, timestamp, points):
    return grid.advance_pointcloud(frame_id,timestamp,points,np.zeros(len(points),dtype=np.float32),
        world_from_lidar=np.eye(4),minimum_range=0.,maximum_range=10.,target_world_from_body=np.eye(4))


def statistics(grid):
    return [grid.cp.asnumpy(a).copy() for a in
            [grid.risk_sum,grid.risk_sq_sum,grid.risk_weight,*grid.risk_histogram]]


def test_geometry_repeated_frame_or_empty_frame_does_not_add_evidence():
    grid=gpu_geometry();points=geometry_points()
    ingest_geometry(grid,'first',1.,points)
    before=statistics(grid)
    ingest_geometry(grid,'first',1.,points)
    for a,b in zip(before,statistics(grid)):np.testing.assert_array_equal(a,b)
    ingest_geometry(grid,'empty',2.,np.empty((0,3),dtype=np.float32))
    for a,b in zip(before,statistics(grid)):np.testing.assert_array_equal(a,b)


def test_geometry_new_frame_at_same_stamp_is_rejected_without_mutation():
    grid=gpu_geometry();points=geometry_points()
    ingest_geometry(grid,'first',1.,points)
    before=statistics(grid)
    fresh=np.array([[.11,.11,0]],dtype=np.float32)
    with pytest.raises(ValueError, match='strictly increasing'):
        ingest_geometry(grid,'second',1.,fresh)
    for a,b in zip(before,statistics(grid)):np.testing.assert_array_equal(a,b)


def test_geometry_reobservation_does_not_revive_expired_height_history():
    grid=gpu_geometry();old=geometry_points();old[:,2]=2.
    ingest_geometry(grid,'old_surface',1.,old)
    fresh=geometry_points()
    result=ingest_geometry(grid,'new_surface',200.,fresh)
    valid=result['geometry_valid_mask']
    assert valid.any()
    np.testing.assert_allclose(result['elevation'][valid],0.,atol=1e-5)
    # A new surface after expiry should start its own risk evidence.
    assert float(grid.cp.max(grid.risk_weight)) <= 1.


def test_geometry_pending_observations_are_consumed_per_frame_and_clear_with_storage():
    grid=gpu_geometry();cp=grid.cp
    grid.rolling_grid.move_center(np.zeros(2))
    with grid.device:
        points=cp.asarray(geometry_points())
        grid._ingest_world_arrays('pending',1.,points,cp.ones(len(points)),cp.zeros(len(points),dtype=cp.bool_))
        assert not bool(cp.any(grid.risk_pending_observation))
        grid.rolling_grid.move_center(np.array([10.,0.]))
        assert not bool(cp.any(grid.risk_pending_observation))
    grid.clear()
    assert not bool(cp.any(grid.risk_pending_observation))
