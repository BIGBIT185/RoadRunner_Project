#!/usr/bin/env python3
"""GOOSE ROS1 实时可通行性构建节点。

本文件是在线链的“融合层”：它不负责训练模型，而是把当前 RGB、LiDAR、
CameraInfo 和 TF 组织成两个异步流水线。几何流水线优先发布 GPU 高程图，
语义流水线在 EoMT 预测到达后补充世界系语义栅格；两者最后在当前车体系
BEV 网格上融合并发布为 GridMap。通过 ``enable_semantic:=false`` 可以完全
关闭语义流水线，仅运行几何地图。
"""

from __future__ import annotations

import numbers
import queue
import threading
import time
from types import SimpleNamespace

import message_filters
import numpy as np
import rospy
import tf2_ros
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from grid_map_msgs.msg import GridMap
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Bool, Float32MultiArray, MultiArrayDimension, MultiArrayLayout

from roadrunner_mapping.sensors.pointcloud_adapter import (
    decode_goose_cloud,
    decode_goose_cloud_with_rows,
    rigid_align_cloud_to_image,
)
from roadrunner_mapping.sensors.calibration import rectified_intrinsics
from roadrunner_mapping.online.gpu_pointcloud import decode_pointcloud2_gpu
from roadrunner_mapping.online.gpu_incremental_semantic_grid import (
    GpuIncrementalSemanticGrid,
)
from roadrunner_mapping.msg import EomtSemanticPrediction
from roadrunner_mapping.online.online_backend import (
    BoundedCameraInfoCache,
    EomtHealthMonitor,
    initialize_inline_segmenter,
    resolve_semantic_backend,
    timestamp_is_future,
)
from roadrunner_mapping.online.online_fusion import (
    IncrementalSemanticGrid,
    RecentFrameWindow,
    grid_map_column_major_data,
    traversability_probability_layers,
)
from roadrunner_mapping.semantics.semantic_transport import (
    BoundedSemanticMatcher,
    PredictionValidationError,
    stamp_to_nsec,
    validate_atomic_prediction,
)
from roadrunner_mapping.semantics.semantic_profiles import get_eomt_profile
from roadrunner_mapping.geometry.geometry_hindsight import (
    apply_grass_rule_to_accumulated_geometry,
    fuse_accumulated_geometry_semantics,
    geometry_cvar_risk,
    geometry_risk_variants,
    semantic_cvar_risk,
    semantic_risk_variants,
)
from roadrunner_mapping.geometry.gaussian_elevation import (
    GaussianElevationConfig,
    point_height_variance,
)
from roadrunner_mapping.geometry.elevation_map_lifecycle import (
    ElevationMapLifecycleConfig,
)
from roadrunner_mapping.geometry.gpu_rolling_elevation import GpuRollingElevationMap
from roadrunner_mapping.geometry.coordinates import transform_points
from roadrunner_mapping.geometry.pointcloud import (
    apply_row_transforms,
    interpolate_scan_transforms,
    scan_row_relative_times,
)
from roadrunner_mapping.geometry.schema import (
    ONLINE_MAP_SPEC,
    MapSpec,
    make_online_map_spec,
)


_GEOMETRY_TOP_LEVEL_TIMING_NAMES = (
    "geometry_prepare_sec",
    "geometry_drift_sec",
    "geometry_visibility_sec",
    "geometry_map_update_sec",
    "geometry_world_fields_sec",
    "geometry_risk_sec",
    "geometry_sample_sec",
    "geometry_transfer_sec",
)
_GEOMETRY_MAP_TIMING_NAMES = (
    "geometry_map_center_sec",
    "geometry_map_maintenance_sec",
    "geometry_map_backup_sec",
    "geometry_map_clear_sec",
    "geometry_map_visibility_apply_sec",
    "geometry_map_fusion_sec",
    "geometry_map_commit_sec",
)
_GEOMETRY_DIAGNOSTIC_TIMING_NAMES = (
    _GEOMETRY_TOP_LEVEL_TIMING_NAMES[:4]
    + _GEOMETRY_MAP_TIMING_NAMES
    + _GEOMETRY_TOP_LEVEL_TIMING_NAMES[4:]
)


def _load_inline_deeplab_segmenter(checkpoint: str, device: str):
    """按需加载 DeepLab，避免 EoMT/几何模式依赖 torchvision。"""
    from roadrunner_mapping.geometry.rugd import OfflineRUGDSegmenter

    return OfflineRUGDSegmenter(checkpoint, device=device)
from roadrunner_mapping.geometry.semantic import (
    RUGD_CLASSES,
    DYNAMIC_CLASS_IDS,
    GROUND_CLASS_IDS,
    SEMANTIC_RISK,
    project_semantics_to_points,
)


def transform_matrix(message) -> np.ndarray:
    """把 geometry_msgs/TransformStamped 转换为 4x4 齐次变换矩阵。

    TF 中的四元数可能来自不同发布者，因此这里重新归一化并拒绝无效范数，
    避免一个坏 TF 直接污染后续的世界坐标点云。
    """
    translation = message.transform.translation
    rotation = message.transform.rotation
    x, y, z, w = rotation.x, rotation.y, rotation.z, rotation.w
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("TF quaternion is invalid")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    matrix[:3, 3] = (translation.x, translation.y, translation.z)
    return matrix


def _ros_bool(value) -> bool:
    """Normalize ROS YAML/launch boolean parameters."""

    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "off", "no"}
    return bool(value)


def _lidar_height_variance(
    points_lidar: np.ndarray,
    gaussian_config: GaussianElevationConfig,
) -> np.ndarray:
    """在任何世界变换前，由 LiDAR 原始三维距离生成逐点高度方差。"""

    points = np.asarray(points_lidar)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("LiDAR points must have shape [N,3]")
    ranges = np.linalg.norm(points, axis=1)
    return np.asarray(
        point_height_variance(np, ranges, gaussian_config),
        dtype=np.float32,
    )


def _required_real_ros_param(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError("{} must be a real number, not bool or string".format(name))
    return float(value)


def _required_integer_ros_param(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError("{} must be an integer, not bool, float or string".format(name))
    return int(value)


def _required_bool_ros_param(name: str, value) -> bool:
    if not isinstance(value, bool):
        raise TypeError("{} must be a boolean".format(name))
    return value


def _lifecycle_config_from_ros() -> ElevationMapLifecycleConfig:
    """仅在总开关打开时读取完整生命周期配置。"""

    enabled = _required_bool_ros_param(
        "enable_elevation_map_lifecycle",
        rospy.get_param("~enable_elevation_map_lifecycle", False),
    )
    if not enabled:
        return ElevationMapLifecycleConfig(enabled=False)
    boolean_names = (
        "enable_visibility_cleanup",
        "enable_upper_bound",
        "enable_height_drift_compensation",
        "enable_overhead_filter",
        "enable_wall_edge_sharpening",
        "enable_overlap_clearance",
        "enable_continuous_validity",
    )
    integer_names = (
        "drift_min_frames",
        "drift_min_cells",
        "wall_point_count_threshold",
        "max_visibility_rays_per_frame",
    )
    values = {
        name: _required_bool_ros_param(name, rospy.get_param("~" + name))
        for name in boolean_names
    }
    values.update(
        {
            name: _required_integer_ros_param(
                name, rospy.get_param("~" + name)
            )
            for name in integer_names
        }
    )
    excluded = {"enabled", *boolean_names, *integer_names}
    values.update(
        {
            name: _required_real_ros_param(
                name, rospy.get_param("~" + name)
            )
            for name in ElevationMapLifecycleConfig.__dataclass_fields__
            if name not in excluded
        }
    )
    return ElevationMapLifecycleConfig(enabled=True, **values)


def _gaussian_config_from_ros() -> GaussianElevationConfig:
    """读取完整高斯配置；所有字段必填并由数据类统一校验。"""

    return GaussianElevationConfig(
        sensor_std_m=_required_real_ros_param(
            "sensor_std_m", rospy.get_param("~sensor_std_m")
        ),
        range_variance_factor=_required_real_ros_param(
            "range_variance_factor", rospy.get_param("~range_variance_factor")
        ),
        measurement_variance_min=_required_real_ros_param(
            "measurement_variance_min",
            rospy.get_param("~measurement_variance_min"),
        ),
        measurement_variance_max=_required_real_ros_param(
            "measurement_variance_max",
            rospy.get_param("~measurement_variance_max"),
        ),
        pose_vertical_variance=_required_real_ros_param(
            "pose_vertical_variance", rospy.get_param("~pose_vertical_variance")
        ),
        frame_variance_min=_required_real_ros_param(
            "frame_variance_min", rospy.get_param("~frame_variance_min")
        ),
        posterior_variance_min=_required_real_ros_param(
            "posterior_variance_min", rospy.get_param("~posterior_variance_min")
        ),
        single_point_initial_variance=_required_real_ros_param(
            "single_point_initial_variance",
            rospy.get_param("~single_point_initial_variance"),
        ),
        multi_point_initial_variance=_required_real_ros_param(
            "multi_point_initial_variance",
            rospy.get_param("~multi_point_initial_variance"),
        ),
        max_variance=_required_real_ros_param(
            "max_variance", rospy.get_param("~max_variance")
        ),
        process_variance_per_sec=_required_real_ros_param(
            "process_variance_per_sec", rospy.get_param("~process_variance_per_sec")
        ),
        outlier_variance_per_frame=_required_real_ros_param(
            "outlier_variance_per_frame",
            rospy.get_param("~outlier_variance_per_frame"),
        ),
        mahalanobis_threshold=_required_real_ros_param(
            "mahalanobis_threshold", rospy.get_param("~mahalanobis_threshold")
        ),
        max_effective_points_per_frame=_required_integer_ros_param(
            "max_effective_points_per_frame",
            rospy.get_param("~max_effective_points_per_frame"),
        ),
        initial_cluster_gap_m=_required_real_ros_param(
            "initial_cluster_gap_m", rospy.get_param("~initial_cluster_gap_m")
        ),
        initial_min_cluster_points=_required_integer_ros_param(
            "initial_min_cluster_points",
            rospy.get_param("~initial_min_cluster_points"),
        ),
        conflict_tau_sec=_required_real_ros_param(
            "conflict_tau_sec", rospy.get_param("~conflict_tau_sec")
        ),
        confidence_variance_good=_required_real_ros_param(
            "confidence_variance_good",
            rospy.get_param("~confidence_variance_good"),
        ),
        confidence_variance_bad=_required_real_ros_param(
            "confidence_variance_bad", rospy.get_param("~confidence_variance_bad")
        ),
        uncertainty_std_good=_required_real_ros_param(
            "uncertainty_std_good", rospy.get_param("~uncertainty_std_good")
        ),
        uncertainty_std_bad=_required_real_ros_param(
            "uncertainty_std_bad", rospy.get_param("~uncertainty_std_bad")
        ),
        local_coverage_low=_required_real_ros_param(
            "local_coverage_low", rospy.get_param("~local_coverage_low")
        ),
        local_coverage_high=_required_real_ros_param(
            "local_coverage_high", rospy.get_param("~local_coverage_high")
        ),
    )


def _geometry_frame_time_action(
    frame_id: str,
    timestamp: float,
    ingested_ids,
    last_timestamp,
) -> str:
    """先去重，再区分正常新帧、同时间新 ID 与真正的时间回退。"""

    if frame_id in ingested_ids:
        return "duplicate"
    if last_timestamp is None:
        return "accept"
    if timestamp == last_timestamp:
        raise ValueError("new frame ID reuses the previous timestamp")
    if timestamp < last_timestamp:
        return "rewind"
    return "accept"


def _record_successful_traversability_publish(node, now=None) -> None:
    """只在整套概率图发布成功后更新输出频率 EMA。"""

    current = time.monotonic() if now is None else float(now)
    if not np.isfinite(current):
        raise ValueError("publish monotonic time must be finite")
    previous = getattr(node, "last_traversability_publish_monotonic", None)
    if previous is not None and np.isfinite(previous) and current > previous:
        interval = current - previous
        old_ema = getattr(node, "traversability_publish_interval_ema_sec", None)
        ema = (
            interval
            if old_ema is None or not np.isfinite(old_ema)
            else 0.2 * interval + 0.8 * old_ema
        )
        node.traversability_publish_interval_ema_sec = ema
        node.traversability_publish_hz = 1.0 / ema
    node.last_traversability_publish_monotonic = current


def _snapshot_online_state(node):
    """备份低频重建可能修改的完整在线滚动状态。

    语义图存在时，调用方必须已持有 ``semantic_map_lock``；函数内再按
    项目统一顺序获取 ``rolling_grid.lock``，避免语义/GPU 锁顺序反转。
    """

    geometry_map = node.geometry_map
    rolling_grid = getattr(geometry_map, "rolling_grid", None)
    rolling = None
    if rolling_grid is not None:
        with rolling_grid.lock:
            with geometry_map.device:
                layers = tuple(
                    layer.copy() for layer, _ in rolling_grid._layers
                )
        rolling = {
            "layers": layers,
            "center": rolling_grid.center,
            "initialized": rolling_grid.initialized,
        }
    map_metadata = {}
    for name in (
        "_time_origin",
        "_latest_timestamp",
        "_ingested_ids",
        "_ingested_timestamps",
        "_ingested_set",
        "_last_world_geometry_fields",
        "height_drift_bias_z",
        "height_drift_last_update",
        "last_lifecycle_stats",
    ):
        if hasattr(geometry_map, name):
            value = getattr(geometry_map, name)
            if isinstance(value, list):
                value = list(value)
            elif isinstance(value, set):
                value = set(value)
            elif isinstance(value, dict):
                value = dict(value)
            map_metadata[name] = value
    fake_frames = (
        list(geometry_map.frames) if hasattr(geometry_map, "frames") else None
    )
    fake_clear_count = getattr(geometry_map, "clear_count", None)
    semantic_state = None
    semantic_map = node.semantic_map
    semantic_metadata_names = (
        "_last_timestamp",
        "_last_position",
        "_last_input_points",
        "_last_aggregated_cells",
    )
    if isinstance(semantic_map, IncrementalSemanticGrid):
        cpu_array_names = (
            "_key_x",
            "_key_y",
            "_risk_sum",
            "_risk_sq_sum",
            "_weight_sum",
            "_grass_sum",
            "_count_sum",
            "_confidence",
            "_latest_risk",
            "_latest_confidence",
            "_last_update",
            "_last_conflict",
            "_last_dynamic",
        )
        semantic_state = {
            "kind": "cpu",
            "arrays": {
                name: getattr(semantic_map, name).copy()
                for name in cpu_array_names
            },
            "histogram": tuple(
                histogram.copy() for histogram in semantic_map._risk_histogram
            ),
            "metadata": {
                name: (
                    getattr(semantic_map, name).copy()
                    if isinstance(getattr(semantic_map, name), np.ndarray)
                    else getattr(semantic_map, name)
                )
                for name in semantic_metadata_names
            },
        }
    elif isinstance(semantic_map, GpuIncrementalSemanticGrid):
        # GPU 风险数组和直方图均已注册到 rolling_grid._layers，上面的
        # rolling 快照覆盖它们；这里只保存未注册的时序/统计元数据。
        semantic_state = {
            "kind": "gpu",
            "metadata": {
                name: (
                    getattr(semantic_map, name).copy()
                    if isinstance(getattr(semantic_map, name), np.ndarray)
                    else getattr(semantic_map, name)
                )
                for name in semantic_metadata_names
            },
        }
    elif semantic_map is not None:
        # 测试替身或兼容实现仅约定这四项公共在线元数据。
        semantic_state = {
            "kind": "metadata",
            "metadata": {
                name: (
                    getattr(semantic_map, name).copy()
                    if isinstance(getattr(semantic_map, name), np.ndarray)
                    else getattr(semantic_map, name)
                )
                for name in semantic_metadata_names
                if hasattr(semantic_map, name)
            },
        }
    with node.pose_lock:
        local_from_world = (
            None
            if node.local_from_world is None
            else node.local_from_world.copy()
        )
    with node.scan_transform_cache_lock:
        scan_transform_cache = dict(node.scan_transform_cache)
    return {
        "rolling": rolling,
        "map_metadata": map_metadata,
        "fake_frames": fake_frames,
        "fake_clear_count": fake_clear_count,
        "semantic_state": semantic_state,
        "window": tuple(node.geometry_window),
        "geometry_by_id": dict(node.geometry_by_id),
        "geometry_ingested_ids": set(node.geometry_ingested_ids),
        "local_from_world": local_from_world,
        "last_timestamp": node.last_timestamp,
        "scan_period_estimate_sec": node.scan_period_estimate_sec,
        "scan_transform_cache": scan_transform_cache,
        "active_epoch": node.active_epoch,
        "last_stage": node.last_stage,
    }


def _restore_online_state(node, snapshot) -> None:
    geometry_map = node.geometry_map
    rolling = snapshot["rolling"]
    if rolling is not None:
        rolling_grid = geometry_map.rolling_grid
        with rolling_grid.lock:
            with geometry_map.device:
                for (layer, _), backup in zip(
                    rolling_grid._layers, rolling["layers"]
                ):
                    layer[...] = backup
            rolling_grid._center = rolling["center"].copy()
            rolling_grid._initialized = bool(rolling["initialized"])
    for name, value in snapshot["map_metadata"].items():
        if isinstance(value, list):
            value = list(value)
        elif isinstance(value, set):
            value = set(value)
        elif isinstance(value, dict):
            value = dict(value)
        setattr(geometry_map, name, value)
    if snapshot["fake_frames"] is not None:
        geometry_map.frames = list(snapshot["fake_frames"])
    if snapshot["fake_clear_count"] is not None:
        geometry_map.clear_count = snapshot["fake_clear_count"]
    semantic_state = snapshot["semantic_state"]
    if node.semantic_map is not None and semantic_state is not None:
        if semantic_state["kind"] == "cpu":
            for name, value in semantic_state["arrays"].items():
                setattr(node.semantic_map, name, value.copy())
            node.semantic_map._risk_histogram = [
                histogram.copy()
                for histogram in semantic_state["histogram"]
            ]
        for name, value in semantic_state["metadata"].items():
            if isinstance(value, np.ndarray):
                value = value.copy()
            setattr(node.semantic_map, name, value)
    node.geometry_window.clear()
    for frame in snapshot["window"]:
        node.geometry_window.append(frame)
    node.geometry_by_id.clear()
    node.geometry_by_id.update(snapshot["geometry_by_id"])
    node.geometry_ingested_ids.clear()
    node.geometry_ingested_ids.update(snapshot["geometry_ingested_ids"])
    with node.pose_lock:
        node.local_from_world = (
            None
            if snapshot["local_from_world"] is None
            else snapshot["local_from_world"].copy()
        )
    node.last_timestamp = snapshot["last_timestamp"]
    node.scan_period_estimate_sec = snapshot["scan_period_estimate_sec"]
    with node.scan_transform_cache_lock:
        node.scan_transform_cache.clear()
        node.scan_transform_cache.update(snapshot["scan_transform_cache"])
    node.active_epoch = snapshot["active_epoch"]
    node.last_stage = snapshot["last_stage"]


def image_message_to_rgb(message: Image) -> np.ndarray:
    """解码 ROS 图像消息为连续的 RGB 数组。

    融合节点不导入 OpenCV，避免 ROS Python 运行时和语义环境发生依赖冲突；
    bgr8 仅在这里做通道翻转，后续分割器统一接收 RGB。
    """

    height, width, step = int(message.height), int(message.width), int(message.step)
    encoding = str(message.encoding).lower()
    if encoding not in {"rgb8", "bgr8"}:
        raise ValueError(f"unsupported online image encoding: {message.encoding}")
    row_bytes = width * 3
    if height <= 0 or width <= 0 or step < row_bytes:
        raise ValueError("invalid online image dimensions or step")
    buffer = np.frombuffer(message.data, dtype=np.uint8)
    if buffer.size < height * step:
        raise ValueError("online image data is truncated")
    packed = buffer[: height * step].reshape(height, step)[:, :row_bytes]
    packed = packed.reshape(height, width, 3)
    return packed.copy() if encoding == "rgb8" else packed[:, :, ::-1].copy()


def float_image_message(values: np.ndarray, stamp, frame_id: str) -> Image:
    """把二维浮点地图包装成 RViz 可订阅的 ``sensor_msgs/Image/32FC1``。"""
    array = np.ascontiguousarray(values, dtype="<f4")
    if array.ndim != 2:
        raise ValueError("published float map must be two-dimensional")
    message = Image()
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.height, message.width = array.shape
    message.encoding = "32FC1"
    message.is_bigendian = 0
    message.step = array.shape[1] * array.dtype.itemsize
    message.data = array.tobytes()
    return message


def probability_grid_map_message(
    probability_layers,
    stamp,
    frame_id,
    map_spec=ONLINE_MAP_SPEC,
    center_xy=None,
    elevation=None,
) -> GridMap:
    """创建包含多个可通行概率图层的 GridMap 消息。

    GridMap 使用 column-major 数据布局，而 NumPy 数组通常按 row-major 处理，
    因此最终写入前必须通过 ``grid_map_column_major_data`` 转换；这里发布的
    是概率（1-risk），不是风险值。
    """
    if elevation is not None:
        height_values = np.asarray(elevation, dtype=np.float32)
        if height_values.shape != map_spec.shape:
            raise ValueError("elevation shape must match map_spec.shape")
        height_valid = np.isfinite(height_values)
        if "geometry_probability" in probability_layers:
            height_valid &= np.isfinite(probability_layers["geometry_probability"])
        probability_layers = dict(probability_layers)
        probability_layers["elevation"] = np.where(
            height_valid, height_values, np.nan
        ).astype(np.float32)
    message = GridMap()
    message.info.header.stamp = stamp
    message.info.header.frame_id = frame_id
    message.info.resolution = float(map_spec.resolution)
    message.info.length_x = float(map_spec.x_max - map_spec.x_min)
    message.info.length_y = float(map_spec.y_max - map_spec.y_min)
    if center_xy is None:
        center_xy = (
            0.5 * (map_spec.x_min + map_spec.x_max),
            0.5 * (map_spec.y_min + map_spec.y_max),
        )
    center_xy = np.asarray(center_xy, dtype=np.float64).reshape(2)
    message.info.pose.position.x = float(center_xy[0])
    message.info.pose.position.y = float(center_xy[1])
    message.info.pose.orientation.w = 1.0
    message.layers = list(probability_layers)
    message.basic_layers = ["final_probability"]
    if elevation is not None:
        message.basic_layers.append("elevation")
    message.data = []
    for values in probability_layers.values():
        height, width = values.shape
        layer = Float32MultiArray()
        layer.layout = MultiArrayLayout()
        layer.layout.dim = [
            MultiArrayDimension(
                label="column_index",
                size=width,
                stride=width * height,
            ),
            MultiArrayDimension(
                label="row_index",
                size=height,
                stride=height,
            ),
        ]
        layer.data = tuple(grid_map_column_major_data(values))
        message.data.append(layer)
    message.outer_start_index = 0
    message.inner_start_index = 0
    return message


class RoadRunnerMapping:
    """GOOSE 在线几何主链和异步语义增强的 ROS 节点。"""

    def __init__(self) -> None:
        # 所有参数都从私有命名空间读取，launch 文件可以在不改代码的情况下
        # 切换 EoMT/DeepLab、地图窗口和传感器 topic。
        self.world_frame = rospy.get_param("~world_frame", "utm").lstrip("/")
        self.base_frame = rospy.get_param("~base_frame", "base_link").lstrip("/")
        self.camera_frame = rospy.get_param(
            "~camera_frame", "sensor/camera/windshield/vis"
        ).lstrip("/")
        self.lidar_frame = rospy.get_param(
            "~lidar_frame", "sensor/lidar/vls128_roof"
        ).lstrip("/")
        self.window_sec = float(rospy.get_param("~window_sec", 60.0))
        self.strict_window = bool(rospy.get_param("~strict_window", False))
        self.minimum_range = float(rospy.get_param("~minimum_range", 1.5))
        self.maximum_range = float(rospy.get_param("~maximum_range", 120.0))
        configured_minimum_forward_x = rospy.get_param(
            "~minimum_forward_x", None
        )
        self.minimum_forward_x = (
            None
            if configured_minimum_forward_x in (None, "")
            else float(configured_minimum_forward_x)
        )
        if self.minimum_forward_x is not None and not np.isfinite(
            self.minimum_forward_x
        ):
            raise ValueError("minimum_forward_x must be finite when provided")
        self.gpu_pointcloud = bool(rospy.get_param("~gpu_pointcloud", True))
        self.gpu_semantic_map = bool(rospy.get_param("~gpu_semantic_map", True))
        self.sample_mode = str(rospy.get_param("~sample_mode", "bilinear"))
        self.sample_min_valid_weight = float(
            rospy.get_param("~sample_min_valid_weight", 0.5)
        )
        self.risk_histogram_bins = int(
            rospy.get_param("~risk_histogram_bins", 16)
        )
        self.risk_cvar_alpha = float(
            rospy.get_param("~risk_cvar_alpha", 0.8)
        )
        self.risk_distribution_decay_sec = float(
            rospy.get_param(
                "~risk_distribution_decay_sec", self.window_sec
            )
        )
        self.use_geometry_cvar = _ros_bool(
            rospy.get_param("~use_geometry_cvar", True)
        )
        self.use_semantic_cvar = _ros_bool(
            rospy.get_param("~use_semantic_cvar", True)
        )
        self.gaussian_config = _gaussian_config_from_ros()
        self.lifecycle_config = _lifecycle_config_from_ros()
        self.scan_deskew_mode = str(
            rospy.get_param("~scan_deskew_mode", "off")
        ).strip().lower()
        if self.scan_deskew_mode not in {"off", "auto", "model"}:
            raise ValueError("scan_deskew_mode must be off, auto or model")
        self.scan_stamp_reference = str(
            rospy.get_param("~scan_stamp_reference", "end")
        ).strip().lower()
        self.scan_row_direction = int(
            rospy.get_param("~scan_row_direction", 1)
        )
        self.scan_period_config_sec = float(
            rospy.get_param("~scan_period_sec", 0.1)
        )
        self.scan_period_min_sec = float(
            rospy.get_param("~scan_period_min_sec", 0.08)
        )
        self.scan_period_max_sec = float(
            rospy.get_param("~scan_period_max_sec", 0.12)
        )
        self.scan_period_ema_alpha = float(
            rospy.get_param("~scan_period_ema_alpha", 0.1)
        )
        if not (
            np.isfinite(self.scan_period_config_sec)
            and self.scan_period_min_sec > 0.0
            and self.scan_period_min_sec <= self.scan_period_config_sec
            <= self.scan_period_max_sec
            and 0.0 < self.scan_period_ema_alpha <= 1.0
            and self.scan_stamp_reference in {"start", "center", "end"}
            and self.scan_row_direction in {-1, 1}
        ):
            raise ValueError("invalid scan deskew parameters")
        self.scan_period_estimate_sec = self.scan_period_config_sec
        self.tf_timeout = float(rospy.get_param("~tf_timeout_sec", 0.25))
        self.sync_slop_sec = float(rospy.get_param("~sync_slop_sec", 0.08))
        self.future_stamp_tolerance_sec = float(
            rospy.get_param("~future_stamp_tolerance_sec", 1.0)
        )
        self.semantic_decay_tau_sec = float(
            rospy.get_param("~semantic_decay_tau_sec", 30.0)
        )
        self.semantic_stale_sec = float(
            rospy.get_param("~semantic_stale_sec", 90.0)
        )
        self.device = str(rospy.get_param("~device", "cuda:0"))
        configured_backend = resolve_semantic_backend(
            rospy.get_param("~semantic_backend", "eomt")
        )
        enabled_param = _ros_bool(rospy.get_param("~enable_semantic", True))
        self.semantic_enabled = bool(enabled_param) and configured_backend != "disabled"
        self.semantic_backend = configured_backend if self.semantic_enabled else "disabled"
        self.publish_legacy_probability = _ros_bool(
            rospy.get_param("~publish_legacy_probability", True)
        )
        self.semantic_profile = (
            get_eomt_profile(rospy.get_param("~semantic_profile", "rugd24"))
            if self.semantic_backend == "eomt"
            else None
        )
        checkpoint = rospy.get_param("~rugd_checkpoint", "/models/rugd_best.pth")
        self.output_frame = str(rospy.get_param("~output_frame", self.base_frame))
        square_length = rospy.get_param("~map_length_m", None)
        if square_length is None:
            length_x = rospy.get_param(
                "~map_length_x_m",
                ONLINE_MAP_SPEC.x_max - ONLINE_MAP_SPEC.x_min,
            )
            length_y = rospy.get_param(
                "~map_length_y_m",
                ONLINE_MAP_SPEC.y_max - ONLINE_MAP_SPEC.y_min,
            )
        else:
            length_x = square_length
            length_y = None
        self.map_spec = make_online_map_spec(
            length_x_m=length_x,
            length_y_m=length_y,
            resolution_m=rospy.get_param(
                "~map_resolution_m", ONLINE_MAP_SPEC.resolution
            ),
        )
        self.world_map_length = float(
            rospy.get_param("~world_map_length_m", 80.0)
        )
        output_diagonal = np.hypot(
            self.map_spec.x_max - self.map_spec.x_min,
            self.map_spec.y_max - self.map_spec.y_min,
        )
        if (
            not np.isfinite(self.world_map_length)
            or self.world_map_length < output_diagonal
        ):
            raise ValueError(
                "world_map_length_m must contain the rotated online map"
            )
        self.world_map_spec = MapSpec(
            x_min=-0.5 * self.world_map_length,
            x_max=0.5 * self.world_map_length,
            y_min=-0.5 * self.world_map_length,
            y_max=0.5 * self.world_map_length,
            resolution=self.map_spec.resolution,
        )
        self.publish_world_map = bool(
            rospy.get_param("~publish_world_map", False)
        )
        self.world_map_publish_hz = float(
            rospy.get_param("~world_map_publish_hz", 1.0)
        )
        if not np.isfinite(self.world_map_publish_hz) or self.world_map_publish_hz < 0.0:
            raise ValueError("world_map_publish_hz must be finite and non-negative")
        self.world_map_publish_period = (
            1.0 / self.world_map_publish_hz
            if self.world_map_publish_hz > 0.0
            else np.inf
        )
        self.last_world_map_publish_monotonic = -np.inf
        self.last_traversability_publish_monotonic = None
        self.traversability_publish_interval_ema_sec = None
        self.traversability_publish_hz = 0.0

        # TF buffer 至少覆盖一个在线窗口，并额外预留 10 秒用于回放抖动。
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(self.window_sec + 10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        # DeepLab 只在显式选择 deeplab 时创建；EoMT 模式的模型在另一个 ROS 节点中。
        self.segmenter = initialize_inline_segmenter(
            self.semantic_backend,
            lambda: _load_inline_deeplab_segmenter(checkpoint, device=self.device),
        )
        if self.semantic_backend == "eomt":
            self.semantic_class_count = self.semantic_profile.class_count
            self.semantic_risk = self.semantic_profile.semantic_risk
            self.ground_class_ids = self.semantic_profile.ground_class_ids
            self.dynamic_class_ids = self.semantic_profile.dynamic_class_ids
            self.invalid_class_ids = self.semantic_profile.invalid_class_ids
            self.grass_class_ids = self.semantic_profile.grass_class_ids
        elif self.semantic_backend == "deeplab":
            self.semantic_class_count = len(RUGD_CLASSES)
            self.semantic_risk = SEMANTIC_RISK
            self.ground_class_ids = GROUND_CLASS_IDS
            self.dynamic_class_ids = DYNAMIC_CLASS_IDS
            self.invalid_class_ids = frozenset()
            self.grass_class_ids = frozenset({3})
        else:
            # geometry-only 模式不创建任何语义类别表，避免误把空语义当成有效观测。
            self.semantic_class_count = 0
            self.semantic_risk = {}
            self.ground_class_ids = frozenset()
            self.dynamic_class_ids = frozenset()
            self.invalid_class_ids = frozenset()
            self.grass_class_ids = frozenset()
        # 几何地图保存长期 GPU 状态；非 strict 模式只增量写入新帧，避免每帧重建窗口。
        self.geometry_map = GpuRollingElevationMap(
            map_spec=self.map_spec,
            world_map_length=self.world_map_length,
            device=self.device,
            max_geometry_age_sec=self.window_sec,
            memory_pool_limit_bytes=(
                int(rospy.get_param("~cupy_memory_pool_limit_bytes", 1610612736))
                if self.semantic_backend == "eomt"
                else None
            ),
            sample_mode=self.sample_mode,
            sample_min_valid_weight=self.sample_min_valid_weight,
            risk_histogram_bins=self.risk_histogram_bins,
            risk_cvar_alpha=self.risk_cvar_alpha,
            risk_distribution_decay_sec=self.risk_distribution_decay_sec,
            gaussian_config=self.gaussian_config,
            lifecycle_config=self.lifecycle_config,
        )
        # 该窗口用于记录当前输入帧和 strict_window 重建时的活动帧 ID。
        self.geometry_window = RecentFrameWindow(self.window_sec)
        self.geometry_ingested_ids = set()
        # 语义层复用几何地图的 GPU 滚动中心、索引和边缘清理；显式关闭时
        # 保留原 CPU NumPy 实现用于回归和故障回退。
        if self.semantic_backend == "disabled":
            self.semantic_map = None
        elif self.gpu_semantic_map:
            self.semantic_map = GpuIncrementalSemanticGrid(
                map_spec=self.map_spec,
                semantic_risk=self.semantic_risk,
                grass_class_ids=self.grass_class_ids,
                decay_tau_sec=self.semantic_decay_tau_sec,
                stale_sec=self.semantic_stale_sec,
                rolling_grid=self.geometry_map.rolling_grid,
                sample_mode=self.sample_mode,
                sample_min_valid_weight=self.sample_min_valid_weight,
                risk_histogram_bins=self.risk_histogram_bins,
                risk_cvar_alpha=self.risk_cvar_alpha,
            )
        else:
            self.semantic_map = IncrementalSemanticGrid(
                map_spec=self.map_spec,
                semantic_risk=self.semantic_risk,
                grass_class_ids=self.grass_class_ids,
                decay_tau_sec=self.semantic_decay_tau_sec,
                stale_sec=self.semantic_stale_sec,
                risk_histogram_bins=self.risk_histogram_bins,
                risk_cvar_alpha=self.risk_cvar_alpha,
            )
        self.last_semantic_stats = {
            "semantic_input_points": 0,
            "semantic_aggregated_cells": 0,
            "semantic_valid_cells": 0,
        }
        if not self.semantic_enabled:
            self.last_semantic_stage = "disabled"
        self.geometry_by_id = {}
        self.local_from_world = None
        self.pose_lock = threading.Lock()
        self.semantic_map_lock = threading.Lock()
        self.scan_transform_cache = {}
        self.scan_transform_cache_lock = threading.Lock()
        self.last_timestamp = None
        self.active_epoch = 0
        self.stream_epoch = 0
        self.last_input_stamp_ns = None
        self.stream_lock = threading.Lock()
        self.camera_info_lock = threading.Lock()
        self.pending_camera_sensor = None
        self.camera_info_wait_sec = float(rospy.get_param("~camera_info_wait_sec", 0.5))
        if not np.isfinite(self.camera_info_wait_sec) or self.camera_info_wait_sec <= 0:
            raise ValueError("camera_info_wait_sec must be finite and positive")
        self.camera_info_pending_dropped = 0
        self.camera_info_late_matched = 0
        self.camera_info_cache = BoundedCameraInfoCache(
            max_entries=int(rospy.get_param("~camera_info_cache_entries", 64)),
            tolerance_sec=self.sync_slop_sec,
        )
        # 两条队列都只保留最新工作项；实时系统宁可丢旧帧，也不让延迟无限增长。
        self.geometry_work = queue.Queue(maxsize=1)
        self.semantic_work = queue.Queue(maxsize=1)
        self.matcher_lock = threading.Lock()
        self.semantic_matcher = (
            BoundedSemanticMatcher(
                max_age_sec=float(rospy.get_param("~semantic_cache_sec", 5.0)),
                max_entries=int(rospy.get_param("~semantic_cache_entries", 64)),
            )
            if self.semantic_backend == "eomt"
            else None
        )
        self.eomt_health = (
            EomtHealthMonitor(
                started_monotonic=time.monotonic(),
                readiness_timeout_sec=float(
                    rospy.get_param("~eomt_readiness_timeout_sec", 120.0)
                ),
                staleness_timeout_sec=float(
                    rospy.get_param("~eomt_staleness_timeout_sec", 5.0)
                ),
            )
            if self.semantic_backend == "eomt"
            else None
        )
        self.stop_event = threading.Event()
        self.geometry_worker = threading.Thread(
            target=self._geometry_worker,
            daemon=True,
        )
        self.semantic_worker = threading.Thread(
            target=self._semantic_worker,
            daemon=True,
        )
        # 世界图只用于观察，单独线程异步序列化，不能阻塞车体系实时图。
        self.world_map_work = queue.Queue(maxsize=1)
        self.world_map_worker = threading.Thread(
            target=self._world_map_worker,
            daemon=True,
        )

        self.probability_pub = (
            rospy.Publisher(
                "~traversability_probability", GridMap, queue_size=1
            )
            if self.publish_legacy_probability
            else None
        )
        self.cvar_probability_pub = rospy.Publisher(
            "~cvar_traversability_probability", GridMap, queue_size=1
        )
        self.world_probability_pub = (
            rospy.Publisher(
                "~world_traversability_probability",
                GridMap,
                queue_size=1,
                latch=True,
            )
            if self.publish_world_map and self.publish_legacy_probability
            else None
        )
        self.world_cvar_probability_pub = (
            rospy.Publisher(
                "~world_cvar_traversability_probability",
                GridMap,
                queue_size=1,
                latch=True,
            )
            if self.publish_world_map
            else None
        )
        self.risk_pub = (
            rospy.Publisher("~traversability_risk", Image, queue_size=1)
            if self.publish_legacy_probability
            else None
        )
        self.cvar_risk_pub = rospy.Publisher(
            "~cvar_traversability_risk", Image, queue_size=1
        )
        self.confidence_pub = rospy.Publisher("~confidence", Image, queue_size=1)
        self.elevation_pub = rospy.Publisher("~elevation", Image, queue_size=1)
        self.geometry_risk_std_pub = (
            rospy.Publisher("~geometry_risk_std", Image, queue_size=1)
            if self.publish_legacy_probability
            else None
        )
        self.geometry_risk_cvar_pub = rospy.Publisher(
            "~geometry_risk_cvar", Image, queue_size=1
        )
        self.semantic_risk_std_pub = (
            rospy.Publisher("~semantic_risk_std", Image, queue_size=1)
            if self.publish_legacy_probability
            else None
        )
        self.semantic_risk_cvar_pub = rospy.Publisher(
            "~semantic_risk_cvar", Image, queue_size=1
        )
        self.diagnostics_pub = rospy.Publisher(
            "~diagnostics", DiagnosticArray, queue_size=1
        )
        camera_info_topic = rospy.get_param(
            "~camera_info_topic", "/sensor/camera/windshield/vis/camera_info"
        )
        rospy.Subscriber(
            camera_info_topic, CameraInfo, self._camera_info_callback, queue_size=1
        )
        image_topic = rospy.get_param(
            "~image_topic", "/sensor/camera/windshield/vis/image_rect_color"
        )
        cloud_topic = rospy.get_param(
            "~cloud_topic", "/sensor/lidar/vls128_roof/velodyne_points"
        )
        image_sub = message_filters.Subscriber(image_topic, Image, queue_size=2)
        cloud_sub = message_filters.Subscriber(cloud_topic, PointCloud2, queue_size=2)
        synchronizer = message_filters.ApproximateTimeSynchronizer(
            [image_sub, cloud_sub],
            queue_size=int(rospy.get_param("~sync_queue_size", 20)),
            slop=self.sync_slop_sec,
        )
        synchronizer.registerCallback(self._synchronized_callback)
        self.image_sub = image_sub
        self.cloud_sub = cloud_sub
        self.synchronizer = synchronizer
        if self.semantic_backend == "eomt":
            prediction_topic = rospy.get_param(
                "~eomt_prediction_topic",
                "/roadrunner_semantic/prediction",
            )
            ready_topic = rospy.get_param(
                "~eomt_ready_topic",
                "/roadrunner_semantic/ready",
            )
            self.prediction_sub = rospy.Subscriber(
                prediction_topic,
                EomtSemanticPrediction,
                self._prediction_callback,
                queue_size=2,
            )
            self.ready_sub = rospy.Subscriber(
                ready_topic,
                Bool,
                self._ready_callback,
                queue_size=1,
            )
        self.received = 0
        self.processed = 0
        self.dropped = 0
        self.geometry_epoch_discards = 0
        self.semantic_processed = 0
        self.semantic_dropped = 0
        self.semantic_failures = 0
        self.semantic_epoch_discards = 0
        self.gpu_decode_fallbacks = 0
        self.scan_deskew_applied = 0
        self.scan_deskew_fallbacks = 0
        self.failures = 0
        self.last_stage = "waiting_for_synchronized_input"
        self.last_semantic_stage = "waiting_for_semantic_input"
        self.last_error = ""
        self.last_semantic_error = ""
        self.last_timing = {}
        self.last_semantic_timing = {}
        self.heartbeat_worker = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
        )
        # 几何、语义和心跳线程独立运行，语义推理阻塞时不会阻塞几何发布。
        self.geometry_worker.start()
        if self.semantic_enabled:
            self.semantic_worker.start()
        self.world_map_worker.start()
        self.heartbeat_worker.start()
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo(
            "GOOSE online traversability ready: backend=%s device=%s "
            "window=%.1fs strict=%s",
            self.semantic_backend,
            self.device,
            self.window_sec,
            self.strict_window,
        )

    def _camera_info_callback(self, message: CameraInfo) -> None:
        ready = None
        with self.camera_info_lock:
            self.camera_info_cache.add(stamp_to_nsec(message.header.stamp), message)
            pending = self.pending_camera_sensor
            if pending is not None:
                epoch, image, cloud, queued_at = pending
                if time.monotonic() - queued_at >= self.camera_info_wait_sec:
                    self.pending_camera_sensor = None
                    self.camera_info_pending_dropped += 1
                else:
                    info = self.camera_info_cache.match(stamp_to_nsec(image.header.stamp))
                    if info is not None:
                        self.pending_camera_sensor = None
                        self.camera_info_late_matched += 1
                        ready = (epoch, image, cloud, info)
        if ready is not None:
            self._submit_semantic_sensor(*ready)

    def _synchronized_callback(self, image: Image, cloud: PointCloud2) -> None:
        """接收一对近似同步的 RGB/点云，并拆分到几何、语义两条队列。

        80 ms 只约束“当前图像和当前点云”的对应关系，不限制后续语义教师
        的历史窗口。几何项先入队；CameraInfo 或 EoMT 缺失只能影响语义项。
        """
        self.received += 1
        rospy.loginfo_once("received first synchronized GOOSE image/cloud pair")
        image_stamp_ns = stamp_to_nsec(image.header.stamp)
        # 仿真时间倒退通常意味着 rosbag 被重新播放。epoch 让延迟到达的旧
        # 语义预测在写入前可以被识别并丢弃。
        with self.stream_lock:
            rewind = (
                self.last_input_stamp_ns is not None
                and image_stamp_ns < self.last_input_stamp_ns
            )
            if rewind:
                self.stream_epoch += 1
            self.last_input_stamp_ns = image_stamp_ns
            epoch = self.stream_epoch
        if rewind and self.semantic_matcher is not None:
            with self.matcher_lock:
                self.semantic_matcher.clear()

        # 几何是主时钟：这里不等待 CameraInfo、EoMT 或语义投影。
        self._enqueue_geometry((epoch, cloud))
        if not self.semantic_enabled:
            return
        with self.camera_info_lock:
            if self.pending_camera_sensor is not None:
                self.camera_info_pending_dropped += 1
                self.pending_camera_sensor = None
            camera_info = self.camera_info_cache.match(image_stamp_ns)
            if camera_info is None:
                # Only the latest sensor pair waits; geometry was already queued.
                self.pending_camera_sensor = (epoch, image, cloud, time.monotonic())
                return
        self._submit_semantic_sensor(epoch, image, cloud, camera_info)

    def _submit_semantic_sensor(self, epoch, image, cloud, camera_info) -> None:
        with self.stream_lock:
            if epoch != self.stream_epoch:
                return
        image_stamp_ns = stamp_to_nsec(image.header.stamp)
        sensor_item = (epoch, image, cloud, camera_info)
        # DeepLab 在融合进程内推理；EoMT 则先把传感器项放入纳秒时间戳匹配器。
        if self.semantic_backend == "deeplab":
            self._enqueue_semantic(sensor_item + (None,))
            return
        if self.eomt_health.fatal_reason:
            return
        now = time.monotonic()
        with self.matcher_lock:
            match = self.semantic_matcher.add_sensor(
                image_stamp_ns,
                sensor_item,
                now=now,
            )
        if match is not None:
            sensor, prediction = match
            self._enqueue_semantic(sensor + (prediction,))

    def _ready_callback(self, message: Bool) -> None:
        if message.data:
            self.eomt_health.mark_ready(now=time.monotonic())

    def _prediction_callback(self, message: EomtSemanticPrediction) -> None:
        """校验 EoMT 原子预测，并与等待中的传感器项按时间戳配对。"""
        if self.eomt_health.fatal_reason:
            return
        try:
            prediction = validate_atomic_prediction(
                message,
                class_count=self.semantic_class_count,
            )
        except PredictionValidationError as error:
            self.eomt_health.fail(error.reason)
            self.last_semantic_error = str(error)
            rospy.logerr("invalid EoMT prediction: %s", error)
            return
        now = time.monotonic()
        with self.matcher_lock:
            duplicate_before = self.semantic_matcher.counters[
                "duplicate_prediction"
            ]
            match = self.semantic_matcher.add_prediction(
                prediction.stamp_ns,
                prediction,
                now=now,
            )
            duplicate = (
                self.semantic_matcher.counters["duplicate_prediction"]
                != duplicate_before
            )
        if duplicate:
            return
        self.eomt_health.mark_output(now=now)
        if match is not None:
            sensor, matched_prediction = match
            self._enqueue_semantic(sensor + (matched_prediction,))

    @staticmethod
    def _put_latest(work_queue, item) -> bool:
        """向容量为 1 的队列写入最新项，返回是否替换了旧项。"""
        try:
            work_queue.put_nowait(item)
            return False
        except queue.Full:
            try:
                work_queue.get_nowait()
                work_queue.task_done()
            except queue.Empty:
                pass
            work_queue.put_nowait(item)
            return True

    def _enqueue_geometry(self, item) -> None:
        if self._put_latest(self.geometry_work, item):
            self.dropped += 1

    def _enqueue_semantic(self, item) -> None:
        if self._put_latest(self.semantic_work, item):
            self.semantic_dropped += 1

    def _lookup(self, target: str, source: str, stamp) -> np.ndarray:
        """查询指定时间的 TF，并统一转换为 NumPy 齐次矩阵。"""
        message = self.tf_buffer.lookup_transform(
            target, source, stamp, rospy.Duration(self.tf_timeout)
        )
        return transform_matrix(message)

    def _local_pose(self, stamp) -> np.ndarray:
        """获取相对于首帧的局部世界位姿。

        RT3000 的原始世界坐标可能很大；首个有效位姿的逆矩阵被保存为
        ``local_from_world``，后续计算在局部世界系中进行，减少浮点误差。
        """
        world_from_body = self._lookup(self.world_frame, self.base_frame, stamp)
        with self.pose_lock:
            if self.local_from_world is None:
                self.local_from_world = np.linalg.inv(world_from_body)
            local_from_world = self.local_from_world.copy()
        return local_from_world @ world_from_body

    def _calibration(self, stamp):
        """获取图像时刻的 LiDAR→相机和 base_link→LiDAR 外参。"""
        camera_from_lidar = self._lookup(self.camera_frame, self.lidar_frame, stamp)
        lidar_from_base = self._lookup(self.lidar_frame, self.base_frame, stamp)
        return camera_from_lidar, lidar_from_base

    def _reset_stream(self, *, semantic_lock_held: bool = False) -> None:
        """清空一轮 rosbag 回放的所有时序状态。"""
        self.geometry_map.clear()
        self.geometry_window.clear()
        # frame ID 是节点生命周期内的全局幂等键。流/时间重置只清活动
        # 地图与窗口，不遗忘已成功融合的 ID，避免旧帧在新 epoch 再次进入。
        if self.semantic_map is not None:
            if semantic_lock_held:
                self.semantic_map.clear()
            else:
                with self.semantic_map_lock:
                    self.semantic_map.clear()
        self.geometry_by_id.clear()
        with self.pose_lock:
            self.local_from_world = None
        self.last_timestamp = None
        self.scan_period_estimate_sec = self.scan_period_config_sec
        with self.scan_transform_cache_lock:
            self.scan_transform_cache.clear()

    def _update_scan_period(self, timestamp: float, previous_timestamp) -> None:
        """Estimate the VLS scan period from adjacent cloud header stamps."""

        if previous_timestamp is None:
            return
        period = float(timestamp) - float(previous_timestamp)
        if not self.scan_period_min_sec <= period <= self.scan_period_max_sec:
            return
        alpha = self.scan_period_ema_alpha
        self.scan_period_estimate_sec = (
            (1.0 - alpha) * self.scan_period_estimate_sec + alpha * period
        )

    def _scan_row_world_from_lidar(
        self,
        cloud_stamp,
        cloud_pose: np.ndarray,
        lidar_from_base: np.ndarray,
        *,
        height: int,
        scan_period_sec: float | None = None,
        local_from_world: np.ndarray | None = None,
    ):
        """Build one world←LiDAR pose per organized VLS scan row.

        The rosbag has no ``time`` field, so this path is explicitly model based:
        row order and scan period are configurable, while the endpoint poses come
        from TF.  Any TF extrapolation or malformed layout falls back to the
        proven rigid whole-frame path for that frame.
        """

        if self.scan_deskew_mode == "off" or int(height) <= 1:
            return None
        if scan_period_sec is None:
            scan_period_sec = self.scan_period_estimate_sec
        offsets = scan_row_relative_times(
            int(height),
            scan_period_sec=scan_period_sec,
            stamp_reference=self.scan_stamp_reference,
            direction=self.scan_row_direction,
        )
        timestamp = float(cloud_stamp.to_sec())
        cache_key = (
            int(cloud_stamp.to_nsec()),
            int(height),
            round(scan_period_sec, 7),
        )
        with self.scan_transform_cache_lock:
            cached = self.scan_transform_cache.get(cache_key)
        if cached is not None:
            return cached
        start_offset = float(np.min(offsets))
        end_offset = float(np.max(offsets))

        def pose_at(offset: float) -> np.ndarray:
            if abs(offset) <= 1e-9:
                return np.asarray(cloud_pose, dtype=np.float64)
            if local_from_world is None:
                return self._local_pose(rospy.Time.from_sec(timestamp + offset))
            world_from_body = self._lookup(
                self.world_frame,
                self.base_frame,
                rospy.Time.from_sec(timestamp + offset),
            )
            return local_from_world @ world_from_body

        try:
            start_pose = pose_at(start_offset)
            end_pose = pose_at(end_offset)
            denominator = end_offset - start_offset
            ratios = (offsets - start_offset) / denominator
            world_from_body = interpolate_scan_transforms(
                start_pose,
                end_pose,
                ratios,
            )
            base_from_lidar = np.linalg.inv(
                np.asarray(lidar_from_base, dtype=np.float64).reshape(4, 4)
            )
            self.scan_deskew_applied += 1
            result = world_from_body @ base_from_lidar
            with self.scan_transform_cache_lock:
                self.scan_transform_cache[cache_key] = result
                if len(self.scan_transform_cache) > 8:
                    self.scan_transform_cache.pop(next(iter(self.scan_transform_cache)))
            return result
        except Exception as error:
            self.scan_deskew_fallbacks += 1
            rospy.logwarn_throttle(
                5.0,
                "VLS scan deskew unavailable; using rigid frame pose: %s",
                error,
            )
            return None

    @staticmethod
    def _geometry_teacher(maps):
        """把 GPU 高程图的内部字段映射为统一教师字段名。"""
        shape = maps["geometry_risk"].shape
        zeros = np.zeros(shape, dtype=np.float32)
        return {
            "geometry_hindsight_target": maps["geometry_risk"],
            "geometry_hindsight_valid_mask": maps["geometry_valid_mask"],
            "geometry_hindsight_confidence": maps["geometry_confidence"],
            "geometry_hindsight_observation_count": maps[
                "geometry_support_count"
            ],
            "geometry_hindsight_elevation": maps["elevation"],
            "geometry_hindsight_slope_risk": maps["slope_risk"],
            "geometry_hindsight_positive_step_risk": maps[
                "positive_step_risk"
            ],
            "geometry_hindsight_negative_step_risk": maps[
                "negative_step_risk"
            ],
            "geometry_hindsight_non_ground_ratio": maps["non_ground_ratio"],
            "geometry_hindsight_risk_std": maps.get(
                "geometry_risk_std", zeros.copy()
            ),
            "geometry_hindsight_risk_cvar": maps.get(
                "geometry_risk_cvar", zeros.copy()
            ),
            "geometry_hindsight_risk_distribution_weight": maps.get(
                "geometry_risk_distribution_weight", zeros.copy()
            ),
            "geometry_hindsight_risk_distribution_valid": maps.get(
                "geometry_risk_distribution_valid",
                maps["geometry_valid_mask"],
            ),
        }

    def _empty_semantic_snapshot(self) -> dict[str, np.ndarray]:
        """返回 geometry-only 模式使用的全无效语义快照。"""

        shape = self.map_spec.shape
        zeros = np.zeros(shape, dtype=np.float32)
        return {
            "semantic_hindsight_target": zeros.copy(),
            "semantic_hindsight_valid_mask": np.zeros(shape, dtype=bool),
            "semantic_hindsight_confidence": zeros.copy(),
            "semantic_hindsight_observation_count": np.zeros(
                shape, dtype=np.int32
            ),
            "semantic_hindsight_risk_std": zeros.copy(),
            "semantic_hindsight_risk_cvar": zeros.copy(),
            "semantic_hindsight_risk_distribution_valid": np.zeros(
                shape, dtype=bool
            ),
            "grass_probability": zeros.copy(),
            "grass_observation_weight": zeros.copy(),
            "grass_mask": np.zeros(shape, dtype=bool),
            "semantic_input_points": 0,
            "semantic_aggregated_cells": 0,
            "semantic_valid_cells": 0,
        }

    def _process_geometry(self, epoch, cloud_message) -> None:
        """处理一帧点云并发布当前车体系地图。

        处理顺序固定为：解码/时间检查 → TF → 世界坐标变换 → 持久 GPU 高程图
        → 语义栅格当前快照 → 草地规则与风险融合 → GridMap 发布。
        """
        epoch = int(epoch)
        if epoch < self.active_epoch:
            return
        cloud_stamp = cloud_message.header.stamp
        frame_id = "{}:{}".format(
            int(cloud_stamp.to_nsec()),
            int(getattr(cloud_message.header, "seq", 0)),
        )
        timestamp = float(cloud_stamp.to_sec())
        # frame ID 是跨 epoch 的全局幂等键。重复帧必须在任何新流重置或
        # timestamp 门禁前返回，否则 rosbag 边界重复消息会清空有效地图。
        if frame_id in self.geometry_ingested_ids:
            return

        if epoch > self.active_epoch:
            def process_new_epoch():
                previous_stream = _snapshot_online_state(self)
                try:
                    self.last_stage = "stream_reset"
                    self._reset_stream(
                        semantic_lock_held=self.semantic_map is not None
                    )
                    self.active_epoch = epoch
                    self._process_geometry_active_epoch(
                        cloud_message,
                        cloud_stamp,
                        frame_id,
                        timestamp,
                        semantic_lock_held=self.semantic_map is not None,
                    )
                except Exception:
                    _restore_online_state(self, previous_stream)
                    raise

            if self.semantic_map is None:
                process_new_epoch()
            else:
                # 全局锁顺序固定为 semantic_map_lock -> rolling_grid.lock。
                # 从快照到提交/回滚始终阻止并发语义更新进入事务中间态。
                with self.semantic_map_lock:
                    process_new_epoch()
            return

        self._process_geometry_active_epoch(
            cloud_message,
            cloud_stamp,
            frame_id,
            timestamp,
        )

    def _process_geometry_active_epoch(
        self,
        cloud_message,
        cloud_stamp,
        frame_id: str,
        timestamp: float,
        semantic_lock_held: bool = False,
    ) -> None:
        """处理已通过 epoch 与全局重复 ID 门禁的候选帧。"""
        started = time.perf_counter()
        timings = {}
        time_action = _geometry_frame_time_action(
            frame_id,
            timestamp,
            self.geometry_ingested_ids,
            self.last_timestamp,
        )
        # 同一消息可能因同步器或 rosbag 边界重复到达。已融合 ID 必须在
        # epoch/时间回退处理前直接忽略，避免重复帧意外清空滚动状态。
        if time_action == "duplicate":
            return

        self.last_stage = "decode"
        stage_started = time.perf_counter()
        previous_timestamp = self.last_timestamp
        if timestamp_is_future(
            timestamp,
            float(rospy.Time.now().to_sec()),
            tolerance_sec=self.future_stamp_tolerance_sec,
        ):
            self.geometry_epoch_discards += 1
            return
        if time_action == "rewind":
            rospy.logwarn("ROS time moved backwards; clearing online rolling maps")
            self._reset_stream()
            previous_timestamp = None
        candidate_scan_period = self.scan_period_estimate_sec
        if previous_timestamp is not None:
            observed_period = timestamp - float(previous_timestamp)
            if self.scan_period_min_sec <= observed_period <= self.scan_period_max_sec:
                alpha = self.scan_period_ema_alpha
                candidate_scan_period = (
                    (1.0 - alpha) * candidate_scan_period
                    + alpha * observed_period
                )
        # GOOSE 点云没有可靠逐点时间字段；scan_deskew_mode=auto/model 时，
        # 后面会按组织化 VLS 行和 TF 端点构造显式的模型去畸变。
        # 默认非 strict 在线路径在 GPU 上完成字节解码；strict 窗口或不支持的
        # 字段布局保留 CPU 回退，确保旧的精确窗口路径仍可用。
        gpu_cloud = None
        cloud = None
        cloud_rows = None
        if self.gpu_pointcloud and not self.strict_window:
            try:
                gpu_cloud = decode_pointcloud2_gpu(
                    cloud_message,
                    cp=self.geometry_map.cp,
                )
            except Exception as error:
                self.gpu_decode_fallbacks += 1
                rospy.logwarn_throttle(
                    5.0,
                    "GPU PointCloud2 decode unavailable; using CPU fallback: %s",
                    error,
                )
        if gpu_cloud is None:
            if self.scan_deskew_mode == "off":
                cloud = decode_goose_cloud(
                    cloud_message,
                    minimum_range=self.minimum_range,
                    maximum_range=self.maximum_range,
                    minimum_forward_x=self.minimum_forward_x,
                )
            else:
                cloud, cloud_rows = decode_goose_cloud_with_rows(
                    cloud_message,
                    minimum_range=self.minimum_range,
                    maximum_range=self.maximum_range,
                    minimum_forward_x=self.minimum_forward_x,
                )
        timings["decode_sec"] = time.perf_counter() - stage_started
        if timestamp_is_future(
            timestamp,
            float(rospy.Time.now().to_sec()),
            tolerance_sec=self.future_stamp_tolerance_sec,
        ):
            self.geometry_epoch_discards += 1
            return

        self.last_stage = "tf_lookup"
        stage_started = time.perf_counter()
        world_from_body = self._lookup(
            self.world_frame,
            self.base_frame,
            cloud_stamp,
        )
        with self.pose_lock:
            candidate_local_from_world = (
                np.linalg.inv(world_from_body)
                if self.local_from_world is None
                else self.local_from_world.copy()
            )
        cloud_pose = candidate_local_from_world @ world_from_body
        lidar_from_base = self._lookup(
            self.lidar_frame,
            self.base_frame,
            cloud_stamp,
        )
        timings["tf_lookup_sec"] = time.perf_counter() - stage_started

        self.last_stage = "window_update"
        stage_started = time.perf_counter()
        base_from_lidar = np.linalg.inv(lidar_from_base)
        world_from_lidar = cloud_pose @ base_from_lidar
        deskew_started = time.perf_counter()
        row_world_from_lidar = self._scan_row_world_from_lidar(
            cloud_stamp,
            cloud_pose,
            lidar_from_base,
            height=int(cloud_message.height),
            scan_period_sec=candidate_scan_period,
            local_from_world=candidate_local_from_world,
        )
        timings["scan_deskew_sec"] = time.perf_counter() - deskew_started
        if gpu_cloud is None:
            # CPU fallback and strict mode retain the original observation shape.
            height_variance = _lidar_height_variance(
                cloud[:, :3], self.gaussian_config
            )
            if row_world_from_lidar is None or cloud_rows is None:
                xyz_world = transform_points(cloud[:, :3], world_from_lidar)
            else:
                xyz_world = apply_row_transforms(
                    cloud[:, :3],
                    cloud_rows,
                    row_world_from_lidar,
                )
            observation = SimpleNamespace(
                xyz=xyz_world,
                intensity=cloud[:, 3],
                ground_probability=np.ones(xyz_world.shape[0], dtype=np.float32),
                dynamic=np.zeros(xyz_world.shape[0], dtype=bool),
                height_variance=height_variance,
                relative_points_lidar=cloud[:, :3].copy(),
            )
            geometry_frame = SimpleNamespace(
                frame_id=frame_id,
                timestamp=timestamp,
                world_from_body=cloud_pose,
                sensor_origin_world=world_from_lidar[:3, 3].copy(),
                observation_world=observation,
            )
        else:
            # The rolling GPU map only needs the timestamp for active-frame
            # bookkeeping; points stay on the device until voxel accumulation.
            geometry_frame = SimpleNamespace(
                frame_id=frame_id,
                timestamp=timestamp,
                world_from_body=cloud_pose,
                sensor_origin_world=world_from_lidar[:3, 3].copy(),
                observation_world=None,
            )
        if self.strict_window:
            cutoff = timestamp - self.geometry_window.duration_sec
            active_frames = tuple(
                frame
                for frame in self.geometry_window
                if float(frame.timestamp) >= cutoff
            ) + (geometry_frame,)
            has_expired = len(active_frames) - 1 < len(self.geometry_window)
        else:
            active_frames = (geometry_frame,)
            has_expired = False
        timings["window_update_sec"] = time.perf_counter() - stage_started

        self.last_stage = "geometry_gpu"
        geometry_started = time.perf_counter()
        geometry_committed = False

        def commit_geometry_candidate():
            expired = self.geometry_window.append(geometry_frame)
            self.geometry_by_id[frame_id] = geometry_frame
            for old in expired:
                self.geometry_by_id.pop(old.frame_id, None)
            with self.pose_lock:
                self.local_from_world = candidate_local_from_world
            self.scan_period_estimate_sec = candidate_scan_period
            self.last_timestamp = timestamp
            self.geometry_ingested_ids.add(frame_id)

        # 默认模式只增量摄入当前帧；strict 模式在窗口过期时重放剩余帧，
        # 以换取更精确的时间窗口，但会产生一次性计算尖峰。
        if gpu_cloud is not None:
            point_rows = None
            if row_world_from_lidar is not None:
                point_rows = self.geometry_map.cp.repeat(
                    self.geometry_map.cp.arange(
                        int(gpu_cloud.height),
                        dtype=self.geometry_map.cp.int32,
                    ),
                    int(gpu_cloud.width),
                )
            maps = self.geometry_map.advance_pointcloud(
                frame_id,
                timestamp,
                gpu_cloud.xyz,
                gpu_cloud.intensity,
                world_from_lidar=world_from_lidar,
                minimum_range=self.minimum_range,
                maximum_range=self.maximum_range,
                minimum_forward_x=self.minimum_forward_x,
                target_world_from_body=cloud_pose,
                row_world_from_lidar=row_world_from_lidar,
                point_rows=point_rows,
            )
        elif has_expired and self.strict_window:
            # strict 窗口过期会先清空并重放整张图。仅这个低频分支备份
            # 全部注册层（包含共享语义层），保证任意一帧失败都可原样恢复。
            def rebuild_strict_window():
                rebuild_snapshot = _snapshot_online_state(self)
                try:
                    self.geometry_map.clear()
                    active_ids = tuple(
                        frame.frame_id for frame in active_frames
                    )

                    def load_active_frame(active_frame_id):
                        if active_frame_id == frame_id:
                            return geometry_frame
                        return self.geometry_by_id[active_frame_id]

                    rebuilt_maps = self.geometry_map.advance(
                        active_ids,
                        load_active_frame,
                        target_world_from_body=cloud_pose,
                    )
                    commit_geometry_candidate()
                    return rebuilt_maps
                except Exception:
                    _restore_online_state(self, rebuild_snapshot)
                    raise

            if self.semantic_map is None or semantic_lock_held:
                maps = rebuild_strict_window()
            else:
                # 与语义更新保持同一锁顺序，并覆盖快照至提交/回滚全过程。
                with self.semantic_map_lock:
                    maps = rebuild_strict_window()
            geometry_committed = True
        else:
            maps = self.geometry_map.advance(
                (frame_id,),
                lambda _: geometry_frame,
                target_world_from_body=cloud_pose,
            )
        if not geometry_committed:
            commit_geometry_candidate()
        timings["geometry_sec"] = time.perf_counter() - geometry_started
        geometry_timing = self.geometry_map.last_geometry_timing
        timings.update(
            (name, float(geometry_timing.get(name, 0.0)))
            for name in _GEOMETRY_DIAGNOSTIC_TIMING_NAMES
        )
        timings["geometry_profile_gap_sec"] = max(
            0.0,
            timings["geometry_sec"]
            - sum(
                timings[name] for name in _GEOMETRY_TOP_LEVEL_TIMING_NAMES
            ),
        )

        self.last_stage = "semantic_grid"
        stage_started = time.perf_counter()
        # 语义可能尚未到达；没有语义节点时直接使用全无效快照，融合严格退化为几何结果。
        if self.semantic_map is None:
            semantic = self._empty_semantic_snapshot()
        elif semantic_lock_held:
            semantic = self.semantic_map.sample(
                timestamp=timestamp,
                world_from_body=cloud_pose,
            )
        else:
            with self.semantic_map_lock:
                semantic = self.semantic_map.sample(
                    timestamp=timestamp,
                    world_from_body=cloud_pose,
                )
        self.last_semantic_stats["semantic_valid_cells"] = int(
            semantic["semantic_valid_cells"]
        )
        timings["semantic_grid_sec"] = time.perf_counter() - stage_started

        self.last_stage = "risk_fusion"
        stage_started = time.perf_counter()
        geometry = self._geometry_teacher(maps)
        geometry = apply_grass_rule_to_accumulated_geometry(geometry, semantic)
        if self.publish_legacy_probability:
            legacy_geometry, cvar_geometry = geometry_risk_variants(
                geometry,
                use_cvar=self.use_geometry_cvar,
            )
            legacy_semantic, cvar_semantic = semantic_risk_variants(
                semantic,
                use_cvar=self.use_semantic_cvar,
            )
            fused = fuse_accumulated_geometry_semantics(
                legacy_geometry, legacy_semantic
            )
        else:
            cvar_geometry = geometry_cvar_risk(
                geometry,
                use_cvar=self.use_geometry_cvar,
            )
            cvar_semantic = semantic_cvar_risk(
                semantic,
                use_cvar=self.use_semantic_cvar,
            )
            legacy_geometry = None
            legacy_semantic = None
            fused = None
        cvar_fused = fuse_accumulated_geometry_semantics(
            cvar_geometry, cvar_semantic
        )
        timings["risk_fusion_sec"] = time.perf_counter() - stage_started

        self.last_stage = "publish"
        stage_started = time.perf_counter()
        self._publish_maps(
            cloud_stamp,
            fused,
            cvar_fused,
            maps,
            legacy_geometry,
            cvar_geometry,
            legacy_semantic,
            cvar_semantic,
        )
        _record_successful_traversability_publish(self)
        timings["publish_sec"] = time.perf_counter() - stage_started
        timings["total_sec"] = time.perf_counter() - started
        timed_sum = sum(
            value
            for name, value in timings.items()
            if name.endswith("_sec")
            and name != "total_sec"
            and name not in _GEOMETRY_DIAGNOSTIC_TIMING_NAMES
            and name != "geometry_profile_gap_sec"
        )
        timings["unaccounted_sec"] = max(0.0, timings["total_sec"] - timed_sum)
        self.last_timing = timings
        self.processed += 1
        self.last_stage = "idle"
        self.last_error = ""
        self._publish_diagnostics(cloud_stamp)

    def _process_semantic(
        self,
        epoch,
        image_message,
        cloud_message,
        camera_info,
        semantic_prediction,
    ) -> None:
        """将一帧延迟语义预测投影并写入世界系增量语义栅格。"""
        if not self.semantic_enabled or self.semantic_map is None:
            return
        epoch = int(epoch)
        if epoch != self.active_epoch:
            self.semantic_epoch_discards += 1
            return
        started = time.perf_counter()
        timings = {}
        self.last_semantic_stage = "decode"
        stage_started = time.perf_counter()
        image_stamp = image_message.header.stamp
        cloud_stamp = cloud_message.header.stamp
        timestamp = float(cloud_stamp.to_sec())
        if timestamp_is_future(
            timestamp,
            float(rospy.Time.now().to_sec()),
            tolerance_sec=self.future_stamp_tolerance_sec,
        ):
            self.semantic_epoch_discards += 1
            return
        cloud_rows = None
        if self.scan_deskew_mode == "off":
            cloud = decode_goose_cloud(
                cloud_message,
                minimum_range=self.minimum_range,
                maximum_range=self.maximum_range,
                minimum_forward_x=self.minimum_forward_x,
            )
        else:
            cloud, cloud_rows = decode_goose_cloud_with_rows(
                cloud_message,
                minimum_range=self.minimum_range,
                maximum_range=self.maximum_range,
                minimum_forward_x=self.minimum_forward_x,
            )
        timings["semantic_decode_sec"] = time.perf_counter() - stage_started
        if timestamp_is_future(
            timestamp,
            float(rospy.Time.now().to_sec()),
            tolerance_sec=self.future_stamp_tolerance_sec,
        ):
            self.semantic_epoch_discards += 1
            return

        self.last_semantic_stage = "tf_lookup"
        stage_started = time.perf_counter()
        cloud_pose = self._local_pose(cloud_stamp)
        image_pose = self._local_pose(image_stamp)
        camera_from_lidar, lidar_from_base = self._calibration(image_stamp)
        base_from_lidar = np.linalg.inv(lidar_from_base)
        timings["semantic_tf_lookup_sec"] = time.perf_counter() - stage_started

        self.last_semantic_stage = "cloud_alignment"
        stage_started = time.perf_counter()
        # 当前点云和图像可能有几十毫秒时差。若组织化 VLS 行可用，先在
        # 世界系按扫描内运动补偿，再统一变换到图像时刻的 LiDAR 系；否则
        # 回退到原有的整帧刚体对齐。
        row_world_from_lidar = self._scan_row_world_from_lidar(
            cloud_stamp,
            cloud_pose,
            lidar_from_base,
            height=int(cloud_message.height),
        )
        if row_world_from_lidar is not None and cloud_rows is not None:
            image_lidar_from_world = lidar_from_base @ np.linalg.inv(image_pose)
            row_image_from_lidar = (
                image_lidar_from_world[None, ...] @ row_world_from_lidar
            )
            aligned = apply_row_transforms(
                cloud,
                cloud_rows,
                row_image_from_lidar,
            ).astype(np.float32, copy=False)
        else:
            aligned = rigid_align_cloud_to_image(
                cloud,
                world_from_cloud_body=cloud_pose,
                world_from_image_body=image_pose,
                lidar_from_base=lidar_from_base,
            )
        timings["semantic_cloud_alignment_sec"] = (
            time.perf_counter() - stage_started
        )

        if self.semantic_backend == "eomt":
            if semantic_prediction is None:
                raise ValueError("EoMT prediction is required")
            labels = semantic_prediction.labels
            semantic_confidence = semantic_prediction.confidence
            expected_shape = (int(image_message.height), int(image_message.width))
            if labels.shape != expected_shape:
                self.eomt_health.fail("prediction_image_dimension_mismatch")
                raise ValueError(
                    "EoMT prediction does not match source image dimensions"
                )
            timings["semantic_inference_sec"] = semantic_prediction.inference_sec
        else:
            self.last_semantic_stage = "semantic_gpu"
            semantic_started = time.perf_counter()
            image_rgb = image_message_to_rgb(image_message)
            labels, semantic_confidence = self.segmenter.predict(image_rgb)
            timings["semantic_inference_sec"] = (
                time.perf_counter() - semantic_started
            )

        self.last_semantic_stage = "projection"
        stage_started = time.perf_counter()
        # 投影输出包含 labels、confidence、valid、dynamic，只有 valid 点会
        # 进入语义栅格；原始点仍按 cloud_pose 写入世界坐标用于关联。
        projected = project_semantics_to_points(
            aligned[:, :3],
            labels,
            semantic_confidence,
            rectified_intrinsics(
                camera_info,
                image_shape=(
                    int(image_message.height),
                    int(image_message.width),
                    3,
                ),
                camera_frame=self.camera_frame,
            ),
            camera_from_lidar=camera_from_lidar,
            class_count=self.semantic_class_count,
            ground_class_ids=self.ground_class_ids,
            dynamic_class_ids=self.dynamic_class_ids,
            invalid_class_ids=self.invalid_class_ids,
        )
        if row_world_from_lidar is None or cloud_rows is None:
            xyz_world = transform_points(cloud[:, :3], cloud_pose @ base_from_lidar)
        else:
            xyz_world = apply_row_transforms(
                cloud[:, :3],
                cloud_rows,
                row_world_from_lidar,
            )
        timings["semantic_projection_sec"] = time.perf_counter() - stage_started

        if epoch != self.active_epoch:
            self.semantic_epoch_discards += 1
            return
        self.last_semantic_stage = "grid_update"
        stage_started = time.perf_counter()
        with self.semantic_map_lock:
            stats = self.semantic_map.update(
                timestamp=timestamp,
                world_from_body=cloud_pose,
                xyz_world=xyz_world,
                labels=projected.labels,
                confidence=projected.confidence,
                valid=projected.valid,
                dynamic=projected.dynamic,
            )
        self.last_semantic_stats.update(stats)
        timings["semantic_grid_update_sec"] = time.perf_counter() - stage_started
        timings["semantic_total_sec"] = time.perf_counter() - started
        self.last_semantic_timing = timings
        self.semantic_processed += 1
        self.last_semantic_stage = "idle"
        self.last_semantic_error = ""

    def _diagnostic_status(self, **timings):
        """组合几何、语义、队列和 EoMT 健康信息为 DiagnosticStatus。"""
        level = DiagnosticStatus.OK if self.processed else DiagnosticStatus.WARN
        if self.processed:
            message = (
                "geometry-only online map active"
                if not self.semantic_enabled
                else "geometry-led online map active"
            )
        else:
            message = "waiting for first completed online map"
        semantic_problem = self.last_semantic_error
        if self.eomt_health is not None:
            semantic_problem = (
                self.eomt_health.fatal_reason
                or self.eomt_health.recoverable_reason
                or semantic_problem
            )
        if self.last_error:
            level = DiagnosticStatus.ERROR
            message = self.last_error
        elif semantic_problem:
            level = DiagnosticStatus.WARN
            message = f"geometry active; semantic unavailable: {semantic_problem}"
        values = [
            KeyValue("stage", self.last_stage),
            KeyValue("semantic_stage", self.last_semantic_stage),
            KeyValue("window_sec", f"{self.window_sec:.3f}"),
            KeyValue("window_span_sec", f"{self.geometry_window.span_sec:.3f}"),
            KeyValue("geometry_frames", str(len(self.geometry_window))),
            KeyValue(
                "semantic_decay_tau_sec",
                f"{self.semantic_decay_tau_sec:.3f}",
            ),
            KeyValue("semantic_stale_sec", f"{self.semantic_stale_sec:.3f}"),
            KeyValue("received", str(self.received)),
            KeyValue("processed", str(self.processed)),
            KeyValue("dropped", str(self.dropped)),
            KeyValue("failures", str(self.failures)),
            KeyValue(
                "geometry_epoch_discards",
                str(self.geometry_epoch_discards),
            ),
            KeyValue("semantic_processed", str(self.semantic_processed)),
            KeyValue("semantic_dropped", str(self.semantic_dropped)),
            KeyValue("semantic_failures", str(self.semantic_failures)),
            KeyValue(
                "semantic_epoch_discards",
                str(self.semantic_epoch_discards),
            ),
            KeyValue("gpu_decode_fallbacks", str(self.gpu_decode_fallbacks)),
            KeyValue("gpu_pointcloud", str(self.gpu_pointcloud)),
            KeyValue("gpu_semantic_map", str(self.gpu_semantic_map)),
            KeyValue("sample_mode", self.sample_mode),
            KeyValue(
                "sample_min_valid_weight",
                f"{self.sample_min_valid_weight:.3f}",
            ),
            KeyValue("scan_deskew_mode", self.scan_deskew_mode),
            KeyValue("scan_stamp_reference", self.scan_stamp_reference),
            KeyValue("scan_row_direction", str(self.scan_row_direction)),
            KeyValue(
                "scan_period_estimate_sec",
                f"{self.scan_period_estimate_sec:.5f}",
            ),
            KeyValue("scan_deskew_applied", str(self.scan_deskew_applied)),
            KeyValue("scan_deskew_fallbacks", str(self.scan_deskew_fallbacks)),
            KeyValue("strict_window", str(self.strict_window)),
            KeyValue("geometry_backend", "cuda"),
            KeyValue("semantic_enabled", str(self.semantic_enabled)),
            KeyValue("semantic_backend", self.semantic_backend),
            KeyValue(
                "publish_legacy_probability",
                str(self.publish_legacy_probability),
            ),
            KeyValue("camera_info_late_matched", str(self.camera_info_late_matched)),
            KeyValue("camera_info_pending_dropped", str(self.camera_info_pending_dropped)),
            KeyValue(
                "camera_info_cache_evicted",
                str(self.camera_info_cache.evicted_count),
            ),
            KeyValue(
                "cupy_pool_used_bytes",
                str(self.geometry_map.memory_pool_used_bytes),
            ),
            KeyValue(
                "cupy_pool_total_bytes",
                str(self.geometry_map.memory_pool_total_bytes),
            ),
            KeyValue(
                "traversability_publish_hz",
                f"{float(getattr(self, 'traversability_publish_hz', 0.0)):.3f}",
            ),
        ]
        values.extend(
            KeyValue(name, str(value))
            for name, value in sorted(
                getattr(self.geometry_map, "last_lifecycle_stats", {}).items()
            )
        )
        values.extend(
            KeyValue(name, str(value))
            for name, value in self.last_semantic_stats.items()
        )
        if self.semantic_matcher is not None:
            values.extend(
                [
                    KeyValue(
                        "pending_sensors",
                        str(self.semantic_matcher.pending_sensor_count),
                    ),
                    KeyValue(
                        "pending_predictions",
                        str(self.semantic_matcher.pending_prediction_count),
                    ),
                ]
            )
            values.extend(
                KeyValue(f"semantic_{name}", str(count))
                for name, count in sorted(self.semantic_matcher.counters.items())
            )
        if self.eomt_health is not None:
            values.append(
                KeyValue("semantic_fatal_reason", self.eomt_health.fatal_reason)
            )
            values.extend(
                KeyValue(f"health_{name}", str(count))
                for name, count in sorted(
                    self.eomt_health.reason_counters.items()
                )
            )
        timing_names = (
            "decode_sec",
            "tf_lookup_sec",
            "window_update_sec",
            "geometry_sec",
            *_GEOMETRY_DIAGNOSTIC_TIMING_NAMES,
            "geometry_profile_gap_sec",
            "semantic_grid_sec",
            "risk_fusion_sec",
            "publish_sec",
            "scan_deskew_sec",
            "unaccounted_sec",
            "total_sec",
            "semantic_inference_sec",
            "semantic_decode_sec",
            "semantic_tf_lookup_sec",
            "semantic_cloud_alignment_sec",
            "semantic_projection_sec",
            "semantic_grid_update_sec",
            "semantic_total_sec",
        )
        values.extend(
            KeyValue(name, f"{float(timings.get(name, 0.0)):.4f}")
            for name in timing_names
        )
        return DiagnosticStatus(
            level=level,
            name="roadrunner",
            message=message,
            hardware_id=self.device,
            values=values,
        )

    def _heartbeat_loop(self) -> None:
        while not self.stop_event.wait(1.0) and not rospy.is_shutdown():
            self._heartbeat()

    def _heartbeat(self) -> None:
        if self.eomt_health is not None:
            self.eomt_health.check(now=time.monotonic())
            with self.matcher_lock:
                self.semantic_matcher.evict(now=time.monotonic())
        self._publish_diagnostics(rospy.Time.now())

    def _publish_diagnostics(self, stamp) -> None:
        diagnostics = DiagnosticArray()
        diagnostics.header.stamp = stamp
        combined_timing = dict(self.last_timing)
        combined_timing.update(self.last_semantic_timing)
        diagnostics.status = [self._diagnostic_status(**combined_timing)]
        self.diagnostics_pub.publish(diagnostics)

    def _publish_maps(
        self,
        stamp,
        fused,
        cvar_fused,
        maps,
        geometry,
        cvar_geometry,
        semantic,
        cvar_semantic,
    ):
        """发布启用的局部风险分支；CVaR 分支始终保留。"""
        if self.publish_legacy_probability:
            layers = traversability_probability_layers(
                geometry_risk=geometry["geometry_hindsight_target"],
                geometry_valid=geometry["geometry_hindsight_valid_mask"],
                semantic_risk=semantic["semantic_hindsight_target"],
                semantic_valid=semantic["semantic_hindsight_valid_mask"],
                final_risk=fused["traversability_risk"],
                final_valid=fused["valid_mask"],
                map_spec=self.map_spec,
            )
        cvar_layers = traversability_probability_layers(
            geometry_risk=cvar_geometry["geometry_hindsight_target"],
            geometry_valid=cvar_geometry["geometry_hindsight_valid_mask"],
            semantic_risk=cvar_semantic["semantic_hindsight_target"],
            semantic_valid=cvar_semantic["semantic_hindsight_valid_mask"],
            final_risk=cvar_fused["traversability_risk"],
            final_valid=cvar_fused["valid_mask"],
            map_spec=self.map_spec,
        )
        if self.publish_legacy_probability:
            self.probability_pub.publish(
                probability_grid_map_message(
                    layers,
                    stamp,
                    self.output_frame,
                    map_spec=self.map_spec,
                    elevation=maps["elevation"],
                )
            )
        self.cvar_probability_pub.publish(
            probability_grid_map_message(
                cvar_layers,
                stamp,
                self.output_frame,
                map_spec=self.map_spec,
                elevation=maps["elevation"],
            )
        )
        image_outputs = [
            (self.cvar_risk_pub, cvar_fused["traversability_risk"]),
            (
                self.confidence_pub,
                cvar_fused["reliability_map"]
                if not self.publish_legacy_probability
                else fused["reliability_map"],
            ),
            (
                self.elevation_pub,
                np.where(geometry["geometry_hindsight_valid_mask"], maps["elevation"], np.nan),
            ),
            (self.geometry_risk_cvar_pub, maps["geometry_risk_cvar"]),
            (
                self.semantic_risk_cvar_pub,
                cvar_semantic["semantic_hindsight_risk_cvar"],
            ),
        ]
        if self.publish_legacy_probability:
            image_outputs[0:0] = [
                (self.risk_pub, fused["traversability_risk"]),
            ]
            image_outputs.insert(
                3,
                (self.geometry_risk_std_pub, maps["geometry_risk_std"]),
            )
            image_outputs.insert(
                5,
                (
                    self.semantic_risk_std_pub,
                    semantic["semantic_hindsight_risk_std"],
                ),
            )
        for publisher, values in image_outputs:
            publisher.publish(float_image_message(values, stamp, self.output_frame))
        self._schedule_world_map(stamp)

    def _schedule_world_map(self, stamp) -> None:
        """按限频把最新时间戳交给世界图线程，队列满时丢弃旧请求。"""
        if not self.publish_world_map or self.world_map_publish_hz <= 0.0:
            return
        now = time.monotonic()
        if (
            now - self.last_world_map_publish_monotonic
            < self.world_map_publish_period
        ):
            return
        self.last_world_map_publish_monotonic = now
        try:
            self.world_map_work.put_nowait(stamp)
        except queue.Full:
            try:
                self.world_map_work.get_nowait()
                self.world_map_work.task_done()
            except queue.Empty:
                pass
            try:
                self.world_map_work.put_nowait(stamp)
            except queue.Full:
                pass

    def _world_map_worker(self) -> None:
        """异步构造并发布世界系滚动概率图。"""
        while not self.stop_event.is_set() and not rospy.is_shutdown():
            try:
                stamp = self.world_map_work.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._publish_world_map(stamp)
            finally:
                self.world_map_work.task_done()

    def _publish_world_map(self, stamp) -> None:
        """以世界坐标系发布滚动地图，限频以保护车体系实时链。

        局部 ``~traversability_probability`` 仍然每个几何帧发布；世界图只
        用于 RViz/调试，直接读取共享滚动栅格，避免再次遍历历史点云或重新
        做坐标变换。地图中心随滚动栅格移动，消息的 pose 记录该中心。
        """
        if self.world_cvar_probability_pub is None:
            return
        try:
            if self.semantic_map is None:
                with self.geometry_map.rolling_grid.lock:
                    center = self.geometry_map.rolling_grid.center
                    geometry_maps = self.geometry_map.world_geometry_snapshot()
                    if geometry_maps is None:
                        return
                semantic = self._empty_semantic_snapshot()
            else:
                # 语义更新的锁顺序也是 semantic_map_lock -> rolling_grid.lock；
                # 在同一临界区获取几何和语义快照，保证中心与两个数组一致。
                with self.semantic_map_lock:
                    with self.geometry_map.rolling_grid.lock:
                        center = self.geometry_map.rolling_grid.center
                        geometry_maps = self.geometry_map.world_geometry_snapshot()
                        if geometry_maps is None:
                            return
                        semantic = self.semantic_map.sample_world(
                            timestamp=float(stamp.to_sec()),
                            center=center,
                            world_map_length=self.world_map_length,
                            resolution=self.map_spec.resolution,
                        )
            geometry = self._geometry_teacher(geometry_maps)
            geometry = apply_grass_rule_to_accumulated_geometry(geometry, semantic)
            if self.publish_legacy_probability:
                legacy_geometry, cvar_geometry = geometry_risk_variants(
                    geometry,
                    use_cvar=self.use_geometry_cvar,
                )
                legacy_semantic, cvar_semantic = semantic_risk_variants(
                    semantic,
                    use_cvar=self.use_semantic_cvar,
                )
                fused = fuse_accumulated_geometry_semantics(
                    legacy_geometry, legacy_semantic
                )
            else:
                cvar_geometry = geometry_cvar_risk(
                    geometry,
                    use_cvar=self.use_geometry_cvar,
                )
                cvar_semantic = semantic_cvar_risk(
                    semantic,
                    use_cvar=self.use_semantic_cvar,
                )
                legacy_geometry = None
                legacy_semantic = None
                fused = None
            cvar_fused = fuse_accumulated_geometry_semantics(
                cvar_geometry, cvar_semantic
            )
            if self.publish_legacy_probability:
                layers = traversability_probability_layers(
                    geometry_risk=legacy_geometry["geometry_hindsight_target"],
                    geometry_valid=legacy_geometry["geometry_hindsight_valid_mask"],
                    semantic_risk=legacy_semantic["semantic_hindsight_target"],
                    semantic_valid=legacy_semantic["semantic_hindsight_valid_mask"],
                    final_risk=fused["traversability_risk"],
                    final_valid=fused["valid_mask"],
                    map_spec=self.world_map_spec,
                )
            cvar_layers = traversability_probability_layers(
                geometry_risk=cvar_geometry["geometry_hindsight_target"],
                geometry_valid=cvar_geometry["geometry_hindsight_valid_mask"],
                semantic_risk=cvar_semantic["semantic_hindsight_target"],
                semantic_valid=cvar_semantic["semantic_hindsight_valid_mask"],
                final_risk=cvar_fused["traversability_risk"],
                final_valid=cvar_fused["valid_mask"],
                map_spec=self.world_map_spec,
            )
            if self.publish_legacy_probability:
                self.world_probability_pub.publish(
                    probability_grid_map_message(
                        layers,
                        stamp,
                        self.world_frame,
                        map_spec=self.world_map_spec,
                        center_xy=center,
                    )
                )
            self.world_cvar_probability_pub.publish(
                probability_grid_map_message(
                    cvar_layers,
                    stamp,
                    self.world_frame,
                    map_spec=self.world_map_spec,
                    center_xy=center,
                )
            )
        except Exception as error:
            rospy.logerr_throttle(
                5.0,
                "world traversability map publish failed; local map continues: %s",
                error,
            )

    def _geometry_worker(self) -> None:
        """几何工作线程：持续处理最新点云，异常不会杀死 ROS 主进程。"""
        while not self.stop_event.is_set() and not rospy.is_shutdown():
            try:
                item = self.geometry_work.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._process_geometry(*item)
            except Exception as error:
                self.failures += 1
                self.last_error = f"{type(error).__name__}: {error}"
                rospy.logerr_throttle(2.0, "online traversability frame failed: %s", error)
            finally:
                self.geometry_work.task_done()

    def _semantic_worker(self) -> None:
        """语义工作线程：处理 EoMT/DeepLab 结果，不阻塞几何线程。"""
        while not self.stop_event.is_set() and not rospy.is_shutdown():
            try:
                item = self.semantic_work.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._process_semantic(*item)
            except Exception as error:
                self.semantic_failures += 1
                self.last_semantic_error = f"{type(error).__name__}: {error}"
                rospy.logerr_throttle(
                    2.0,
                    "online semantic enhancement failed: %s",
                    error,
                )
            finally:
                self.semantic_work.task_done()

    def shutdown(self) -> None:
        self.stop_event.set()
        current = threading.current_thread()
        for worker in (
            self.geometry_worker,
            self.semantic_worker,
            self.world_map_worker,
        ):
            if worker.is_alive() and worker is not current:
                worker.join(timeout=2.0)
        if (
            self.heartbeat_worker.is_alive()
            and self.heartbeat_worker is not current
        ):
            self.heartbeat_worker.join(timeout=2.0)


def main() -> None:
    rospy.init_node("roadrunner")
    RoadRunnerMapping()
    rospy.spin()


if __name__ == "__main__":
    main()
