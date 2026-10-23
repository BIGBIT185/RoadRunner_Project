"""在线语义后端选择、时间门禁和健康状态。

这些函数不依赖 rospy，因而可以在普通 Python 单元测试中验证后端策略，
也把“语义异常不应拖垮几何主链”的规则集中在一个模块中。
"""

from __future__ import annotations

import math
from collections import Counter, OrderedDict


VALID_SEMANTIC_BACKENDS = frozenset({"disabled", "eomt", "deeplab"})


def resolve_semantic_backend(value) -> str:
    """把 ROS 参数规范化为 EoMT、DeepLab 或 geometry-only。"""
    backend = "eomt" if value is None else str(value).strip().lower()
    if backend in {"none", "off", "disabled", "geometry", "geometry-only"}:
        backend = "disabled"
    if backend not in VALID_SEMANTIC_BACKENDS:
        choices = ", ".join(sorted(VALID_SEMANTIC_BACKENDS))
        raise ValueError(
            f"semantic_backend must be one of {{{choices}}}, received {value!r}"
        )
    return backend


def initialize_inline_segmenter(backend: str, factory):
    """仅在显式选择 DeepLab 时在融合进程内创建旧分割器。"""

    backend = resolve_semantic_backend(backend)
    return factory() if backend == "deeplab" else None


def camera_info_matches_image(
    camera_stamp_ns: int,
    image_stamp_ns: int,
    *,
    tolerance_sec: float = 0.08,
) -> bool:
    """判断 CameraInfo 是否为无时间戳或足够接近当前图像。"""

    camera_stamp_ns = int(camera_stamp_ns)
    image_stamp_ns = int(image_stamp_ns)
    if camera_stamp_ns == 0:
        return True
    tolerance_ns = int(round(float(tolerance_sec) * 1_000_000_000))
    return abs(camera_stamp_ns - image_stamp_ns) <= tolerance_ns


def timestamp_is_future(
    timestamp: float,
    current_time: float,
    *,
    tolerance_sec: float = 1.0,
) -> bool:
    """检测仿真时间倒退后仍排队的“未来”工作项。"""

    timestamp_value = float(timestamp)
    current_value = float(current_time)
    tolerance_value = float(tolerance_sec)
    if (
        not math.isfinite(timestamp_value)
        or not math.isfinite(current_value)
        or not math.isfinite(tolerance_value)
        or tolerance_value < 0.0
    ):
        raise ValueError("timestamps and tolerance must be finite")
    if current_value <= 0.0:
        return False
    return timestamp_value - current_value > tolerance_value


class BoundedCameraInfoCache:
    """用有界时间戳缓存把图像绑定到对应 CameraInfo。"""

    def __init__(self, *, max_entries: int = 64, tolerance_sec: float = 0.08):
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = int(max_entries)
        self.tolerance_ns = int(round(float(tolerance_sec) * 1_000_000_000))
        self._messages = OrderedDict()
        self._timeless = None
        self.evicted_count = 0

    def add(self, stamp_ns: int, message) -> None:
        # stamp=0 表示静态/无时间 CameraInfo，始终作为回退项保留。
        stamp_ns = int(stamp_ns)
        if stamp_ns == 0:
            self._timeless = message
            return
        self._messages.pop(stamp_ns, None)
        self._messages[stamp_ns] = message
        while len(self._messages) > self.max_entries:
            self._messages.popitem(last=False)
            self.evicted_count += 1

    def match(self, image_stamp_ns: int):
        # 优先精确匹配，再在 80 ms 门禁内选择最近的一条，避免跨帧误用标定。
        image_stamp_ns = int(image_stamp_ns)
        exact = self._messages.pop(image_stamp_ns, None)
        if exact is not None:
            return exact
        candidates = [
            (abs(stamp - image_stamp_ns), stamp)
            for stamp in self._messages
            if abs(stamp - image_stamp_ns) <= self.tolerance_ns
        ]
        if candidates:
            _, best_stamp = min(candidates)
            return self._messages.pop(best_stamp)
        return self._timeless


class EomtHealthMonitor:
    """基于单调墙钟的 EoMT 健康状态机。

    readiness/消息契约错误是致命状态；输出暂时停止是可恢复 WARN。几何线程
    不读取 ``can_publish`` 作为发布门禁，因此语义故障不会阻塞几何地图。
    """

    def __init__(
        self,
        *,
        started_monotonic: float,
        readiness_timeout_sec: float = 120.0,
        staleness_timeout_sec: float = 5.0,
    ) -> None:
        self.started_monotonic = float(started_monotonic)
        self.readiness_timeout_sec = float(readiness_timeout_sec)
        self.staleness_timeout_sec = float(staleness_timeout_sec)
        self.ready_monotonic = None
        self.last_output_monotonic = None
        self.fatal_reason = ""
        self.recoverable_reason = ""
        self.reason_counters = Counter()

    @property
    def can_publish(self) -> bool:
        return (
            not self.fatal_reason
            and not self.recoverable_reason
            and self.ready_monotonic is not None
            and self.last_output_monotonic is not None
        )

    def mark_ready(self, *, now: float) -> None:
        if self.ready_monotonic is None and not self.fatal_reason:
            self.ready_monotonic = float(now)

    def mark_output(self, *, now: float) -> None:
        if not self.fatal_reason:
            self.last_output_monotonic = float(now)
            if self.recoverable_reason == "eomt_output_stale":
                self.recoverable_reason = ""

    def fail(self, reason: str) -> str:
        reason = str(reason)
        self.reason_counters[reason] += 1
        if not self.fatal_reason:
            self.fatal_reason = reason
        return self.fatal_reason

    def check(self, *, now: float):
        # 这里使用 time.monotonic() 的时间，不使用 /clock，避免仿真暂停时
        # EoMT 长时间无输出却无法被诊断。
        if self.fatal_reason:
            return self.fatal_reason
        now = float(now)
        if self.ready_monotonic is None:
            if now - self.started_monotonic > self.readiness_timeout_sec:
                return self.fail("eomt_readiness_timeout")
            return None
        output_anchor = (
            self.last_output_monotonic
            if self.last_output_monotonic is not None
            else self.ready_monotonic
        )
        if now - output_anchor > self.staleness_timeout_sec:
            if self.recoverable_reason != "eomt_output_stale":
                self.reason_counters["eomt_output_stale"] += 1
            self.recoverable_reason = "eomt_output_stale"
            return self.recoverable_reason
        self.recoverable_reason = ""
        return None
