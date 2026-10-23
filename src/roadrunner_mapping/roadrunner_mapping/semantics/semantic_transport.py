"""EoMT 预测消息的校验和延迟时间戳匹配。

EoMT 推理独立于融合节点，预测可能早于或晚于传感器项到达；本模块保证只有
header、尺寸、编码和类别范围都正确的原子消息才能进入语义线程。
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass

import numpy as np


class PredictionValidationError(ValueError):
    """带机器可读 reason 的预测契约错误。"""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(reason if not detail else f"{reason}: {detail}")


@dataclass(frozen=True)
class DecodedSemanticPrediction:
    stamp_ns: int
    frame_id: str
    labels: np.ndarray
    confidence: np.ndarray
    inference_sec: float


def stamp_to_nsec(stamp) -> int:
    """兼容 rospy.Time 和伪 ROS 测试对象，统一成整数纳秒。"""
    if hasattr(stamp, "to_nsec"):
        return int(stamp.to_nsec())
    if hasattr(stamp, "secs") and hasattr(stamp, "nsecs"):
        return int(stamp.secs) * 1_000_000_000 + int(stamp.nsecs)
    return int(stamp)


def _header_key(header):
    return (
        stamp_to_nsec(header.stamp),
        str(header.frame_id),
    )


def _decode_image(message, *, encoding: str, dtype) -> np.ndarray:
    """按 ROS Image 的 step、字节序和编码安全解码二维数组。"""
    if str(message.encoding) != encoding:
        raise PredictionValidationError(
            "bad_encoding", f"expected {encoding}, received {message.encoding}"
        )
    height = int(message.height)
    width = int(message.width)
    itemsize = np.dtype(dtype).itemsize
    if height <= 0 or width <= 0:
        raise PredictionValidationError("dimension_mismatch", "empty image")
    if int(message.step) != width * itemsize:
        raise PredictionValidationError("dimension_mismatch", "unexpected image step")
    expected_bytes = height * int(message.step)
    if len(message.data) != expected_bytes:
        raise PredictionValidationError("dimension_mismatch", "unexpected data size")
    byte_order = ">" if bool(message.is_bigendian) and itemsize > 1 else "<"
    wire_dtype = np.dtype(dtype).newbyteorder(byte_order)
    return np.frombuffer(message.data, dtype=wire_dtype).astype(
        dtype, copy=True
    ).reshape(height, width)


def validate_atomic_prediction(message, *, class_count: int = 64):
    """校验一条原子预测并复制数组所有权，避免引用 ROS 消息底层 buffer。"""

    outer_header = _header_key(message.header)
    if (
        _header_key(message.labels.header) != outer_header
        or _header_key(message.confidence.header) != outer_header
    ):
        raise PredictionValidationError("header_mismatch")
    if (
        int(message.labels.height) != int(message.confidence.height)
        or int(message.labels.width) != int(message.confidence.width)
    ):
        raise PredictionValidationError("dimension_mismatch")

    labels = _decode_image(message.labels, encoding="mono8", dtype=np.uint8)
    confidence = _decode_image(
        message.confidence, encoding="32FC1", dtype=np.float32
    )
    inference_sec = float(message.inference_sec)
    if not np.isfinite(inference_sec) or not np.isfinite(confidence).all():
        raise PredictionValidationError("nonfinite_value")
    if labels.size and int(labels.max()) >= int(class_count):
        raise PredictionValidationError("label_range")
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise PredictionValidationError("confidence_range")
    return DecodedSemanticPrediction(
        stamp_ns=outer_header[0],
        frame_id=outer_header[1],
        labels=labels,
        confidence=confidence,
        inference_sec=inference_sec,
    )


class BoundedSemanticMatcher:
    """按图像纳秒时间戳匹配延迟到达的传感器项和语义预测。"""

    def __init__(self, *, max_age_sec: float = 5.0, max_entries: int = 64) -> None:
        if not np.isfinite(max_age_sec) or max_age_sec <= 0.0:
            raise ValueError("max_age_sec must be finite and positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.max_age_sec = float(max_age_sec)
        self.max_entries = int(max_entries)
        self.counters = Counter()
        self._sensors = OrderedDict()
        self._predictions = OrderedDict()
        self._processed = OrderedDict()

    @property
    def pending_sensor_count(self) -> int:
        return len(self._sensors)

    @property
    def pending_prediction_count(self) -> int:
        return len(self._predictions)

    def _expire_cache(self, cache, now: float, counter: str) -> None:
        expired = [
            stamp
            for stamp, (arrival, _) in cache.items()
            if now - arrival > self.max_age_sec
        ]
        for stamp in expired:
            cache.pop(stamp, None)
            self.counters[counter] += 1

    def evict(self, *, now: float) -> None:
        now = float(now)
        self._expire_cache(self._sensors, now, "sensor_age_eviction")
        self._expire_cache(self._predictions, now, "prediction_age_eviction")
        old_processed = [
            stamp
            for stamp, arrival in self._processed.items()
            if now - arrival > self.max_age_sec
        ]
        for stamp in old_processed:
            self._processed.pop(stamp, None)

    def _add(self, side: str, stamp_ns: int, payload, now: float):
        # 两侧使用同一个 stamp 作为键；配对完成后进入 processed，防止重复预测
        # 再次触发语义更新。缓存容量和年龄共同限制内存与延迟。
        now = float(now)
        stamp_ns = int(stamp_ns)
        self.evict(now=now)
        own = self._sensors if side == "sensor" else self._predictions
        other = self._predictions if side == "sensor" else self._sensors
        if stamp_ns in self._processed or stamp_ns in own:
            self.counters[f"duplicate_{side}"] += 1
            return None
        if stamp_ns in other:
            _, other_payload = other.pop(stamp_ns)
            self._processed[stamp_ns] = now
            while len(self._processed) > self.max_entries:
                self._processed.popitem(last=False)
            if side == "sensor":
                return payload, other_payload
            return other_payload, payload
        own[stamp_ns] = (now, payload)
        while len(own) > self.max_entries:
            own.popitem(last=False)
            self.counters[f"{side}_capacity_eviction"] += 1
        return None

    def add_sensor(self, stamp_ns: int, payload, *, now: float):
        return self._add("sensor", stamp_ns, payload, now)

    def add_prediction(self, stamp_ns: int, payload, *, now: float):
        return self._add("prediction", stamp_ns, payload, now)

    def clear(self) -> None:
        """清除当前 replay epoch 的未配对数据，但保留累计诊断计数。"""

        self._sensors.clear()
        self._predictions.clear()
        self._processed.clear()
