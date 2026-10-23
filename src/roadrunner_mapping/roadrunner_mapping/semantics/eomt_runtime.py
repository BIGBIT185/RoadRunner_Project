"""隔离 EoMT ROS 进程共享的模型加载、推理和运行时辅助函数。"""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np


def decode_rgb_image(message) -> np.ndarray:
    height, width, step = int(message.height), int(message.width), int(message.step)
    encoding = str(message.encoding).lower()
    if encoding not in {"rgb8", "bgr8"}:
        raise ValueError(f"unsupported EoMT image encoding: {message.encoding}")
    row_bytes = width * 3
    if height <= 0 or width <= 0 or step < row_bytes:
        raise ValueError("invalid EoMT image dimensions or step")
    wire = np.frombuffer(message.data, dtype=np.uint8)
    if wire.size != height * step:
        raise ValueError("EoMT image data size does not match dimensions")
    packed = wire.reshape(height, step)[:, :row_bytes].reshape(height, width, 3)
    return packed.copy() if encoding == "rgb8" else packed[:, :, ::-1].copy()


class LatestFrameSlot:
    """A blocking single-item slot that always retains the latest input."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._item = None
        self._closed = False
        self.replaced_count = 0

    def put(self, item) -> bool:
        with self._condition:
            if self._closed:
                raise RuntimeError("latest frame slot is closed")
            replaced = self._item is not None
            if replaced:
                self.replaced_count += 1
            self._item = item
            self._condition.notify()
            return replaced

    def get(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            while self._item is None and not self._closed:
                remaining = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                if remaining == 0.0:
                    raise TimeoutError("latest frame slot is empty")
                self._condition.wait(remaining)
            if self._item is None:
                raise RuntimeError("latest frame slot is closed")
            item = self._item
            self._item = None
            return item

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class PredictionResultWriter:
    """可选的异步 EoMT 结果保存器。

    默认处于关闭状态；启用后不会阻塞推理线程，队列满时丢弃待保存结果而不
    影响语义预测发布。标签和置信度同时保存为 NumPy 原始数组及 PNG，彩色
    mask 仅用于人工检查，不参与在线融合。
    """

    _RUGD24_PALETTE = np.asarray(
        (
            (128, 64, 0),
            (194, 178, 128),
            (0, 180, 0),
            (34, 139, 34),
            (128, 128, 128),
            (255, 0, 0),
            (135, 206, 235),
            (0, 0, 255),
            (128, 0, 128),
            (80, 80, 80),
            (160, 160, 160),
            (180, 180, 180),
            (255, 165, 0),
            (165, 42, 42),
            (100, 50, 20),
            (255, 20, 147),
            (255, 255, 0),
            (0, 128, 128),
            (0, 100, 0),
            (255, 255, 255),
            (0, 0, 0),
            (0, 255, 255),
            (200, 200, 200),
            (255, 192, 203),
        ),
        dtype=np.uint8,
    )

    def __init__(self, output_dir, classes, *, queue_size: int = 4) -> None:
        self.output_dir = Path(output_dir).expanduser()
        self.classes = tuple(str(value) for value in classes)
        self._queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._enabled = False
        self._closed = False
        self._next_index = 0
        self.saved_count = 0
        self.dropped_count = 0
        self._worker = threading.Thread(
            target=self._run,
            name="eomt-result-writer",
            daemon=True,
        )
        self._worker.start()

    @property
    def enabled(self) -> bool:
        with self._lock:
            return bool(self._enabled and not self._closed)

    def enable(self) -> None:
        """创建输出目录并开始接收新结果。"""

        with self._lock:
            if self._closed:
                raise RuntimeError("prediction result writer is closed")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            legend = {
                "profile": "rugd24",
                "classes": [
                    {"id": index, "name": name}
                    for index, name in enumerate(self.classes)
                ],
            }
            (self.output_dir / "class_legend.json").write_text(
                json.dumps(legend, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self._enabled = True

    def disable(self) -> None:
        """停止接收新结果；已入队结果仍会由后台线程写完。"""

        with self._lock:
            self._enabled = False

    def submit_message(self, message) -> bool:
        """将一条 EomtSemanticPrediction 非阻塞地交给保存线程。"""

        with self._lock:
            if not self._enabled or self._closed:
                return False
            index = self._next_index
            self._next_index += 1
        labels = self._decode_image(message.labels, np.uint8)
        confidence = self._decode_image(message.confidence, np.dtype("<f4"))
        if labels.shape != confidence.shape:
            raise ValueError("prediction labels/confidence dimensions do not match")
        item = (
            index,
            labels.copy(),
            confidence.copy(),
            int(message.header.stamp.to_nsec()),
            float(message.inference_sec),
        )
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            with self._lock:
                self.dropped_count += 1
            return False
        return True

    @staticmethod
    def _decode_image(message, dtype) -> np.ndarray:
        height, width, step = int(message.height), int(message.width), int(message.step)
        if height <= 0 or width <= 0 or step < width * np.dtype(dtype).itemsize:
            raise ValueError("invalid prediction image dimensions")
        wire = np.frombuffer(message.data, dtype=dtype)
        row_values = step // np.dtype(dtype).itemsize
        if wire.size != height * row_values:
            raise ValueError("prediction image data size does not match dimensions")
        return wire.reshape(height, row_values)[:, :width]

    def _run(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._write_item(item)
            except Exception:
                with self._lock:
                    self.dropped_count += 1

    def _write_item(self, item) -> None:
        from PIL import Image

        index, labels, confidence, stamp_nsec, inference_sec = item
        safe_labels = np.where(
            labels < len(self._RUGD24_PALETTE), labels, 0
        ).astype(np.uint8, copy=False)
        stem = f"prediction_{index:06d}_{stamp_nsec}"
        np.save(self.output_dir / f"{stem}_labels.npy", labels)
        np.save(self.output_dir / f"{stem}_confidence.npy", confidence)
        Image.fromarray(labels).save(self.output_dir / f"{stem}_labels.png")
        Image.fromarray(self._RUGD24_PALETTE[safe_labels]).save(
            self.output_dir / f"{stem}_mask.png"
        )
        confidence_u8 = np.clip(confidence * 255.0, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(confidence_u8).save(
            self.output_dir / f"{stem}_confidence.png"
        )
        metadata = {
            "index": int(index),
            "stamp_nsec": int(stamp_nsec),
            "height": int(labels.shape[0]),
            "width": int(labels.shape[1]),
            "inference_sec": float(inference_sec),
        }
        (self.output_dir / f"{stem}.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        with self._lock:
            self.saved_count += 1

    def close(self) -> None:
        """停止保存线程，并尽量写完已入队结果。"""

        with self._lock:
            self._enabled = False
            self._closed = True
        self._stop_event.set()
        if self._worker.is_alive() and threading.current_thread() is not self._worker:
            self._worker.join(timeout=5.0)


def resolve_attention_backend(requested, *, sdpa_available, machine):
    """Keep legacy Jetson eager behavior; use SDPA on supported desktop builds."""
    if requested not in {"auto", "eager", "sdpa"}:
        raise ValueError("attention_backend must be auto, eager or sdpa")
    if requested == "sdpa" and not sdpa_available:
        raise ValueError("SDPA is not supported by this model/PyTorch installation")
    if requested == "auto":
        return "sdpa" if sdpa_available and machine == "x86_64" else "eager"
    return requested


class InferencePacer:
    """Wall-clock start-to-start rate limit; wait BEFORE taking the latest frame."""

    def __init__(self, max_inference_hz, *, clock=time.monotonic):
        hz = float(max_inference_hz)
        if not np.isfinite(hz) or hz < 0:
            raise ValueError("max_inference_hz must be finite and non-negative")
        self.period = 1.0 / hz if hz else 0.0
        self.clock = clock
        self.next_start = 0.0

    def wait(self, stop_event):
        delay = max(0.0, self.next_start - self.clock())
        return stop_event.wait(delay) if delay > 0 else False

    def mark_started(self):
        self.next_start = self.clock() + self.period


def load_eomt_model_compatibly(
    model_class, config_class, model_directory, *, attention_backend="eager"
):
    """Bind EomtConfig explicitly for older Transformers; keep eager API default.

    The ROS node resolves auto separately. Explicit backend/model failures propagate
    without retrying, so OOM or corrupt weights cannot silently select another path.
    """
    if attention_backend not in {"eager", "sdpa"}:
        raise ValueError("attention_backend must be eager or sdpa")
    config = config_class.from_pretrained(model_directory, local_files_only=True)
    model_class.config_class = config_class
    return model_class.from_pretrained(
        model_directory, config=config, local_files_only=True,
        attn_implementation=attention_backend,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_model_directory(model_directory, hash_manifest, *, expected_classes):
    model_path = Path(model_directory)
    manifest_path = Path(hash_manifest)
    if not model_path.is_dir():
        raise ValueError(f"EoMT model directory is missing: {model_path}")
    expected_hashes = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        expected_hashes[name.strip()] = digest.lower()
    required = {"config.json", "preprocessor_config.json", "model.safetensors"}
    if set(expected_hashes) != required:
        raise ValueError("EoMT SHA256 manifest has unexpected files")
    for name, expected in expected_hashes.items():
        path = model_path / name
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"EoMT SHA256 mismatch: {name}")

    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    labels = config.get("id2label", {})
    ordered = tuple(labels.get(str(index)) for index in range(len(expected_classes)))
    if ordered != tuple(expected_classes) or len(labels) != len(expected_classes):
        raise ValueError("EoMT class configuration does not match the selected semantic profile")
    preprocessor = json.loads(
        (model_path / "preprocessor_config.json").read_text(encoding="utf-8")
    )
    if (
        preprocessor.get("image_processor_type")
        not in {"EomtImageProcessorFast", "EomtImageProcessor"}
        or preprocessor.get("do_split_image") is not True
    ):
        raise ValueError("EoMT split-image processor configuration is required")
    return config, preprocessor
