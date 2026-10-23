#!/usr/bin/env python3
"""GOOSE 的独立 EoMT 语义推理节点。

该节点运行在 Python 3.10/PyTorch 2.7 环境，和 Python 3.8 的 ROS 融合节点
通过 ``EomtSemanticPrediction`` 消息通信。它只负责图像推理和原子结果发布，
不负责点云、TF、坐标变换或风险融合。
"""

from __future__ import annotations

import copy
import platform
import threading
import time
from pathlib import Path

import numpy as np
import rospy
import torch
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_srvs.srv import Trigger, TriggerResponse
from transformers.utils import is_torch_sdpa_available
from transformers import (
    AutoImageProcessor,
    EomtConfig,
    EomtForUniversalSegmentation,
)

from roadrunner_mapping.semantics.eomt_postprocess import (
    postprocess_semantic_prediction,
)
from roadrunner_mapping.semantics.eomt_runtime import (
    InferencePacer,
    resolve_attention_backend,
    LatestFrameSlot,
    PredictionResultWriter,
    decode_rgb_image,
    load_eomt_model_compatibly,
    validate_model_directory,
)
from roadrunner_mapping.msg import EomtSemanticPrediction
from roadrunner_mapping.semantics.semantic_profiles import get_eomt_profile


def image_message(values: np.ndarray, header, encoding: str) -> Image:
    """把标签或置信度数组编码成带原始图像 header 的 ROS Image。"""
    if encoding == "mono8":
        array = np.ascontiguousarray(values, dtype=np.uint8)
    elif encoding == "32FC1":
        array = np.ascontiguousarray(values, dtype="<f4")
    else:
        raise ValueError(f"unsupported prediction encoding: {encoding}")
    if array.ndim != 2:
        raise ValueError("semantic output must be two-dimensional")
    message = Image()
    message.header = copy.deepcopy(header)
    message.height, message.width = array.shape
    message.encoding = encoding
    message.is_bigendian = 0
    message.step = int(array.strides[0])
    message.data = array.tobytes()
    return message


class EomtSemanticNode:
    """以“最新帧优先”策略运行 EoMT 推理。"""

    def __init__(self) -> None:
        # 模型目录和 hash 清单都通过 ROS 参数传入，便于容器内外使用同一份代码。
        root = Path(rospy.get_param("~project_root", str(Path(__file__).resolve().parents[3])))
        self.semantic_profile = get_eomt_profile(
            rospy.get_param("~semantic_profile", "rugd24")
        )
        self.model_directory = Path(
            rospy.get_param(
                "~model_directory",
                str(
                    root
                    / "models/eomt_rugd24"
                ),
            )
        )
        self.hash_manifest = Path(
            rospy.get_param(
                "~hash_manifest",
                str(
                    root
                    / "src/roadrunner_mapping/config/eomt_rugd_model.sha256"
                ),
            )
        )
        self.device = str(rospy.get_param("~device", "cuda:0"))
        self.max_inference_hz = float(rospy.get_param("~max_inference_hz", 2.0))
        self.pacer = InferencePacer(self.max_inference_hz)
        self.attention_backend = resolve_attention_backend(
            str(rospy.get_param("~attention_backend", "auto")),
            sdpa_available=(
                bool(getattr(EomtForUniversalSegmentation, "_supports_sdpa", False))
                and is_torch_sdpa_available()
            ),
            machine=platform.machine(),
        )
        # LatestFrameSlot 会替换尚未推理的旧图像，防止 EoMT 延迟随 rosbag 速度累积。
        self.slot = LatestFrameSlot()
        prediction_output_dir = rospy.get_param(
            "~prediction_output_dir",
            str(root / "outputs/eomt_semantic_predictions"),
        )
        self.prediction_writer = PredictionResultWriter(
            prediction_output_dir,
            self.semantic_profile.classes,
            queue_size=int(rospy.get_param("~prediction_save_queue_size", 4)),
        )
        self.processed = 0
        self.failures = 0
        self.last_inference_sec = 0.0
        self.last_output_monotonic = None
        self.stop_event = threading.Event()
        self.prediction_pub = rospy.Publisher(
            "~prediction", EomtSemanticPrediction, queue_size=1
        )
        self.ready_pub = rospy.Publisher("~ready", Bool, queue_size=1, latch=True)
        self.diagnostics_pub = rospy.Publisher(
            "~diagnostics", DiagnosticArray, queue_size=1, latch=True
        )
        self.start_saving_service = rospy.Service(
            "~start_saving", Trigger, self._start_saving
        )
        self.stop_saving_service = rospy.Service(
            "~stop_saving", Trigger, self._stop_saving
        )
        if bool(rospy.get_param("~save_predictions", False)):
            self.prediction_writer.enable()

        # 模型必须在订阅图像之前加载成功；失败时发布 ERROR 并退出，避免静默无输出。
        self._load_model()
        image_topic = rospy.get_param(
            "~image_topic", "/sensor/camera/windshield/vis/image_rect_color"
        )
        self.subscriber = rospy.Subscriber(
            image_topic, Image, self._image_callback, queue_size=1
        )
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.heartbeat_worker = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
        )
        self.worker.start()
        self.heartbeat_worker.start()
        rospy.on_shutdown(self.shutdown)
        self.ready_pub.publish(Bool(data=True))
        self._publish_diagnostics(DiagnosticStatus.OK, "EoMT ready")

    def _load_model(self) -> None:
        """检查 CUDA、模型文件、类别顺序和快速预处理器，然后加载半精度模型。"""
        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("EoMT requires an available CUDA device")
        validate_model_directory(
            self.model_directory,
            self.hash_manifest,
            expected_classes=self.semantic_profile.classes,
        )
        torch.cuda.set_per_process_memory_fraction(0.72, device=self.device)
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_directory,
            local_files_only=True,
            use_fast=True,
        )
        if type(self.processor).__name__ not in {
            "EomtImageProcessorFast",
            "EomtImageProcessor",
        } or not hasattr(self.processor, "merge_image_patches"):
            raise RuntimeError(
                "EoMT requires a split-image processor with merge_image_patches"
            )
        self.model = load_eomt_model_compatibly(
            EomtForUniversalSegmentation,
            EomtConfig,
            self.model_directory,
            attention_backend=self.attention_backend,
        ).eval().to(self.device)
        self.model.half()
        rospy.loginfo("EoMT attention=%s max_inference_hz=%.3f",
                      self.attention_backend, self.max_inference_hz)

    def _image_callback(self, message: Image) -> None:
        """只保存最新图像，输入队列不会积压。"""
        self.slot.put(message)

    def _infer(self, message: Image):
        """完成一次 EoMT 前向、后处理并封装原子预测消息。"""
        rgb = decode_rgb_image(message)
        inputs = self.processor(images=rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=torch.float16)
        patch_offsets = inputs.get("patch_offsets")
        started = time.perf_counter()
        # half + autocast 降低 V100/RTX 上的显存占用；同步 CUDA 后再记录耗时，
        # 避免异步 kernel 导致统计值偏小。
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            outputs = self.model(
                pixel_values=pixel_values,
                patch_offsets=patch_offsets,
            )
        # 后处理把模型输出恢复到原始图像尺寸，保证融合节点可直接按像素投影。
        labels, confidence = postprocess_semantic_prediction(
            outputs,
            processor=self.processor,
            target_sizes=[(int(message.height), int(message.width))],
        )
        torch.cuda.synchronize(self.device)
        inference_sec = time.perf_counter() - started
        result = EomtSemanticPrediction()
        result.header = copy.deepcopy(message.header)
        label_array = labels.cpu().numpy()
        result.labels = image_message(
            label_array, result.header, "mono8"
        )
        result.confidence = image_message(
            confidence.cpu().numpy(), result.header, "32FC1"
        )
        result.inference_sec = float(inference_sec)
        return result

    def _worker(self) -> None:
        """推理线程：取最新图像、发布预测，单次异常则按致命错误退出。"""
        while not self.stop_event.is_set() and not rospy.is_shutdown():
            # Wait before selecting the newest frame; ROS clock rewinds cannot
            # stall the pacer, and shutdown can interrupt a long wait.
            if self.pacer.wait(self.stop_event):
                return
            try:
                message = self.slot.get(timeout=0.2)
            except TimeoutError:
                continue
            except RuntimeError:
                return
            try:
                self.pacer.mark_started()
                prediction = self._infer(message)
                self.prediction_pub.publish(prediction)
                try:
                    self.prediction_writer.submit_message(prediction)
                except Exception as error:
                    # 保存是可选旁路，磁盘或消息异常不能中断实时语义发布。
                    rospy.logwarn_throttle(
                        5.0, "semantic result saving failed: %s", error
                    )
                self.processed += 1
                self.last_inference_sec = float(prediction.inference_sec)
                self.last_output_monotonic = time.monotonic()
            except Exception as error:
                self.failures += 1
                self._publish_diagnostics(DiagnosticStatus.ERROR, str(error))
                rospy.logfatal("EoMT inference failed: %s", error)
                rospy.signal_shutdown(str(error))
                return

    def _heartbeat_loop(self) -> None:
        """使用墙钟持续发布诊断，即使仿真时间暂停也能观察进程状态。"""
        while not self.stop_event.wait(1.0) and not rospy.is_shutdown():
            self._publish_diagnostics(DiagnosticStatus.OK, "EoMT ready")

    def _publish_diagnostics(self, level: int, message: str) -> None:
        """发布推理计数、替换帧数、耗时和 CUDA 显存占用。"""
        diagnostics = DiagnosticArray()
        diagnostics.header.stamp = rospy.Time.now()
        allocation = (
            int(torch.cuda.memory_allocated(self.device))
            if torch.cuda.is_available()
            else 0
        )
        diagnostics.status = [
            DiagnosticStatus(
                level=level,
                name="roadrunner_semantic",
                message=message,
                hardware_id=self.device,
                values=[
                    KeyValue("processed", str(self.processed)),
                    KeyValue("failures", str(self.failures)),
                    KeyValue("replaced", str(self.slot.replaced_count)),
                    KeyValue("attention_backend", self.attention_backend),
                    KeyValue("max_inference_hz", str(self.max_inference_hz)),
                    KeyValue("saving_enabled", str(self.prediction_writer.enabled)),
                    KeyValue("saved_predictions", str(self.prediction_writer.saved_count)),
                    KeyValue(
                        "dropped_predictions",
                        str(self.prediction_writer.dropped_count),
                    ),
                    KeyValue(
                        "inference_sec", f"{self.last_inference_sec:.6f}"
                    ),
                    KeyValue("torch_allocated_bytes", str(allocation)),
                ],
            )
        ]
        self.diagnostics_pub.publish(diagnostics)

    def shutdown(self) -> None:
        self.stop_event.set()
        self.slot.close()
        self.prediction_writer.close()
        current = threading.current_thread()
        if (
            hasattr(self, "worker")
            and self.worker.is_alive()
            and self.worker is not current
        ):
            self.worker.join(timeout=2.0)
        if (
            hasattr(self, "heartbeat_worker")
            and self.heartbeat_worker.is_alive()
            and self.heartbeat_worker is not current
        ):
            self.heartbeat_worker.join(timeout=2.0)

    def _start_saving(self, _request) -> TriggerResponse:
        """通过 ROS service 动态开启语义结果保存。"""

        try:
            self.prediction_writer.enable()
        except Exception as error:
            rospy.logerr("failed to start semantic result saving: %s", error)
            return TriggerResponse(success=False, message=str(error))
        return TriggerResponse(
            success=True,
            message=f"semantic results are saved to {self.prediction_writer.output_dir}",
        )

    def _stop_saving(self, _request) -> TriggerResponse:
        """通过 ROS service 动态关闭语义结果保存。"""

        self.prediction_writer.disable()
        return TriggerResponse(success=True, message="semantic result saving disabled")


def main() -> None:
    rospy.init_node("roadrunner_semantic")
    try:
        EomtSemanticNode()
    except Exception as error:
        diagnostics_pub = rospy.Publisher(
            "~diagnostics", DiagnosticArray, queue_size=1, latch=True
        )
        diagnostics = DiagnosticArray()
        diagnostics.header.stamp = rospy.Time.now()
        diagnostics.status = [
            DiagnosticStatus(
                level=DiagnosticStatus.ERROR,
                name="roadrunner_semantic",
                message=str(error),
                hardware_id="cuda:0",
            )
        ]
        diagnostics_pub.publish(diagnostics)
        time.sleep(0.25)
        rospy.logfatal("EoMT startup failed: %s", error)
        raise
    rospy.spin()


if __name__ == "__main__":
    main()
