"""Offline RUGD-25 inference used only by the dataset teacher chain."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchvision.models.segmentation import deeplabv3_resnet101


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_rugd_model(class_count: int) -> nn.Module:
    model = deeplabv3_resnet101(
        pretrained=False,
        pretrained_backbone=False,
        aux_loss=True,
    )
    model.classifier[4] = nn.Conv2d(256, int(class_count), kernel_size=1)
    return model


def semantic_from_logits(
    logits: torch.Tensor,
    *,
    output_shape: Tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if logits.ndim != 4 or logits.shape[0] != 1:
        raise ValueError("semantic logits must have shape [1,C,H,W]")
    probabilities = torch.softmax(logits, dim=1)
    confidence, labels = torch.max(probabilities, dim=1, keepdim=True)
    labels = functional.interpolate(
        labels.float(),
        size=output_shape,
        mode="nearest",
    ).squeeze(0).squeeze(0)
    confidence = functional.interpolate(
        confidence,
        size=output_shape,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    return (
        labels.to(torch.uint8).cpu().numpy(),
        confidence.float().cpu().numpy().astype(np.float32, copy=False),
    )


def semantic_batch_from_logits(
    logits: torch.Tensor,
    *,
    output_shapes: list[Tuple[int, int]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    if logits.ndim != 4 or logits.shape[0] != len(output_shapes):
        raise ValueError("semantic batch logits and output_shapes must share batch size")
    return [
        semantic_from_logits(logits[index : index + 1], output_shape=shape)
        for index, shape in enumerate(output_shapes)
    ]


class OfflineRUGDSegmenter:
    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device: str = "auto",
        input_shape: Tuple[int, int] = (512, 512),
        class_count: int = 25,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.input_shape = tuple(int(value) for value in input_shape)
        self.class_count = int(class_count)
        self.model = make_rugd_model(self.class_count)
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location=self.device,
        )
        state = (
            checkpoint["model_state"]
            if isinstance(checkpoint, dict) and "model_state" in checkpoint
            else checkpoint
        )
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.mean = torch.tensor(
            IMAGENET_MEAN,
            device=self.device,
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            IMAGENET_STD,
            device=self.device,
        ).view(1, 3, 1, 1)

    def predict(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.predict_batch([image_rgb])[0]

    def predict_batch(
        self,
        images_rgb: list[np.ndarray],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if not images_rgb:
            raise ValueError("images_rgb must not be empty")
        images = [np.asarray(image) for image in images_rgb]
        if any(image.ndim != 3 or image.shape[2] != 3 for image in images):
            raise ValueError("each image_rgb must have shape [H,W,3]")
        tensor = torch.stack(
            [
                torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
                for image in images
            ]
        ).to(self.device)
        tensor = tensor.float() / 255.0
        tensor = functional.interpolate(
            tensor,
            size=self.input_shape,
            mode="bilinear",
            align_corners=False,
        )
        tensor = (tensor - self.mean) / self.std
        with torch.no_grad():
            logits = self.model(tensor)["out"]
        return semantic_batch_from_logits(
            logits,
            output_shapes=[(image.shape[0], image.shape[1]) for image in images],
        )
