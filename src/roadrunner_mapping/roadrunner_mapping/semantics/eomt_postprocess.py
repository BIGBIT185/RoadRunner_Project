"""Hugging Face EoMT 语义分数的确定性后处理和重建。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def postprocess_semantic_prediction(
    outputs,
    *,
    processor,
    target_sizes,
    epsilon: float = 1e-8,
):
    """Return one original-resolution label/confidence image.

    The score construction intentionally mirrors Transformers 4.56.1, while
    retaining normalized evidence as a confidence image for point projection.
    """

    mask_logits = outputs.masks_queries_logits
    class_logits = outputs.class_queries_logits
    if mask_logits.ndim != 4 or class_logits.ndim != 3:
        raise ValueError("unexpected EoMT output rank")
    if mask_logits.shape[:2] != class_logits.shape[:2]:
        raise ValueError("EoMT query dimensions do not match")
    if len(target_sizes) != 1:
        raise ValueError("online EoMT post-processing expects one source image")

    resized_masks = F.interpolate(
        mask_logits.float(),
        size=(512, 512),
        mode="bilinear",
        align_corners=False,
    ).sigmoid()
    class_probability = class_logits.float().softmax(dim=-1)[..., :-1]
    crop_scores = torch.einsum(
        "bqc,bqhw->bchw", class_probability, resized_masks
    )
    reconstructed = processor.merge_image_patches(
        crop_scores,
        outputs.patch_offsets,
        target_sizes,
        processor.size,
    )
    if len(reconstructed) != 1:
        raise ValueError("EoMT processor returned an unexpected image count")

    scores = reconstructed[0]
    total = scores.sum(dim=0)
    maximum, labels = scores.max(dim=0)
    finite = torch.isfinite(scores).all(dim=0) & torch.isfinite(total)
    valid = finite & (total > float(epsilon))
    confidence = torch.zeros_like(total, dtype=torch.float32)
    confidence[valid] = maximum[valid] / total[valid].clamp_min(float(epsilon))
    labels = labels.to(torch.uint8)
    labels = torch.where(valid, labels, torch.zeros_like(labels))
    confidence = torch.where(
        valid & torch.isfinite(confidence),
        confidence.clamp(0.0, 1.0),
        torch.zeros_like(confidence),
    )
    return labels, confidence
