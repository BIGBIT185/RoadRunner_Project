from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from roadrunner_mapping.semantics.eomt_postprocess import (
    postprocess_semantic_prediction,
)


class FakeProcessor:
    size = {"shortest_edge": 512, "longest_edge": 2048}

    def merge_image_patches(
        self,
        segmentation_logits,
        patch_offsets,
        target_sizes,
        size,
    ):
        assert patch_offsets == [(0, 0, 512)]
        assert size == self.size
        return [
            F.interpolate(
                segmentation_logits,
                size=target_sizes[0],
                mode="bilinear",
                align_corners=False,
            )[0]
        ]


def test_fixed_logits_exclude_no_object_and_normalize_confidence():
    outputs = SimpleNamespace(
        class_queries_logits=torch.tensor(
            [[[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0]]]
        ),
        masks_queries_logits=torch.tensor([[[[10.0]], [[0.0]]]]),
        patch_offsets=[(0, 0, 512)],
    )

    labels, confidence = postprocess_semantic_prediction(
        outputs,
        processor=FakeProcessor(),
        target_sizes=[(1, 1)],
    )

    assert labels.tolist() == [[0]]
    assert confidence.item() == pytest.approx(2.0 / 3.0, abs=1e-4)


def test_nonfinite_or_empty_evidence_becomes_invalid_class_zero():
    outputs = SimpleNamespace(
        class_queries_logits=torch.tensor([[[0.0, 0.0]]]),
        masks_queries_logits=torch.tensor([[[[float("nan")]]]]),
        patch_offsets=[(0, 0, 512)],
    )

    labels, confidence = postprocess_semantic_prediction(
        outputs,
        processor=FakeProcessor(),
        target_sizes=[(1, 1)],
    )

    assert labels.tolist() == [[0]]
    assert confidence.tolist() == [[0.0]]


def test_real_processor_merges_three_overlapping_crops_to_original_size():
    transformers = pytest.importorskip("transformers")
    model_directory = (
        Path(__file__).resolve().parents[3]
        / "models/eomt_rugd24"
    )
    if not model_directory.is_dir():
        pytest.skip("local EoMT model is unavailable")
    processor = transformers.AutoImageProcessor.from_pretrained(
        model_directory,
        local_files_only=True,
        use_fast=False,
    )
    inputs = processor(
        images=torch.zeros((1000, 2048, 3), dtype=torch.uint8).numpy(),
        return_tensors="pt",
    )
    assert inputs.pixel_values.shape == (3, 3, 512, 512)

    # Left/right crops vote for class 0; the middle crop votes for class 1.
    class_logits = torch.tensor(
        [
            [[8.0, -8.0, -8.0]],
            [[-8.0, 8.0, -8.0]],
            [[8.0, -8.0, -8.0]],
        ]
    )
    outputs = SimpleNamespace(
        class_queries_logits=class_logits,
        masks_queries_logits=torch.full((3, 1, 8, 8), 8.0),
        patch_offsets=inputs.patch_offsets,
    )
    labels, confidence = postprocess_semantic_prediction(
        outputs,
        processor=processor,
        target_sizes=[(1000, 2048)],
    )

    assert labels.shape == (1000, 2048)
    assert confidence.shape == (1000, 2048)
    assert labels[500, 200].item() == 0
    assert labels[500, 1024].item() == 1
    assert labels[500, 1800].item() == 0
    assert torch.isfinite(confidence).all()
