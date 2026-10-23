from types import SimpleNamespace

import numpy as np
import pytest

from roadrunner_mapping.semantics.semantic_transport import (
    BoundedSemanticMatcher,
    PredictionValidationError,
    validate_atomic_prediction,
)


class Stamp:
    def __init__(self, nanoseconds):
        self.nanoseconds = int(nanoseconds)

    def to_nsec(self):
        return self.nanoseconds


def image(values, encoding, stamp=100, frame_id="camera", seq=0):
    array = np.ascontiguousarray(values)
    return SimpleNamespace(
        header=SimpleNamespace(seq=seq, stamp=Stamp(stamp), frame_id=frame_id),
        height=array.shape[0],
        width=array.shape[1],
        encoding=encoding,
        is_bigendian=0,
        step=array.strides[0],
        data=array.tobytes(),
    )


def prediction(
    labels=None,
    confidence=None,
    *,
    stamp=100,
    frame_id="camera",
    inference_sec=0.25,
):
    labels = np.asarray(
        [[1, 63], [50, 0]] if labels is None else labels,
        dtype=np.uint8,
    )
    confidence = np.asarray(
        [[0.9, 0.8], [0.7, 0.0]] if confidence is None else confidence,
        dtype="<f4",
    )
    return SimpleNamespace(
        header=SimpleNamespace(seq=99, stamp=Stamp(stamp), frame_id=frame_id),
        labels=image(labels, "mono8", stamp, frame_id),
        confidence=image(confidence, "32FC1", stamp, frame_id),
        inference_sec=inference_sec,
    )


def test_valid_atomic_prediction_decodes_owned_arrays():
    decoded = validate_atomic_prediction(prediction())

    assert decoded.stamp_ns == 100
    assert decoded.frame_id == "camera"
    assert decoded.labels.dtype == np.uint8
    assert decoded.confidence.dtype == np.float32
    assert decoded.labels.tolist() == [[1, 63], [50, 0]]


def test_ros_publisher_outer_sequence_rewrite_does_not_break_atomic_message():
    msg = prediction()
    msg.header.seq = 123
    msg.labels.header.seq = 7
    msg.confidence.header.seq = 7
    assert validate_atomic_prediction(msg).stamp_ns == 100


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda msg: setattr(msg.labels.header, "frame_id", "other"), "header_mismatch"),
        (lambda msg: setattr(msg.confidence, "width", 1), "dimension_mismatch"),
        (lambda msg: setattr(msg.labels, "encoding", "8UC1"), "bad_encoding"),
        (lambda msg: setattr(msg.confidence, "encoding", "64FC1"), "bad_encoding"),
        (lambda msg: setattr(msg, "inference_sec", float("inf")), "nonfinite_value"),
    ],
)
def test_invalid_atomic_metadata_has_a_specific_reason(mutation, reason):
    msg = prediction()
    mutation(msg)
    with pytest.raises(PredictionValidationError) as caught:
        validate_atomic_prediction(msg)
    assert caught.value.reason == reason


def test_out_of_range_labels_and_nonfinite_confidence_are_rejected():
    with pytest.raises(PredictionValidationError, match="label_range") as labels_error:
        validate_atomic_prediction(prediction(labels=[[64]], confidence=[[1.0]]))
    assert labels_error.value.reason == "label_range"

    with pytest.raises(PredictionValidationError, match="nonfinite_value") as conf_error:
        validate_atomic_prediction(prediction(labels=[[1]], confidence=[[np.nan]]))
    assert conf_error.value.reason == "nonfinite_value"


def test_matcher_accepts_sensor_first_and_prediction_first():
    matcher = BoundedSemanticMatcher(max_age_sec=5.0, max_entries=64)
    assert matcher.add_sensor(100, "sensor-a", now=1.0) is None
    assert matcher.add_prediction(100, "prediction-a", now=2.0) == (
        "sensor-a",
        "prediction-a",
    )

    assert matcher.add_prediction(200, "prediction-b", now=3.0) is None
    assert matcher.add_sensor(200, "sensor-b", now=4.0) == (
        "sensor-b",
        "prediction-b",
    )


def test_matcher_suppresses_duplicates_once_processed():
    matcher = BoundedSemanticMatcher(max_age_sec=5.0, max_entries=64)
    matcher.add_sensor(100, object(), now=1.0)
    assert matcher.add_prediction(100, object(), now=1.1) is not None
    assert matcher.add_prediction(100, object(), now=1.2) is None
    assert matcher.add_sensor(100, object(), now=1.3) is None
    assert matcher.counters["duplicate_prediction"] == 1
    assert matcher.counters["duplicate_sensor"] == 1


def test_matcher_evicts_by_age_and_capacity_with_independent_counters():
    matcher = BoundedSemanticMatcher(max_age_sec=5.0, max_entries=2)
    matcher.add_sensor(1, "s1", now=0.0)
    matcher.add_sensor(2, "s2", now=1.0)
    matcher.add_sensor(3, "s3", now=2.0)
    assert matcher.counters["sensor_capacity_eviction"] == 1

    matcher.add_prediction(10, "p10", now=2.0)
    matcher.add_prediction(11, "p11", now=3.0)
    matcher.add_prediction(12, "p12", now=4.0)
    assert matcher.counters["prediction_capacity_eviction"] == 1

    matcher.evict(now=10.0)
    assert matcher.counters["sensor_age_eviction"] == 2
    assert matcher.counters["prediction_age_eviction"] == 2
    assert matcher.pending_sensor_count == 0
    assert matcher.pending_prediction_count == 0
