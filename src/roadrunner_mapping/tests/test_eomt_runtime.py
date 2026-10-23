import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from roadrunner_mapping.semantics.eomt_runtime import (
    LatestFrameSlot,
    PredictionResultWriter,
    decode_rgb_image,
    load_eomt_model_compatibly,
    validate_model_directory,
)
from roadrunner_mapping.semantics.semantic_profiles import get_eomt_profile


def test_rgb_decoder_handles_row_padding_and_bgr():
    message = SimpleNamespace(
        height=1,
        width=2,
        step=8,
        encoding="bgr8",
        data=bytes([3, 2, 1, 6, 5, 4, 99, 99]),
    )
    assert decode_rgb_image(message).tolist() == [[[1, 2, 3], [4, 5, 6]]]


def test_latest_frame_slot_replaces_an_unconsumed_frame():
    slot = LatestFrameSlot()
    assert slot.put("first") is False
    assert slot.put("latest") is True
    assert slot.get(timeout=0.01) == "latest"
    assert slot.replaced_count == 1


def test_prediction_result_writer_can_be_enabled_and_disabled(tmp_path: Path):
    pytest.importorskip("PIL")
    writer = PredictionResultWriter(tmp_path / "predictions", ("dirt", "grass"))
    assert writer.enabled is False

    header = SimpleNamespace(stamp=SimpleNamespace(to_nsec=lambda: 123456789))
    labels = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
    confidence = np.asarray([[0.2, 0.8], [1.0, 0.0]], dtype="<f4")
    labels_message = SimpleNamespace(
        height=2, width=2, step=2, data=labels.tobytes()
    )
    confidence_message = SimpleNamespace(
        height=2, width=2, step=8, data=confidence.tobytes()
    )
    prediction = SimpleNamespace(
        header=header,
        labels=labels_message,
        confidence=confidence_message,
        inference_sec=0.25,
    )

    assert writer.submit_message(prediction) is False
    writer.enable()
    assert writer.submit_message(prediction) is True
    writer.disable()
    assert writer.submit_message(prediction) is False
    writer.close()

    assert writer.saved_count == 1
    assert (tmp_path / "predictions" / "class_legend.json").is_file()
    assert len(list((tmp_path / "predictions").glob("*_mask.png"))) == 1
    saved_labels = next((tmp_path / "predictions").glob("*_labels.npy"))
    np.testing.assert_array_equal(np.load(saved_labels), labels)


def test_model_directory_checks_hashes_and_exact_class_order(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    classes = ("undefined", "grass")
    files = {
        "config.json": json.dumps(
            {"id2label": {"0": "undefined", "1": "grass"}}
        ).encode(),
        "preprocessor_config.json": json.dumps(
            {
                "image_processor_type": "EomtImageProcessorFast",
                "do_split_image": True,
            }
        ).encode(),
        "model.safetensors": b"weights",
    }
    hashes = []
    for name, data in files.items():
        (model / name).write_bytes(data)
        hashes.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
    manifest = tmp_path / "model.sha256"
    manifest.write_text("\n".join(hashes) + "\n")

    validate_model_directory(model, manifest, expected_classes=classes)

    (model / "model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256"):
        validate_model_directory(model, manifest, expected_classes=classes)


def test_model_directory_accepts_rugd_slow_processor_contract(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    files = {
        "config.json": json.dumps(
            {"id2label": {"0": "dirt", "1": "grass"}}
        ).encode(),
        "preprocessor_config.json": json.dumps(
            {
                "image_processor_type": "EomtImageProcessor",
                "do_split_image": True,
            }
        ).encode(),
        "model.safetensors": b"weights",
    }
    hashes = []
    for name, data in files.items():
        (model / name).write_bytes(data)
        hashes.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
    manifest = tmp_path / "model.sha256"
    manifest.write_text("\n".join(hashes) + "\n")

    validate_model_directory(model, manifest, expected_classes=("dirt", "grass"))


def test_rugd24_eomt_profile_matches_latest_checkpoint_contract():
    profile = get_eomt_profile("rugd24")

    assert len(profile.classes) == 24
    assert profile.classes[0] == "dirt"
    assert profile.classes[23] == "picnic-table"
    assert set(profile.semantic_risk) == set(range(24))
    assert profile.grass_class_ids == frozenset({2})
    assert profile.ground_class_ids == frozenset({0, 1, 2, 9, 10, 12, 13, 21, 22})
    assert profile.dynamic_class_ids == frozenset({7, 15, 16})


def test_unknown_eomt_profile_is_rejected():
    with pytest.raises(ValueError, match="semantic profile"):
        get_eomt_profile("not-a-profile")


def test_load_eomt_model_compatibly_binds_config_and_uses_eager_attention():
    calls = {}

    class FakeConfig:
        @classmethod
        def from_pretrained(cls, model_directory, local_files_only):
            calls["config"] = (model_directory, local_files_only)
            return "loaded-config"

    class FakeModel:
        config_class = "EomtConfig"

        @classmethod
        def from_pretrained(cls, model_directory, **kwargs):
            calls["model"] = (model_directory, kwargs)
            return "loaded-model"

    assert load_eomt_model_compatibly(
        FakeModel,
        FakeConfig,
        "/models/eomt",
    ) == "loaded-model"
    assert FakeModel.config_class is FakeConfig
    assert calls == {
        "config": ("/models/eomt", True),
        "model": (
            "/models/eomt",
            {
                "config": "loaded-config",
                "local_files_only": True,
                "attn_implementation": "eager",
            },
        ),
    }


def test_attention_selection_preserves_jetson_and_checks_explicit_sdpa():
    from roadrunner_mapping.semantics.eomt_runtime import resolve_attention_backend
    assert resolve_attention_backend('auto', sdpa_available=True, machine='x86_64') == 'sdpa'
    assert resolve_attention_backend('auto', sdpa_available=True, machine='aarch64') == 'eager'
    assert resolve_attention_backend('auto', sdpa_available=False, machine='x86_64') == 'eager'
    assert resolve_attention_backend('eager', sdpa_available=True, machine='x86_64') == 'eager'
    assert resolve_attention_backend('sdpa', sdpa_available=True, machine='aarch64') == 'sdpa'
    with pytest.raises(ValueError, match='SDPA'):
        resolve_attention_backend('sdpa', sdpa_available=False, machine='x86_64')
    with pytest.raises(ValueError, match='attention'):
        resolve_attention_backend('typo', sdpa_available=True, machine='x86_64')


def test_loader_accepts_explicit_sdpa_without_retrying_model_errors():
    class Config:
        @classmethod
        def from_pretrained(cls, *args, **kwargs): return object()
    class Model:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            assert kwargs['attn_implementation'] == 'sdpa'
            raise RuntimeError('CUDA out of memory')
    with pytest.raises(RuntimeError, match='CUDA out of memory'):
        load_eomt_model_compatibly(Model, Config, '/model', attention_backend='sdpa')


def test_inference_pacer_waits_before_selecting_latest_and_can_stop():
    from roadrunner_mapping.semantics.eomt_runtime import InferencePacer
    now=[10.0]
    slot=LatestFrameSlot();slot.put('old')
    class Event:
        def wait(self, duration):
            assert duration == pytest.approx(.4)
            slot.put('newest')
            now[0]+=duration
            return False
    pacer=InferencePacer(2., clock=lambda:now[0])
    pacer.mark_started()
    now[0]+=.1
    assert pacer.wait(Event()) is False
    assert slot.get(timeout=0) == 'newest'
    pacer.mark_started()
    now[0]+=1.0
    class NoWait:
        def wait(self, duration): raise AssertionError('slow inference should not sleep')
    assert pacer.wait(NoWait()) is False
    unlimited=InferencePacer(0,clock=lambda:now[0]);unlimited.mark_started()
    assert unlimited.wait(NoWait()) is False
    pacer.mark_started()
    class Stopped:
        def wait(self,duration):return True
    assert pacer.wait(Stopped()) is True


@pytest.mark.parametrize('hz', [-1, float('nan'), float('inf')])
def test_inference_pacer_rejects_invalid_hz(hz):
    from roadrunner_mapping.semantics.eomt_runtime import InferencePacer
    with pytest.raises(ValueError, match='max_inference_hz'):
        InferencePacer(hz)


def test_actual_worker_selects_frame_after_pacing_wait():
    """Exercise the actual ROS worker without constructing a CUDA model."""
    import ast
    from roadrunner_mapping.semantics.eomt_runtime import InferencePacer
    path = Path(__file__).resolve().parents[1] / 'scripts/semantic_node.py'
    tree = ast.parse(path.read_text())
    node_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EomtSemanticNode')
    worker = next(n for n in node_class.body if isinstance(n, ast.FunctionDef) and n.name == '_worker')
    namespace = {'rospy': SimpleNamespace(is_shutdown=lambda: False), 'time': SimpleNamespace(monotonic=lambda: 10.)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[worker], type_ignores=[])), str(path), 'exec'), namespace)
    now = [10.]
    slot = LatestFrameSlot()
    slot.put('stale-before-wait')
    class Event:
        stopped = False
        def is_set(self): return self.stopped
        def wait(self, duration):
            now[0] += duration
            slot.put('fresh-after-wait')
            return False
    event = Event()
    pacer = InferencePacer(2, clock=lambda: now[0])
    pacer.mark_started()
    seen = []
    def infer(message):
        seen.append(message)
        event.stopped = True
        return SimpleNamespace(inference_sec=.1)
    node = SimpleNamespace(stop_event=event, pacer=pacer, slot=slot, _infer=infer,
        prediction_pub=SimpleNamespace(publish=lambda p:None),
        prediction_writer=SimpleNamespace(submit_message=lambda p:None), processed=0)
    namespace['_worker'](node)
    assert seen == ['fresh-after-wait']
    assert node.processed == 1
    assert now[0] == pytest.approx(10.5)
