from pathlib import Path

import pytest

import hashlib
import json

from attributes.aptm_extractor import APTMAttributeExtractor, OfficialAPTMBackend
from attributes.ontology import AttributeOntology


ROOT = Path(__file__).resolve().parents[1]


class FakeBackend:
    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows
        self.calls = 0

    def score(self, image_paths, prompt_texts):
        self.calls += 1
        assert len(prompt_texts) == 54
        return self.rows


def test_selected_prompt_controls_semantics_and_cache_shape() -> None:
    ontology = AttributeOntology.load(ROOT / "ontology" / "aptm_attributes.yaml")
    logits = [0.0] * 54
    logits[0] = 1.0  # gender:female (pair 左侧)
    logits[1] = 3.0  # gender:male (pair 右侧，应选中)
    logits[6] = 4.0  # hat:positive (pair 左侧，应选中)
    logits[7] = 1.0
    backend = FakeBackend([logits])
    extractor = APTMAttributeExtractor(ontology, backend)
    payload = extractor.extract_gallery([("image-1", Path("unused.jpg"))])
    predictions = payload["images"][0]["predictions"]
    assert len(predictions) == 27
    assert predictions[0]["canonical"] == "gender:male"
    assert predictions[0]["selected_prompt"] == "the person is a man"
    assert predictions[3]["canonical"] == "hat:positive"
    assert predictions[3]["label_value"] == 0
    assert predictions[3]["confidence"] > 0.5
    assert backend.calls == 1


def test_rejects_non_54_logit_rows() -> None:
    ontology = AttributeOntology.load(ROOT / "ontology" / "aptm_attributes.yaml")
    extractor = APTMAttributeExtractor(ontology, FakeBackend([]))
    with pytest.raises(ValueError, match="54"):
        extractor.prediction_from_logits("x", [0.0] * 53)


def make_preflight_backend(tmp_path: Path, *, load_swin: bool) -> OfficialAPTMBackend:
    aptm_root = tmp_path / "APTM"
    source_file = aptm_root / "models" / "aptm.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("# pinned source\n", encoding="utf-8")
    source_manifest = tmp_path / "aptm_source.json"
    source_manifest.write_text(
        json.dumps(
            {
                "commit": "test-commit",
                "files": {
                    "models/aptm.py": hashlib.sha256(b"# pinned source\n").hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )
    official_config = aptm_root / "configs" / "Retrieval_gene.yaml"
    official_config.parent.mkdir(parents=True)
    official_config.write_text("mlm: true\n", encoding="utf-8")
    vision_config = aptm_root / "configs" / "config_swinB_384.json"
    vision_config.write_text(json.dumps({"h": 384, "w": 128}), encoding="utf-8")
    checkpoint = tmp_path / "aptm.pth"
    checkpoint.write_bytes(b"checkpoint")
    bert_path = tmp_path / "bert-base-uncased"
    bert_path.mkdir()
    for name in ("config.json", "vocab.txt", "pytorch_model.bin"):
        (bert_path / name).write_bytes(b"x")
    return OfficialAPTMBackend(
        aptm_root=aptm_root,
        source_manifest=source_manifest,
        official_config=official_config,
        checkpoint=checkpoint,
        bert_path=bert_path,
        vision_config=vision_config,
        swin_path=tmp_path / "swin.pth",
        load_swin_pretrained=load_swin,
        prompt_feature_cache=tmp_path / "cache" / "prompts.pt",
    )


def test_preflight_complete_checkpoint_mode_does_not_require_swin(tmp_path: Path) -> None:
    backend = make_preflight_backend(tmp_path, load_swin=False)
    report = backend.preflight(["prompt"] * 54, check_runtime=False)
    assert report.ok
    swin_check = next(check for check in report.checks if check.name == "Swin initialization checkpoint")
    assert swin_check.status == "PASS"
    assert "not required" in swin_check.detail


def test_preflight_swin_initialization_mode_requires_checkpoint(tmp_path: Path) -> None:
    backend = make_preflight_backend(tmp_path, load_swin=True)
    report = backend.preflight(["prompt"] * 54, check_runtime=False)
    assert not report.ok
    swin_check = next(check for check in report.checks if check.name == "Swin initialization checkpoint")
    assert swin_check.status == "FAIL"


def test_runtime_vision_config_rewrites_swin_checkpoint_path(tmp_path: Path) -> None:
    backend = make_preflight_backend(tmp_path, load_swin=True)
    runtime_config = backend._write_runtime_vision_config(tmp_path / "runtime-vision.json")
    payload = json.loads(runtime_config.read_text(encoding="utf-8"))
    assert payload["ckpt"] == str(backend.swin_path)


def test_preflight_reports_missing_bert_artifact_before_model_load(tmp_path: Path) -> None:
    backend = make_preflight_backend(tmp_path, load_swin=False)
    (backend.bert_path / "pytorch_model.bin").unlink()
    report = backend.preflight(["prompt"] * 54, check_runtime=False)
    bert_check = next(check for check in report.checks if check.name == "BERT local directory")
    assert bert_check.status == "FAIL"
    assert "pytorch_model.bin" in bert_check.detail


class FakeTorchLoader:
    @staticmethod
    def load(_path, map_location=None):
        return {"model": {"placeholder": object()}}


class FakeIncompatibleKeys:
    def __init__(self, missing_keys):
        self.missing_keys = missing_keys
        self.unexpected_keys = []


class FakeCheckpointModel:
    def __init__(self, missing_keys):
        self.missing_keys = missing_keys

    def load_state_dict(self, _state_dict, strict=False):
        assert strict is False
        return FakeIncompatibleKeys(self.missing_keys)

    def named_parameters(self):
        return iter(
            [
                ("vision_encoder.weight", object()),
                ("text_encoder.weight", object()),
                ("vision_proj.weight", object()),
                ("text_proj.weight", object()),
                ("temp", object()),
                ("itm_head.weight", object()),
            ]
        )


def test_complete_checkpoint_validation_rejects_missing_inference_parameter(tmp_path: Path) -> None:
    backend = make_preflight_backend(tmp_path, load_swin=False)
    model = FakeCheckpointModel(["vision_proj.weight", "itm_head.weight"])
    with pytest.raises(RuntimeError, match="完整 checkpoint"):
        backend._load_complete_checkpoint(FakeTorchLoader, model)


def test_complete_checkpoint_validation_allows_unused_head_to_be_missing(tmp_path: Path) -> None:
    backend = make_preflight_backend(tmp_path, load_swin=False)
    model = FakeCheckpointModel(["itm_head.weight"])
    backend._load_complete_checkpoint(FakeTorchLoader, model)
    assert backend._checkpoint_load_report == {
        "missing_keys": ["itm_head.weight"],
        "unexpected_keys": [],
    }
