from pathlib import Path

import pytest

from attributes.aptm_extractor import APTMAttributeExtractor
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

