from pathlib import Path

from attributes.canonicalizer import Canonicalizer
from attributes.dynamic_topk import select_dynamic_topk
from attributes.gallery_index import GalleryAttributeIndex
from attributes.ontology import AttributeOntology
from attributes.scorer import score_attributes


ROOT = Path(__file__).resolve().parents[1]


def test_query_to_dynamic_topk_synthetic_pipeline() -> None:
    ontology = AttributeOntology.load(ROOT / "ontology" / "aptm_attributes.yaml")
    canonicalizer = Canonicalizer.load(ontology, ROOT / "ontology" / "alias_map.json")
    query = "A man wearing a red shirt, black pants and carrying a backpack."
    canonical = [attribute.canonical for attribute in canonicalizer.extract(query)]
    expected = {
        "gender:male",
        "upper_red:positive",
        "lower_black:positive",
        "lower_type:pants_or_shorts",
        "backpack:positive",
    }
    assert expected <= set(canonical)

    gallery = GalleryAttributeIndex(
        {
            "g1": expected,
            "g2": expected,
            "g3": expected - {"backpack:positive"},
            "g4": expected - {"upper_red:positive"},
            "g5": {"gender:male"},
        }
    )
    assert gallery.candidates(canonical) == frozenset({"g1", "g2"})
    scores = score_attributes(canonical, gallery, alpha=1.0)
    result = select_dynamic_topk(scores, beta=0.6, k_min=1, k_max=5)
    assert 1 <= result.k <= 5
    assert set(score.canonical for score in result.selected) <= set(canonical)

