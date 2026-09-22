import math

import pytest

from attributes.gallery_index import GalleryAttributeIndex
from attributes.scorer import score_attributes


def make_index() -> GalleryAttributeIndex:
    return GalleryAttributeIndex(
        {
            "1": {"a", "b", "c"},
            "2": {"a", "b", "c"},
            "3": {"a", "b"},
            "4": {"a"},
            "5": {"b"},
        }
    )


def test_candidate_monotonicity() -> None:
    index = make_index()
    full = index.candidates(["a", "b", "c"])
    for removed in ("a", "b", "c"):
        reduced = index.candidates([item for item in ("a", "b", "c") if item != removed])
        assert full <= reduced


def test_score_formula_exact_example() -> None:
    image_attributes = {f"full-{i}": {"a", "b"} for i in range(5)}
    image_attributes.update({f"extra-{i}": {"b"} for i in range(45)})
    index = GalleryAttributeIndex(image_attributes)
    score = score_attributes(["a", "b"], index, alpha=1.0)[0]
    assert score.full_candidate_count == 5
    assert score.without_candidate_count == 50
    assert score.score == pytest.approx(math.log(51 / 6))


def test_empty_attribute_set_matches_whole_gallery() -> None:
    index = make_index()
    assert index.candidates([]) == index.all_image_ids


def test_missing_canonical_produces_empty_full_candidates_but_valid_score() -> None:
    index = make_index()
    score = score_attributes(["missing"], index, alpha=1.0)[0]
    assert score.full_candidate_count == 0
    assert score.without_candidate_count == 5
    assert score.score == pytest.approx(math.log(6))

