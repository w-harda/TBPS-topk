import pytest

from attributes.dynamic_topk import select_dynamic_topk
from attributes.scorer import AttributeScore


def make_scores(values: list[float]) -> list[AttributeScore]:
    return [AttributeScore(f"a{index}", 1, 2, value) for index, value in enumerate(values)]


def test_one_dominant_attribute_selects_one() -> None:
    result = select_dynamic_topk(make_scores([10.0, 0.01, 0.01]), beta=0.6, k_min=1, k_max=3)
    assert result.k == 1
    assert result.selected[0].canonical == "a0"


def test_similar_scores_increase_effective_count() -> None:
    result = select_dynamic_topk(make_scores([1.0, 1.0, 1.0]), beta=0.6, k_min=1, k_max=3)
    assert result.effective_attribute_count == pytest.approx(3.0)
    assert result.k == 2


def test_all_zero_uses_k_min() -> None:
    result = select_dynamic_topk(make_scores([0.0, 0.0, 0.0]), beta=0.6, k_min=2, k_max=3)
    assert result.k == 2
    assert result.entropy == 0.0
    assert result.weights == (0.0, 0.0, 0.0)


def test_single_attribute_always_selects_one() -> None:
    result = select_dynamic_topk(make_scores([0.0]), beta=0.01, k_min=1, k_max=5)
    assert result.k == 1
    assert len(result.selected) == 1


def test_k_min_and_k_max_bounds() -> None:
    scores = make_scores([1.0] * 10)
    assert select_dynamic_topk(scores, beta=0.01, k_min=3, k_max=8).k == 3
    assert select_dynamic_topk(scores, beta=10.0, k_min=1, k_max=4).k == 4


def test_no_attributes_returns_empty_result() -> None:
    result = select_dynamic_topk([], beta=0.6)
    assert result.k == 0
    assert result.selected == ()

