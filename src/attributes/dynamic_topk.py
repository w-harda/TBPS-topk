from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

from .scorer import AttributeScore


@dataclass(frozen=True)
class DynamicTopKResult:
    entropy: float
    effective_attribute_count: float
    k: int
    selected: tuple[AttributeScore, ...]
    weights: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["selected"] = [score.to_dict() for score in self.selected]
        result["weights"] = list(self.weights)
        return result


def select_dynamic_topk(
    scores: Iterable[AttributeScore],
    beta: float,
    k_min: int = 1,
    k_max: int | None = None,
) -> DynamicTopKResult:
    score_list = list(scores)
    n = len(score_list)
    if n == 0:
        return DynamicTopKResult(0.0, 0.0, 0, tuple(), tuple())
    if beta <= 0:
        raise ValueError("beta 必须 > 0")
    if k_min < 1:
        raise ValueError("k_min 必须 >= 1")
    if k_max is not None and k_max < k_min:
        raise ValueError("k_max 必须 >= k_min")

    maximum = n if k_max is None else min(k_max, n)
    minimum = min(k_min, maximum)
    total = sum(max(score.score, 0.0) for score in score_list)
    if n == 1:
        weights = (1.0,)
        entropy = 0.0
        effective = 1.0
        k = 1
    elif total <= 0.0:
        weights = tuple(0.0 for _ in score_list)
        entropy = 0.0
        effective = 0.0
        k = minimum
    else:
        weights = tuple(max(score.score, 0.0) / total for score in score_list)
        entropy = -sum(weight * math.log(weight) for weight in weights if weight > 0.0)
        effective = math.exp(entropy)
        k = math.ceil(beta * effective)
        k = max(minimum, min(k, maximum))

    # Python sort 稳定；同分时保留 query 中原始属性顺序。
    ranked = sorted(score_list, key=lambda score: score.score, reverse=True)
    return DynamicTopKResult(
        entropy=entropy,
        effective_attribute_count=effective,
        k=k,
        selected=tuple(ranked[:k]),
        weights=weights,
    )

