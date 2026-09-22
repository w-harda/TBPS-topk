from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

from .gallery_index import GalleryAttributeIndex


@dataclass(frozen=True)
class AttributeScore:
    canonical: str
    full_candidate_count: int
    without_candidate_count: int
    score: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def score_attributes(
    attributes: Iterable[str],
    gallery_index: GalleryAttributeIndex,
    alpha: float = 1.0,
) -> list[AttributeScore]:
    """计算 S(a_i)=ln((|C(A_-i)|+alpha)/(|C(A)|+alpha))。"""

    if alpha <= 0:
        raise ValueError("alpha 必须 > 0")
    canonical = list(dict.fromkeys(attributes))
    full_count = len(gallery_index.candidates(canonical))
    results = []
    for index, attribute in enumerate(canonical):
        without = canonical[:index] + canonical[index + 1 :]
        without_count = len(gallery_index.candidates(without))
        if without_count < full_count:
            raise AssertionError("倒排索引违反 C(A) ⊆ C(A\\{a_i})")
        score = math.log((without_count + alpha) / (full_count + alpha))
        results.append(
            AttributeScore(
                canonical=attribute,
                full_candidate_count=full_count,
                without_candidate_count=without_count,
                score=score,
            )
        )
    return results

