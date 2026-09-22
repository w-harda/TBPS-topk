from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .ontology import AttributeOntology


@dataclass(frozen=True)
class CanonicalAttribute:
    raw: str
    canonical: str
    span: tuple[int, int]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["span"] = list(self.span)
        return result


@dataclass(frozen=True)
class MappingCandidate:
    raw: str
    count: int
    candidates: tuple[str, ...]
    match_type: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _alias_pattern(alias: str) -> re.Pattern[str]:
    pieces = [re.escape(piece) for piece in alias.strip().split()]
    body = r"[\s-]+".join(pieces)
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)


class Canonicalizer:
    """使用冻结 alias map 完成 Raw phrase → Canonical 映射。"""

    def __init__(
        self,
        ontology: AttributeOntology,
        alias_to_canonical: Mapping[str, str | Sequence[str]],
    ) -> None:
        self.ontology = ontology
        normalized: dict[str, tuple[str, ...]] = {}
        for alias, raw_canonicals in alias_to_canonical.items():
            clean_alias = " ".join(alias.lower().split())
            if not clean_alias:
                raise ValueError("alias 不能为空")
            canonicals = (
                (raw_canonicals,)
                if isinstance(raw_canonicals, str)
                else tuple(dict.fromkeys(raw_canonicals))
            )
            attribute_ids: set[str] = set()
            for canonical in canonicals:
                if canonical not in ontology.canonical_values:
                    raise ValueError(f"alias {alias!r} 指向 ontology 中不存在的 {canonical!r}")
                attribute_id = ontology.canonical_to_attribute_id[canonical]
                if attribute_id in attribute_ids:
                    raise ValueError(f"alias {alias!r} 在同一属性维度 {attribute_id!r} 中存在冲突")
                attribute_ids.add(attribute_id)
            normalized[clean_alias] = canonicals
        self.alias_to_canonical = normalized
        self._patterns = [
            (alias, canonical, _alias_pattern(alias))
            for alias, canonicals in sorted(
                normalized.items(), key=lambda item: (-len(item[0]), item[0], item[1])
            )
            for canonical in canonicals
        ]

    @classmethod
    def load(
        cls, ontology: AttributeOntology, alias_path: str | Path
    ) -> "Canonicalizer":
        with Path(alias_path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        alias_to_canonical: dict[str, list[str]] = {}
        for mapping in data.get("mappings", []):
            canonical = mapping["canonical"]
            for alias in mapping.get("aliases", []):
                key = " ".join(alias.lower().split())
                alias_to_canonical.setdefault(key, []).append(canonical)
        return cls(ontology, alias_to_canonical)

    def extract(self, caption: str) -> list[CanonicalAttribute]:
        candidates: list[CanonicalAttribute] = []
        for _alias, canonical, pattern in self._patterns:
            for match in pattern.finditer(caption):
                candidates.append(
                    CanonicalAttribute(
                        raw=caption[match.start() : match.end()],
                        canonical=canonical,
                        span=(match.start(), match.end()),
                    )
                )

        # 先保留长匹配；避免 "without a hat" 同时产出 hat:negative 与 hat:positive。
        candidates.sort(
            key=lambda item: (
                -(item.span[1] - item.span[0]),
                item.span[0],
                item.canonical,
            )
        )
        accepted: list[CanonicalAttribute] = []
        attribute_seen: set[str] = set()
        for candidate in candidates:
            attribute_id = self.ontology.canonical_to_attribute_id[candidate.canonical]
            # 跨维度的重叠是有效信息，例如 "black shorts" 同时表达颜色、
            # 下装长度和下装类型；同一维度只保留排序最靠前（通常最长）的匹配。
            if attribute_id in attribute_seen:
                continue
            accepted.append(candidate)
            attribute_seen.add(attribute_id)
        accepted.sort(key=lambda item: (item.span[0], item.span[1], item.canonical))
        return accepted

    def map_phrase(self, phrase: str) -> tuple[str, ...]:
        normalized = " ".join(phrase.lower().split())
        exact = self.alias_to_canonical.get(normalized)
        if exact is not None:
            return exact
        matches = {
            canonical
            for _alias, canonical, pattern in self._patterns
            if pattern.search(normalized)
        }
        return tuple(sorted(matches))

    def generate_candidates(
        self, phrase_counts: Mapping[str, int] | Iterable[tuple[str, int]]
    ) -> tuple[list[MappingCandidate], list[dict[str, object]]]:
        items = phrase_counts.items() if isinstance(phrase_counts, Mapping) else phrase_counts
        mapped: list[MappingCandidate] = []
        unmapped: list[dict[str, object]] = []
        for raw, count in sorted(items, key=lambda item: (-item[1], item[0])):
            normalized = " ".join(raw.lower().split())
            exact = self.alias_to_canonical.get(normalized)
            candidates = exact if exact else self.map_phrase(normalized)
            if candidates:
                mapped.append(
                    MappingCandidate(
                        raw=raw,
                        count=int(count),
                        candidates=candidates,
                        match_type="exact" if exact else "contained_alias",
                    )
                )
            else:
                unmapped.append({"raw": raw, "count": int(count)})
        return mapped, unmapped

    def extract_json(self, caption: str) -> dict[str, object]:
        return {
            "text": caption,
            "attributes": [attribute.to_dict() for attribute in self.extract(caption)],
        }
