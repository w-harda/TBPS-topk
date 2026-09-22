from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .canonicalizer import Canonicalizer
from .dynamic_topk import select_dynamic_topk
from .gallery_index import GalleryAttributeIndex
from .scorer import score_attributes


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    text: str


def load_gallery_manifest(
    path: str | Path,
    dataset_root: str | Path,
    limit: int | None = None,
    split_name: str | None = None,
) -> list[tuple[str, Path]]:
    if limit is not None and limit < 1:
        raise ValueError("gallery limit 必须 >= 1")
    manifest_path = Path(path)
    root = Path(dataset_root)
    with manifest_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("gallery manifest 必须是记录列表")
    images: list[tuple[str, Path]] = []
    seen_paths: set[Path] = set()
    for index, record in enumerate(records):
        if limit is not None and len(images) >= limit:
            break
        if not isinstance(record, dict):
            raise ValueError(f"gallery manifest 第 {index} 项必须是对象")
        record_split = record.get("split")
        if split_name and record_split is not None and str(record_split).lower() != split_name.lower():
            continue
        value = record.get("path", record.get("image", record.get("file_path")))
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"gallery manifest 第 {index} 项缺少 path/image/file_path")
        relative = Path(value).expanduser()
        image_path = relative if relative.is_absolute() else root / relative
        if not image_path.is_file():
            raise FileNotFoundError(f"gallery image 不存在: {image_path}")
        resolved_path = image_path.resolve()
        if resolved_path in seen_paths:
            continue
        person_id = str(record.get("image_id", record.get("id", index)))
        image_id = f"{person_id}:{relative.as_posix()}"
        seen_paths.add(resolved_path)
        images.append((image_id, resolved_path))
    if not images:
        raise ValueError(f"gallery manifest 没有可用图片: {manifest_path}")
    return images


def load_query_records(
    path: str | Path,
    limit: int | None = None,
    split_name: str | None = "test",
) -> list[QueryRecord]:
    """兼容 APTM 单 caption 标注与 CUHK-PEDES 原始 captions 列表格式。"""

    if limit is not None and limit < 1:
        raise ValueError("query limit 必须 >= 1")
    annotation_path = Path(path)
    with annotation_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        if split_name and isinstance(payload.get(split_name), list):
            records = payload[split_name]
        elif isinstance(payload.get("queries"), list):
            records = payload["queries"]
        else:
            raise ValueError("query JSON 对象必须包含配置的 split 列表或 queries 列表")
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("query annotations 必须是记录列表或包含 split/queries 的对象")

    queries: list[QueryRecord] = []
    used_ids: Counter[str] = Counter()
    for record_index, record in enumerate(records):
        if limit is not None and len(queries) >= limit:
            break
        if not isinstance(record, dict):
            continue
        record_split = record.get("split")
        if split_name and record_split is not None and str(record_split).lower() != split_name.lower():
            continue
        captions = record.get(
            "captions", record.get("caption", record.get("text", record.get("description")))
        )
        if isinstance(captions, str):
            captions = [captions]
        if not isinstance(captions, list):
            continue
        explicit_id = record.get("query_id", record.get("caption_id"))
        base_id = str(record.get("image_id", record.get("id", record_index)))
        for caption_index, caption in enumerate(captions):
            if limit is not None and len(queries) >= limit:
                break
            if not isinstance(caption, str) or not caption.strip():
                continue
            if explicit_id is not None and len(captions) == 1:
                candidate_id = str(explicit_id)
            elif explicit_id is not None:
                candidate_id = f"{explicit_id}:{caption_index}"
            else:
                candidate_id = f"{base_id}:{record_index}:{caption_index}"
            duplicate_index = used_ids[candidate_id]
            used_ids[candidate_id] += 1
            query_id = candidate_id if duplicate_index == 0 else f"{candidate_id}:dup{duplicate_index}"
            queries.append(QueryRecord(query_id=query_id, text=caption))
    if not queries:
        raise ValueError(f"未从 query annotations 读取到有效 query: {annotation_path}")
    return queries


def evaluate_query_subset(
    queries: Sequence[QueryRecord],
    canonicalizer: Canonicalizer,
    gallery_index: GalleryAttributeIndex,
    *,
    alpha: float,
    beta: float,
    k_min: int,
    k_max: int | None,
    low_coverage_max_attributes: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if low_coverage_max_attributes < 0:
        raise ValueError("low_coverage_max_attributes 必须 >= 0")
    results: list[dict[str, Any]] = []
    canonical_frequency: Counter[str] = Counter()
    k_distribution: Counter[int] = Counter()
    zero_records: list[dict[str, str]] = []
    low_records: list[dict[str, Any]] = []
    total_attributes = 0
    total_k = 0

    for query in queries:
        extracted = canonicalizer.extract(query.text)
        canonical = [attribute.canonical for attribute in extracted]
        scores = score_attributes(canonical, gallery_index, alpha=alpha)
        topk = select_dynamic_topk(
            scores,
            beta=beta,
            k_min=k_min,
            k_max=k_max,
        )
        attribute_count = len(canonical)
        if attribute_count == 0:
            coverage_status = "none"
            zero_records.append({"query_id": query.query_id, "text": query.text})
        elif attribute_count <= low_coverage_max_attributes:
            coverage_status = "low"
            low_records.append(
                {
                    "query_id": query.query_id,
                    "text": query.text,
                    "attribute_count": attribute_count,
                }
            )
        else:
            coverage_status = "covered"
        canonical_frequency.update(canonical)
        k_distribution[topk.k] += 1
        total_attributes += attribute_count
        total_k += topk.k
        results.append(
            {
                "query_id": query.query_id,
                "text": query.text,
                "attributes": [attribute.to_dict() for attribute in extracted],
                "attribute_count": attribute_count,
                "coverage_status": coverage_status,
                "candidate_count": len(gallery_index.candidates(canonical)),
                "scores": [score.to_dict() for score in scores],
                "dynamic_topk": topk.to_dict(),
                "dynamic_k": topk.k,
                "selected_topk_attributes": [score.canonical for score in topk.selected],
            }
        )

    total_queries = len(queries)
    with_attributes = total_queries - len(zero_records)
    summary = {
        "warning": "subset_sanity_check_only_not_formal_statistics",
        "total_queries": total_queries,
        "queries_with_attributes": with_attributes,
        "zero_attribute_queries": len(zero_records),
        "low_coverage_queries": len(low_records),
        "low_coverage_max_attributes": low_coverage_max_attributes,
        "attribute_extraction_query_rate": with_attributes / total_queries if total_queries else 0.0,
        "average_attribute_count": total_attributes / total_queries if total_queries else 0.0,
        "average_k": total_k / total_queries if total_queries else 0.0,
        "k_distribution": {
            str(key): value for key, value in sorted(k_distribution.items())
        },
        "canonical_attribute_frequency": dict(
            sorted(canonical_frequency.items(), key=lambda item: (-item[1], item[0]))
        ),
        "zero_attribute_query_details": zero_records,
        "low_coverage_query_details": low_records,
    }
    return results, summary
