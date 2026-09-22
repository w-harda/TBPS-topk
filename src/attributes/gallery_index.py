from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping


class GalleryAttributeIndex:
    """Canonical Attribute → Gallery image IDs 的倒排索引。"""

    def __init__(self, image_attributes: Mapping[str, Iterable[str]]) -> None:
        self.image_attributes = {
            str(image_id): frozenset(attributes)
            for image_id, attributes in image_attributes.items()
        }
        inverted: dict[str, set[str]] = defaultdict(set)
        for image_id, attributes in self.image_attributes.items():
            for attribute in attributes:
                inverted[attribute].add(image_id)
        self.inverted = {attribute: frozenset(ids) for attribute, ids in inverted.items()}
        self.all_image_ids = frozenset(self.image_attributes)

    @classmethod
    def from_cache(cls, path: str | Path) -> "GalleryAttributeIndex":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        images = payload.get("images") if isinstance(payload, dict) else payload
        if not isinstance(images, list):
            raise ValueError("gallery cache 的 images 必须是列表")
        image_attributes: dict[str, list[str]] = {}
        for image in images:
            image_id = str(image["image_id"])
            predictions = image.get("predictions", [])
            image_attributes[image_id] = [
                prediction["canonical"]
                for prediction in predictions
                if "canonical" in prediction
            ]
        return cls(image_attributes)

    def candidates(self, attributes: Iterable[str]) -> frozenset[str]:
        unique_attributes = tuple(dict.fromkeys(attributes))
        if not unique_attributes:
            return self.all_image_ids
        postings = [self.inverted.get(attribute, frozenset()) for attribute in unique_attributes]
        if not postings:
            return self.all_image_ids
        result = set(postings[0])
        for posting in postings[1:]:
            result.intersection_update(posting)
            if not result:
                break
        return frozenset(result)

