from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PromptDefinition:
    label_value: int
    canonical: str
    semantic: str
    text: str


@dataclass(frozen=True)
class AttributeDefinition:
    index: int
    id: str
    prompts: tuple[PromptDefinition, PromptDefinition]


class AttributeOntology:
    """经校验的 APTM/MALS 27 属性与 54 prompt 定义。"""

    def __init__(self, attributes: tuple[AttributeDefinition, ...], metadata: dict[str, Any]):
        self.attributes = attributes
        self.metadata = metadata
        self._validate()

    @classmethod
    def load(cls, path: str | Path) -> "AttributeOntology":
        source_path = Path(path)
        with source_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        attributes = []
        for raw_attribute in data.get("attributes", []):
            prompts = tuple(PromptDefinition(**prompt) for prompt in raw_attribute["prompts"])
            attributes.append(
                AttributeDefinition(
                    index=int(raw_attribute["index"]),
                    id=str(raw_attribute["id"]),
                    prompts=prompts,  # type: ignore[arg-type]
                )
            )
        metadata = {key: value for key, value in data.items() if key != "attributes"}
        metadata["path"] = str(source_path)
        return cls(tuple(attributes), metadata)

    def _validate(self) -> None:
        if len(self.attributes) != 27:
            raise ValueError(f"APTM ontology 必须包含 27 个属性，实际为 {len(self.attributes)}")
        if [attribute.index for attribute in self.attributes] != list(range(27)):
            raise ValueError("ontology attribute index 必须严格为 0..26 且保持官方顺序")
        ids: set[str] = set()
        canonicals: set[str] = set()
        for attribute in self.attributes:
            if attribute.id in ids:
                raise ValueError(f"重复 attribute id: {attribute.id}")
            ids.add(attribute.id)
            if len(attribute.prompts) != 2:
                raise ValueError(f"{attribute.id} 必须恰好包含两个 prompts")
            if [prompt.label_value for prompt in attribute.prompts] != [0, 1]:
                raise ValueError(f"{attribute.id} 的 prompts 必须按官方 label_value 0,1 排列")
            for prompt in attribute.prompts:
                if not prompt.text.strip():
                    raise ValueError(f"{attribute.id} 存在空 prompt")
                if prompt.canonical in canonicals:
                    raise ValueError(f"重复 canonical value: {prompt.canonical}")
                canonicals.add(prompt.canonical)

    @property
    def prompts(self) -> tuple[PromptDefinition, ...]:
        return tuple(prompt for attribute in self.attributes for prompt in attribute.prompts)

    @property
    def prompt_texts(self) -> tuple[str, ...]:
        return tuple(prompt.text for prompt in self.prompts)

    @property
    def canonical_values(self) -> frozenset[str]:
        return frozenset(prompt.canonical for prompt in self.prompts)

    @property
    def canonical_to_attribute_id(self) -> dict[str, str]:
        return {
            prompt.canonical: attribute.id
            for attribute in self.attributes
            for prompt in attribute.prompts
        }

    @property
    def digest(self) -> str:
        payload = [
            {
                "index": attribute.index,
                "id": attribute.id,
                "prompts": [prompt.__dict__ for prompt in attribute.prompts],
            }
            for attribute in self.attributes
        ]
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def prompt_pair(self, attribute_index: int) -> tuple[PromptDefinition, PromptDefinition]:
        return self.attributes[attribute_index].prompts
