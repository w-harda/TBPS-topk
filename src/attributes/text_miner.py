from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence


@dataclass(frozen=True)
class CaptionRecord:
    caption_id: str
    text: str


@dataclass(frozen=True)
class RawPhraseOccurrence:
    caption_id: str
    caption: str
    raw: str
    normalized: str
    span: tuple[int, int]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["span"] = list(self.span)
        return result


def _captions_from_records(records: Sequence[dict], split_name: str, require_split: bool) -> list[CaptionRecord]:
    result: list[CaptionRecord] = []
    for record_index, record in enumerate(records):
        record_split = record.get("split")
        if require_split and record_split is None:
            raise ValueError("标注记录缺少 split 字段，无法证明 vocabulary 仅来自 train split")
        if record_split is not None and str(record_split).lower() != split_name.lower():
            continue
        values = record.get("captions", record.get("caption", record.get("description")))
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            continue
        base_id = str(record.get("image_id", record.get("id", record_index)))
        for caption_index, text in enumerate(values):
            if isinstance(text, str) and text.strip():
                result.append(CaptionRecord(f"{base_id}:{caption_index}", text))
    return result


def load_train_captions(
    path: str | Path,
    split_name: str = "train",
    require_split: bool = False,
) -> list[CaptionRecord]:
    """读取 train captions；支持 split 映射或带 split 字段的记录列表。"""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and split_name in data:
        records = data[split_name]
        if not isinstance(records, list):
            raise ValueError(f"JSON 中 {split_name!r} 必须是列表")
        return _captions_from_records(records, split_name, require_split=False)
    if not isinstance(data, list):
        raise ValueError("caption 标注必须是记录列表，或包含 train 列表的映射")
    captions = _captions_from_records(data, split_name, require_split)
    if not captions:
        raise ValueError("未找到任何 train caption，请检查 split 名称和标注格式")
    return captions


class NounPhraseMiner:
    """基于 NLTK POS + RegexpParser 的名词短语提取器。"""

    DEFAULT_GRAMMAR = r"NP: {<JJ.*>*<NN.*>+}"

    def __init__(
        self,
        max_phrase_length: int = 3,
        grammar: str = DEFAULT_GRAMMAR,
        tagger: Callable[[list[str]], list[tuple[str, str]]] | None = None,
    ) -> None:
        if max_phrase_length < 1:
            raise ValueError("max_phrase_length 必须 >= 1")
        self.max_phrase_length = max_phrase_length
        self.grammar = grammar
        self.tagger = tagger

    def extract(self, caption_id: str, caption: str) -> list[RawPhraseOccurrence]:
        try:
            import nltk
            from nltk.tokenize import TreebankWordTokenizer
        except ImportError as exc:
            raise RuntimeError("请先安装 requirements.txt 中的 nltk") from exc

        tokenizer = TreebankWordTokenizer()
        spans = list(tokenizer.span_tokenize(caption))
        tokens = [caption[start:end] for start, end in spans]
        try:
            tagged = self.tagger(tokens) if self.tagger else nltk.pos_tag(tokens)
        except LookupError as exc:
            raise RuntimeError(
                "缺少 NLTK POS tagger 数据。联网机器执行 "
                "python -m nltk.downloader averaged_perceptron_tagger_eng，"
                "再通过配置 text_mining.nltk_data_dir 使用离线目录。"
            ) from exc

        tree = nltk.RegexpParser(self.grammar).parse(tagged)
        results: list[RawPhraseOccurrence] = []
        token_cursor = 0
        for node in tree:
            leaf_count = len(node.leaves()) if hasattr(node, "leaves") else 1
            if hasattr(node, "label") and node.label() == "NP" and leaf_count <= self.max_phrase_length:
                start = spans[token_cursor][0]
                end = spans[token_cursor + leaf_count - 1][1]
                raw = caption[start:end]
                results.append(
                    RawPhraseOccurrence(
                        caption_id=caption_id,
                        caption=caption,
                        raw=raw,
                        normalized=" ".join(raw.lower().split()),
                        span=(start, end),
                    )
                )
            token_cursor += leaf_count
        return results

    def build_vocabulary(
        self,
        captions: Iterable[CaptionRecord],
        min_freq: int = 40,
    ) -> dict[str, object]:
        if min_freq < 1:
            raise ValueError("min_freq 必须 >= 1")
        counts: Counter[str] = Counter()
        occurrences: dict[str, list[RawPhraseOccurrence]] = defaultdict(list)
        caption_count = 0
        for record in captions:
            caption_count += 1
            for occurrence in self.extract(record.caption_id, record.text):
                counts[occurrence.normalized] += 1
                occurrences[occurrence.normalized].append(occurrence)
        phrases = []
        for phrase, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            if count < min_freq:
                continue
            phrases.append(
                {
                    "phrase": phrase,
                    "count": count,
                    "occurrences": [item.to_dict() for item in occurrences[phrase]],
                }
            )
        return {
            "schema_version": 1,
            "caption_count": caption_count,
            "max_phrase_length": self.max_phrase_length,
            "min_freq": min_freq,
            "phrases": phrases,
        }

