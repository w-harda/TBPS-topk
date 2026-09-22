import json

import pytest

from attributes.text_miner import CaptionRecord, NounPhraseMiner, load_train_captions


def simple_tagger(tokens: list[str]) -> list[tuple[str, str]]:
    tags = {
        "red": "JJ",
        "shirt": "NN",
        "black": "JJ",
        "pants": "NNS",
        "person": "NN",
        "wears": "VBZ",
    }
    return [(token, tags.get(token.lower(), "IN")) for token in tokens]


def test_miner_keeps_original_character_spans_and_frequency() -> None:
    miner = NounPhraseMiner(max_phrase_length=3, tagger=simple_tagger)
    vocabulary = miner.build_vocabulary(
        [
            CaptionRecord("q1", "A person wears a red shirt."),
            CaptionRecord("q2", "The person wears a red shirt and black pants."),
        ],
        min_freq=2,
    )
    red_shirt = next(item for item in vocabulary["phrases"] if item["phrase"] == "red shirt")
    assert red_shirt["count"] == 2
    assert len(red_shirt["occurrences"]) == 2
    for occurrence in red_shirt["occurrences"]:
        start, end = occurrence["span"]
        assert occurrence["caption"][start:end] == occurrence["raw"]


def test_load_train_captions_filters_non_train_records(tmp_path) -> None:
    annotation = tmp_path / "captions.json"
    annotation.write_text(
        json.dumps(
            [
                {"id": 1, "split": "train", "captions": ["train caption"]},
                {"id": 2, "split": "test", "captions": ["test caption"]},
            ]
        ),
        encoding="utf-8",
    )
    records = load_train_captions(annotation, require_split=True)
    assert [record.text for record in records] == ["train caption"]


def test_require_split_prevents_accidental_test_leakage(tmp_path) -> None:
    annotation = tmp_path / "captions.json"
    annotation.write_text(json.dumps([{"caption": "unknown split"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="split"):
        load_train_captions(annotation, require_split=True)

