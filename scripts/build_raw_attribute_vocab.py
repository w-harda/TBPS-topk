from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from attributes.canonicalizer import Canonicalizer
from attributes.config import load_project_config, project_path, write_json_atomic
from attributes.ontology import AttributeOntology
from attributes.text_miner import NounPhraseMiner, load_train_captions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="仅使用 train split 构建 Raw Attribute Vocabulary")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "attributes.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, project_root = load_project_config(args.config)
    mining = config["text_mining"]
    nltk_data_dir = mining.get("nltk_data_dir")
    if nltk_data_dir:
        import nltk

        nltk.data.path.insert(0, str(project_path(project_root, nltk_data_dir)))
    dataset = config["dataset"]
    captions = load_train_captions(
        project_path(project_root, dataset["train_annotations"]),
        split_name=dataset.get("train_split_name", "train"),
        require_split=bool(dataset.get("require_split_field", False)),
    )
    miner = NounPhraseMiner(max_phrase_length=int(mining["max_phrase_length"]))
    vocabulary = miner.build_vocabulary(captions, min_freq=int(mining["min_freq"]))
    vocabulary["source"] = str(project_path(project_root, dataset["train_annotations"]))
    vocab_path = project_path(project_root, mining["raw_vocab_output"])
    write_json_atomic(vocab_path, vocabulary)

    ontology = AttributeOntology.load(project_path(project_root, config["ontology"]["path"]))
    canonicalizer = Canonicalizer.load(
        ontology, project_path(project_root, config["ontology"]["alias_map"])
    )
    counts = {item["phrase"]: item["count"] for item in vocabulary["phrases"]}
    mapped, unmapped = canonicalizer.generate_candidates(counts)
    write_json_atomic(
        project_path(project_root, mining["mapping_candidates_output"]),
        {
            "schema_version": 1,
            "review_status": "pending_human_review",
            "candidates": [candidate.to_dict() for candidate in mapped],
        },
    )
    write_json_atomic(
        project_path(project_root, mining["unmapped_output"]),
        {"schema_version": 1, "phrases": unmapped},
    )
    print(json.dumps({"captions": len(captions), "phrases": len(counts), "output": str(vocab_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

