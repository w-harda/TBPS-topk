from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from attributes.canonicalizer import Canonicalizer
from attributes.config import load_project_config, project_path, write_json_atomic
from attributes.dynamic_topk import select_dynamic_topk
from attributes.gallery_index import GalleryAttributeIndex
from attributes.ontology import AttributeOntology
from attributes.scorer import score_attributes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取 query 属性并计算 C(A)、S(a_i) 与 Dynamic Top-K")
    parser.add_argument("--query", required=True)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "attributes.yaml"))
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, project_root = load_project_config(args.config)
    ontology = AttributeOntology.load(project_path(project_root, config["ontology"]["path"]))
    canonicalizer = Canonicalizer.load(
        ontology, project_path(project_root, config["ontology"]["alias_map"])
    )
    extracted = canonicalizer.extract(args.query)
    canonical = [attribute.canonical for attribute in extracted]
    gallery = GalleryAttributeIndex.from_cache(
        project_path(project_root, config["cache"]["gallery_attributes"])
    )
    scores = score_attributes(canonical, gallery, alpha=float(config["scoring"]["alpha"]))
    topk_config = config["topk"]
    result = select_dynamic_topk(
        scores,
        beta=float(topk_config["beta"]),
        k_min=int(topk_config["k_min"]),
        k_max=int(topk_config["k_max"]) if topk_config.get("k_max") is not None else None,
    )
    payload = {
        "schema_version": 1,
        "query": args.query,
        "attributes": [attribute.to_dict() for attribute in extracted],
        "candidate_count": len(gallery.candidates(canonical)),
        "scores": [score.to_dict() for score in scores],
        "dynamic_topk": result.to_dict(),
        "selected": [score.canonical for score in result.selected],
    }
    destination = (
        Path(args.output).resolve()
        if args.output
        else project_path(project_root, config["output"]["query_scores"])
    )
    write_json_atomic(destination, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

