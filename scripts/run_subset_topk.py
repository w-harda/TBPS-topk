from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from attributes.aptm_extractor import APTMAttributeExtractor, OfficialAPTMBackend
from attributes.canonicalizer import Canonicalizer
from attributes.config import load_project_config, project_path, set_random_seed, write_json_atomic
from attributes.gallery_index import GalleryAttributeIndex
from attributes.ontology import AttributeOntology
from attributes.subset import evaluate_query_subset, load_gallery_manifest, load_query_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用少量真实 CUHK-PEDES gallery/query 验证属性到 Dynamic Top-K 的完整流程"
    )
    parser.add_argument("--num-gallery", type=int, required=True)
    parser.add_argument("--num-queries", type=int, required=True)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "attributes.yaml"))
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_gallery < 1 or args.num_queries < 1:
        raise ValueError("--num-gallery 和 --num-queries 必须 >= 1")

    config, project_root = load_project_config(args.config)
    set_random_seed(int(config.get("seed", 42)))
    subset_config = config["subset"]
    output_dir = project_path(
        project_root,
        args.output_dir if args.output_dir is not None else subset_config["output_dir"],
    )
    gallery_cache = output_dir / "gallery_attributes_subset.json"
    query_output = output_dir / "query_topk.json"
    summary_output = output_dir / "summary.json"
    formal_cache = project_path(project_root, config["cache"]["gallery_attributes"])
    if gallery_cache.resolve() == formal_cache.resolve():
        raise ValueError("subset cache 禁止覆盖正式 gallery cache")

    ontology = AttributeOntology.load(project_path(project_root, config["ontology"]["path"]))
    canonicalizer = Canonicalizer.load(
        ontology, project_path(project_root, config["ontology"]["alias_map"])
    )
    backend = OfficialAPTMBackend.from_config(config["aptm"], project_root, ontology)
    preflight = backend.preflight(
        ontology.prompt_texts,
        require_cuda=None,
        check_runtime=True,
        writable_directories=[output_dir],
    )
    if not preflight.ok:
        raise RuntimeError(preflight.format_text())

    dataset_config = config["dataset"]
    gallery_manifest = project_path(project_root, dataset_config["gallery_manifest"])
    dataset_root = project_path(project_root, dataset_config["root"])
    query_annotations = project_path(project_root, subset_config["query_annotations"])
    images = load_gallery_manifest(
        gallery_manifest,
        dataset_root,
        limit=args.num_gallery,
        split_name=dataset_config.get("gallery_split"),
    )
    queries = load_query_records(
        query_annotations,
        limit=args.num_queries,
        split_name=subset_config.get("query_split", "test"),
    )

    extractor = APTMAttributeExtractor(ontology, backend)
    gallery_payload = extractor.extract_gallery_to_cache(images, gallery_cache)
    gallery_index = GalleryAttributeIndex.from_cache(gallery_cache)
    scoring = config["scoring"]
    topk = config["topk"]
    query_results, summary = evaluate_query_subset(
        queries,
        canonicalizer,
        gallery_index,
        alpha=float(scoring["alpha"]),
        beta=float(topk["beta"]),
        k_min=int(topk["k_min"]),
        k_max=int(topk["k_max"]) if topk.get("k_max") is not None else None,
        low_coverage_max_attributes=int(
            subset_config.get("low_coverage_max_attributes", 1)
        ),
    )

    query_payload = {
        "schema_version": 1,
        "warning": "subset_sanity_check_only_not_formal_statistics",
        "gallery_cache": str(gallery_cache),
        "gallery_count": gallery_payload["image_count"],
        "query_count": len(query_results),
        "queries": query_results,
    }
    summary.update(
        {
            "requested_gallery_count": args.num_gallery,
            "actual_gallery_count": gallery_payload["image_count"],
            "requested_query_count": args.num_queries,
            "gallery_manifest": str(gallery_manifest),
            "query_annotations": str(query_annotations),
            "output_files": {
                "gallery_attributes": str(gallery_cache),
                "query_topk": str(query_output),
                "summary": str(summary_output),
            },
        }
    )
    write_json_atomic(query_output, query_payload)
    write_json_atomic(summary_output, summary)
    print(
        json.dumps(
            {
                "status": "PASS",
                "warning": "subset 仅用于 pipeline sanity check，不能替代完整 gallery 统计",
                "gallery_count": gallery_payload["image_count"],
                "query_count": len(query_results),
                "output_dir": str(output_dir),
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
