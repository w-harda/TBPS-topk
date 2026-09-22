from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from attributes.aptm_extractor import APTMAttributeExtractor, OfficialAPTMBackend
from attributes.config import load_project_config, project_path, set_random_seed
from attributes.ontology import AttributeOntology
from attributes.subset import load_gallery_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 APTM 一次性提取 gallery canonical attributes")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "attributes.yaml"))
    parser.add_argument("--limit", type=int, default=None, help="只处理 manifest 前 N 张图片")
    parser.add_argument("--output", default=None, help="自定义 cache 输出路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, project_root = load_project_config(args.config)
    set_random_seed(int(config.get("seed", 42)))
    ontology = AttributeOntology.load(project_path(project_root, config["ontology"]["path"]))
    backend = OfficialAPTMBackend.from_config(config["aptm"], project_root, ontology)
    dataset = config["dataset"]
    images = load_gallery_manifest(
        project_path(project_root, dataset["gallery_manifest"]),
        project_path(project_root, dataset["root"]),
        limit=args.limit,
        split_name=dataset.get("gallery_split"),
    )
    formal_destination = project_path(project_root, config["cache"]["gallery_attributes"])
    if args.output:
        destination = project_path(project_root, args.output)
    elif args.limit is not None:
        destination = formal_destination.with_name("gallery_attributes_subset.json")
    else:
        destination = formal_destination
    if args.limit is not None and destination.resolve() == formal_destination.resolve():
        raise ValueError("subset extraction 禁止覆盖正式 gallery cache；请指定其它 --output")
    extractor = APTMAttributeExtractor(ontology, backend)
    payload = extractor.extract_gallery_to_cache(images, destination)
    print(
        json.dumps(
            {
                "images": payload["image_count"],
                "cache": str(destination),
                "subset": args.limit is not None,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
