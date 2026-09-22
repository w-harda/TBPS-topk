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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 APTM 一次性提取 gallery canonical attributes")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "attributes.yaml"))
    return parser.parse_args()


def load_manifest(path: Path, dataset_root: Path) -> list[tuple[str, Path]]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("gallery manifest 必须是列表")
    images = []
    for index, record in enumerate(records):
        relative = Path(record.get("path", record.get("image", "")))
        image_path = relative if relative.is_absolute() else dataset_root / relative
        image_id = str(record.get("image_id", record.get("id", index)))
        if not image_path.is_file():
            raise FileNotFoundError(f"gallery image 不存在: {image_path}")
        images.append((image_id, image_path.resolve()))
    return images


def main() -> None:
    args = parse_args()
    config, project_root = load_project_config(args.config)
    set_random_seed(int(config.get("seed", 42)))
    ontology = AttributeOntology.load(project_path(project_root, config["ontology"]["path"]))
    backend = OfficialAPTMBackend.from_config(config["aptm"], project_root, ontology)
    dataset = config["dataset"]
    images = load_manifest(
        project_path(project_root, dataset["gallery_manifest"]),
        project_path(project_root, dataset["root"]),
    )
    destination = project_path(project_root, config["cache"]["gallery_attributes"])
    extractor = APTMAttributeExtractor(ontology, backend)
    payload = extractor.extract_gallery_to_cache(images, destination)
    print(json.dumps({"images": payload["image_count"], "cache": str(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
