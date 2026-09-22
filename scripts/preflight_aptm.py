from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from attributes.aptm_extractor import OfficialAPTMBackend
from attributes.config import load_project_config, project_path
from attributes.ontology import AttributeOntology


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在加载 APTM 模型前检查离线资源、依赖与 CUDA")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "attributes.yaml"))
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="不要求 CUDA；只用于路径/依赖诊断，不代表正式 APTM GPU 验证通过",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, project_root = load_project_config(args.config)
    ontology = AttributeOntology.load(project_path(project_root, config["ontology"]["path"]))
    backend = OfficialAPTMBackend.from_config(config["aptm"], project_root, ontology)
    writable_directories = [
        project_path(project_root, config["cache"]["gallery_attributes"]).parent,
        project_path(project_root, config["output"]["root"]),
    ]
    report = backend.preflight(
        ontology.prompt_texts,
        require_cuda=False if args.allow_cpu else None,
        check_runtime=True,
        writable_directories=writable_directories,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.json else report.format_text())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

