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
    parser = argparse.ArgumentParser(description="使用真实 checkpoint 和少量图片执行 APTM CUDA smoke test")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "attributes.yaml"))
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="测试图片路径；可重复指定。未指定时读取 config 的 smoke_test.images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import torch
    except ImportError:
        print("FAIL: PyTorch 未安装，无法判断 CUDA 或执行 APTM GPU smoke test。")
        return 1
    if not torch.cuda.is_available():
        print("SKIP: 当前 Python 环境 torch.cuda.is_available() 为 False。")
        return 0

    config, project_root = load_project_config(args.config)
    if not str(config["aptm"].get("device", "cuda")).startswith("cuda"):
        print("FAIL: GPU smoke test 要求 aptm.device 配置为 cuda 或 cuda:N。")
        return 1
    set_random_seed(int(config.get("seed", 42)))
    ontology = AttributeOntology.load(project_path(project_root, config["ontology"]["path"]))
    backend = OfficialAPTMBackend.from_config(config["aptm"], project_root, ontology)
    report = backend.preflight(ontology.prompt_texts, require_cuda=True, check_runtime=True)
    if not report.ok:
        print(report.format_text())
        return 1

    configured_images = args.image or config.get("smoke_test", {}).get("images", [])
    maximum = int(config.get("smoke_test", {}).get("max_images", 2))
    images: list[tuple[str, Path]] = []
    for index, value in enumerate(configured_images[:maximum]):
        raw_path = Path(value).expanduser()
        image_path = raw_path.resolve() if raw_path.is_absolute() else project_path(project_root, raw_path)
        if not image_path.is_file():
            print(f"FAIL: smoke test 图片不存在: {image_path}")
            return 1
        images.append((image_path.stem or f"smoke-{index}", image_path))
    if not images:
        print("FAIL: 未提供 smoke test 图片。请使用 --image PATH 或配置 smoke_test.images。")
        return 1

    extractor = APTMAttributeExtractor(ontology, backend)
    payload = extractor.extract_gallery(images)
    diagnostics = backend.diagnostics()
    device_values = (
        diagnostics["model_device"],
        diagnostics["prompt_features_device"],
        diagnostics["last_image_batch_device"],
    )
    if not all(isinstance(value, str) and value.startswith("cuda") for value in device_values):
        print(json.dumps({"status": "FAIL", "diagnostics": diagnostics}, ensure_ascii=False, indent=2))
        return 1
    if any(len(image["predictions"]) != 27 for image in payload["images"]):
        print("FAIL: APTM smoke test 未对每张图片生成 27 个属性预测。")
        return 1

    print(
        json.dumps(
            {
                "status": "PASS",
                "cuda_device": torch.cuda.get_device_name(torch.device(config["aptm"]["device"]).index or 0),
                "diagnostics": diagnostics,
                "images": payload["images"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
