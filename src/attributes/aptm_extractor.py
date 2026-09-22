from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from .config import project_path, write_json_atomic
from .ontology import AttributeOntology


class PromptImageBackend(Protocol):
    def score(self, image_paths: Sequence[Path], prompt_texts: Sequence[str]) -> list[list[float]]:
        """返回 [image_count, prompt_count] logits。"""


class APTMAttributeExtractor:
    """与官方 APTM 隔离的属性推理适配器。"""

    def __init__(self, ontology: AttributeOntology, backend: PromptImageBackend) -> None:
        self.ontology = ontology
        self.backend = backend

    @staticmethod
    def _pair_probabilities(left: float, right: float) -> tuple[float, float]:
        maximum = max(left, right)
        exp_left = math.exp(left - maximum)
        exp_right = math.exp(right - maximum)
        denominator = exp_left + exp_right
        return exp_left / denominator, exp_right / denominator

    def prediction_from_logits(self, image_id: str, logits: Sequence[float]) -> dict[str, Any]:
        if len(logits) != 54:
            raise ValueError(f"APTM 必须为每张图返回 54 个 prompt logits，实际为 {len(logits)}")
        predictions = []
        for attribute in self.ontology.attributes:
            left_index = 2 * attribute.index
            right_index = left_index + 1
            pair = (float(logits[left_index]), float(logits[right_index]))
            probabilities = self._pair_probabilities(*pair)
            selected_offset = 0 if pair[0] >= pair[1] else 1
            selected_index = left_index + selected_offset
            selected = attribute.prompts[selected_offset]
            predictions.append(
                {
                    "attribute_index": attribute.index,
                    "attribute_id": attribute.id,
                    "prompt_indices": [left_index, right_index],
                    "prompt_logits": list(pair),
                    "prompt_probabilities": list(probabilities),
                    "confidence": probabilities[selected_offset],
                    "selected_prompt_index": selected_index,
                    "selected_prompt": selected.text,
                    "label_value": selected.label_value,
                    "semantic": selected.semantic,
                    "canonical": selected.canonical,
                }
            )
        return {"image_id": str(image_id), "predictions": predictions}

    def extract_gallery(self, images: Sequence[tuple[str, Path]]) -> dict[str, Any]:
        paths = [path for _image_id, path in images]
        score_rows = self.backend.score(paths, self.ontology.prompt_texts)
        if len(score_rows) != len(images):
            raise ValueError("APTM backend 返回的图片数量与输入不一致")
        records = [
            self.prediction_from_logits(image_id, logits)
            for (image_id, _path), logits in zip(images, score_rows)
        ]
        return {
            "schema_version": 1,
            "ontology": self.ontology.metadata.get("name", "MALS_APTM_27"),
            "ontology_digest": self.ontology.digest,
            "image_count": len(records),
            "images": records,
        }

    def extract_gallery_to_cache(
        self, images: Sequence[tuple[str, Path]], destination: str | Path
    ) -> dict[str, Any]:
        payload = self.extract_gallery(images)
        write_json_atomic(destination, payload)
        return payload


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class APTMPreflightReport:
    checks: tuple[PreflightCheck, ...]

    @property
    def ok(self) -> bool:
        return all(check.status != "FAIL" for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": [asdict(check) for check in self.checks]}

    def format_text(self) -> str:
        lines = [f"APTM preflight: {'PASS' if self.ok else 'FAIL'}"]
        lines.extend(f"[{check.status}] {check.name}: {check.detail}" for check in self.checks)
        return "\n".join(lines)


class OfficialAPTMBackend:
    """官方 APTM 的薄封装；所有重依赖均延迟导入。"""

    def __init__(
        self,
        aptm_root: Path,
        source_manifest: Path,
        official_config: Path,
        checkpoint: Path,
        bert_path: Path,
        vision_config: Path,
        swin_path: Path,
        load_swin_pretrained: bool = False,
        offline: bool = True,
        device: str = "cuda",
        batch_size: int = 64,
        image_height: int = 384,
        image_width: int = 128,
        normalization_mean: Sequence[float] = (0.38901278, 0.3651612, 0.34836376),
        normalization_std: Sequence[float] = (0.24344306, 0.23738699, 0.23368555),
        prompt_feature_cache: Path | None = None,
        ontology_digest: str = "",
    ) -> None:
        self.aptm_root = aptm_root
        self.source_manifest = source_manifest
        self.official_config = official_config
        self.checkpoint = checkpoint
        self.bert_path = bert_path
        self.vision_config = vision_config
        self.swin_path = swin_path
        self.load_swin_pretrained = load_swin_pretrained
        self.offline = offline
        self.device_name = device
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.normalization_mean = tuple(normalization_mean)
        self.normalization_std = tuple(normalization_std)
        self.prompt_feature_cache = prompt_feature_cache
        self.ontology_digest = ontology_digest
        self._runtime: tuple[Any, Any, Any, Any] | None = None
        self._prompt_features: Any = None
        self._checkpoint_load_report: dict[str, list[str]] | None = None
        self._last_image_device: str | None = None

    @classmethod
    def from_config(
        cls,
        aptm_config: dict[str, Any],
        project_root: Path,
        ontology: AttributeOntology,
    ) -> "OfficialAPTMBackend":
        normalization = aptm_config["normalization"]
        return cls(
            aptm_root=project_path(project_root, aptm_config["root"]),
            source_manifest=project_path(project_root, aptm_config["source_manifest"]),
            official_config=project_path(project_root, aptm_config["official_config"]),
            checkpoint=project_path(project_root, aptm_config["checkpoint"]),
            bert_path=project_path(project_root, aptm_config["bert_path"]),
            vision_config=project_path(project_root, aptm_config["vision_config"]),
            swin_path=project_path(project_root, aptm_config["swin_path"]),
            load_swin_pretrained=bool(aptm_config.get("load_swin_pretrained", False)),
            offline=bool(aptm_config.get("offline", True)),
            device=str(aptm_config.get("device", "cuda")),
            batch_size=int(aptm_config.get("batch_size", 64)),
            image_height=int(aptm_config.get("image_height", 384)),
            image_width=int(aptm_config.get("image_width", 128)),
            normalization_mean=normalization["mean"],
            normalization_std=normalization["std"],
            prompt_feature_cache=project_path(project_root, aptm_config["prompt_feature_cache"]),
            ontology_digest=ontology.digest,
        )

    @staticmethod
    def _normalized_file_digest(path: Path) -> str:
        content = path.read_bytes().replace(b"\r\n", b"\n")
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _probe_writable_directory(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(prefix=".tbps-write-test-", dir=path)
        os.close(file_descriptor)
        Path(temporary_name).unlink()

    def preflight(
        self,
        prompt_texts: Sequence[str],
        *,
        require_cuda: bool | None = None,
        check_runtime: bool = True,
        writable_directories: Sequence[Path] = (),
    ) -> APTMPreflightReport:
        """在加载大型模型前验证离线资源、版本、CUDA 与可写目录。"""

        checks: list[PreflightCheck] = []

        def record(name: str, status: str, detail: str) -> None:
            checks.append(PreflightCheck(name=name, status=status, detail=detail))

        if self.aptm_root.is_dir():
            record("APTM source root", "PASS", str(self.aptm_root))
        else:
            record("APTM source root", "FAIL", f"Missing directory: {self.aptm_root}")

        manifest_data: dict[str, Any] | None = None
        if self.source_manifest.is_file():
            try:
                manifest_data = json.loads(self.source_manifest.read_text(encoding="utf-8"))
                record(
                    "APTM source manifest",
                    "PASS",
                    f"commit={manifest_data.get('commit', 'unknown')}",
                )
            except (OSError, json.JSONDecodeError) as exc:
                record("APTM source manifest", "FAIL", f"Invalid manifest: {exc}")
        else:
            record("APTM source manifest", "FAIL", f"Missing file: {self.source_manifest}")

        if manifest_data is not None:
            mismatches: list[str] = []
            for relative_name, expected_digest in manifest_data.get("files", {}).items():
                source_file = self.aptm_root / Path(relative_name)
                if not source_file.is_file():
                    mismatches.append(f"missing {relative_name}")
                    continue
                actual_digest = self._normalized_file_digest(source_file)
                if actual_digest != expected_digest:
                    mismatches.append(f"hash mismatch {relative_name}")
            if mismatches:
                record("APTM pinned source", "FAIL", "; ".join(mismatches))
            else:
                record(
                    "APTM pinned source",
                    "PASS",
                    f"verified {len(manifest_data.get('files', {}))} files",
                )

        required_files = {
            "APTM official config": self.official_config,
            "APTM vision config": self.vision_config,
            "APTM checkpoint": self.checkpoint,
        }
        for name, path in required_files.items():
            if path.is_file() and path.stat().st_size > 0:
                record(name, "PASS", str(path))
            else:
                record(name, "FAIL", f"Missing or empty file: {path}")

        if self.bert_path.is_dir():
            missing_bert = [
                name
                for name in ("config.json", "vocab.txt", "pytorch_model.bin")
                if not (self.bert_path / name).is_file()
            ]
            if missing_bert:
                record(
                    "BERT local directory",
                    "FAIL",
                    f"Missing in {self.bert_path}: {', '.join(missing_bert)}",
                )
            else:
                record("BERT local directory", "PASS", str(self.bert_path))
        else:
            record("BERT local directory", "FAIL", f"Missing directory: {self.bert_path}")

        if self.load_swin_pretrained:
            if self.swin_path.is_file() and self.swin_path.stat().st_size > 0:
                record("Swin initialization checkpoint", "PASS", str(self.swin_path))
            else:
                record(
                    "Swin initialization checkpoint",
                    "FAIL",
                    f"load_swin_pretrained=true but file is missing: {self.swin_path}",
                )
        else:
            record(
                "Swin initialization checkpoint",
                "PASS",
                "not required: complete APTM checkpoint mode sets load_params=false",
            )

        if len(prompt_texts) == 54 and all(text.strip() for text in prompt_texts):
            record("APTM ontology prompts", "PASS", "54 non-empty prompts")
        else:
            record("APTM ontology prompts", "FAIL", f"expected 54, got {len(prompt_texts)}")

        try:
            vision_data = json.loads(self.vision_config.read_text(encoding="utf-8"))
            expected_shape = (self.image_height, self.image_width)
            actual_shape = (int(vision_data["h"]), int(vision_data["w"]))
            if expected_shape != actual_shape:
                record(
                    "APTM image shape",
                    "FAIL",
                    f"adapter={expected_shape}, vision_config={actual_shape}",
                )
            else:
                record("APTM image shape", "PASS", f"{self.image_height}x{self.image_width}")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            record("APTM image shape", "FAIL", f"cannot read vision config: {exc}")

        directories = list(writable_directories)
        if self.prompt_feature_cache is not None:
            directories.append(self.prompt_feature_cache.parent)
        for directory in dict.fromkeys(path.resolve() for path in directories):
            try:
                self._probe_writable_directory(directory)
                record("Writable directory", "PASS", str(directory))
            except OSError as exc:
                record("Writable directory", "FAIL", f"{directory}: {exc}")

        if check_runtime:
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            record(
                "Python version",
                "PASS" if python_version == "3.9" else "WARN",
                f"installed={python_version}, formal target=3.9",
            )
            expected_versions = {
                "torch": "2.1.2",
                "torchvision": "0.16.2",
                "numpy": "1.26.4",
                "opencv-python": "4.11.0.86",
                "timm": "0.4.9",
                "transformers": "4.12.5",
            }
            for distribution, expected in expected_versions.items():
                try:
                    actual = importlib.metadata.version(distribution)
                    matches = actual == expected or (
                        distribution in {"torch", "torchvision"}
                        and actual.startswith(expected + "+")
                    )
                    record(
                        f"Python package {distribution}",
                        "PASS" if matches else "FAIL",
                        f"installed={actual}, expected={expected}",
                    )
                except importlib.metadata.PackageNotFoundError:
                    record(f"Python package {distribution}", "FAIL", "not installed")

            cuda_required = self.device_name.startswith("cuda") if require_cuda is None else require_cuda
            try:
                import torch

                cuda_available = torch.cuda.is_available()
                if cuda_required and not cuda_available:
                    record("CUDA", "FAIL", "torch.cuda.is_available() is False")
                elif cuda_available:
                    device_index = torch.device(self.device_name).index or 0
                    record("CUDA", "PASS", torch.cuda.get_device_name(device_index))
                else:
                    record("CUDA", "PASS", "not required")
            except ImportError:
                record("CUDA", "FAIL" if cuda_required else "PASS", "PyTorch not installed")

        return APTMPreflightReport(tuple(checks))

    def _raise_for_preflight(self, prompt_texts: Sequence[str]) -> None:
        report = self.preflight(prompt_texts, check_runtime=True)
        if not report.ok:
            raise RuntimeError(report.format_text())

    def _load_complete_checkpoint(self, torch: Any, model: Any) -> None:
        checkpoint = torch.load(self.checkpoint, map_location="cpu")
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        if not isinstance(state_dict, dict):
            raise RuntimeError(f"APTM checkpoint 不包含有效 state_dict: {self.checkpoint}")
        incompatible = model.load_state_dict(state_dict, strict=False)
        critical_prefixes = (
            "vision_encoder.",
            "text_encoder.",
            "vision_proj.",
            "text_proj.",
        )
        critical_parameter_names = {
            name
            for name, _parameter in model.named_parameters()
            if name.startswith(critical_prefixes) or name == "temp"
        }
        missing_critical = sorted(critical_parameter_names.intersection(incompatible.missing_keys))
        if missing_critical:
            preview = ", ".join(missing_critical[:10])
            raise RuntimeError(
                "APTM checkpoint 缺少 inference 关键参数；这不是完整 checkpoint。"
                f" Missing: {preview}"
            )
        self._checkpoint_load_report = {
            "missing_keys": sorted(incompatible.missing_keys),
            "unexpected_keys": sorted(incompatible.unexpected_keys),
        }

    def _write_runtime_vision_config(self, destination: Path) -> Path:
        """复制官方 vision config，并把 Swin 初始化权重改为配置路径。"""

        vision_data = json.loads(self.vision_config.read_text(encoding="utf-8"))
        vision_data["ckpt"] = str(self.swin_path)
        destination.write_text(
            json.dumps(vision_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return destination

    def _prepare_runtime(
        self, prompt_texts: Sequence[str] | None = None
    ) -> tuple[Any, Any, Any, Any]:
        if self._runtime is not None:
            return self._runtime
        checked_prompts = prompt_texts or [f"prompt-{index}" for index in range(54)]
        self._raise_for_preflight(checked_prompts)
        if self.offline:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        try:
            import torch
            import torch.nn.functional as functional
            import yaml
            from PIL import Image
            from torchvision import transforms
            from torchvision.transforms import InterpolationMode
        except ImportError as exc:
            raise RuntimeError("APTM 推理依赖未安装，请按 README 安装 GPU 环境") from exc

        aptm_root_string = str(self.aptm_root)
        if aptm_root_string not in sys.path:
            sys.path.insert(0, aptm_root_string)
        loaded_models = sys.modules.get("models")
        if loaded_models is not None:
            loaded_path = Path(getattr(loaded_models, "__file__", "")).resolve()
            if self.aptm_root.resolve() not in loaded_path.parents:
                raise RuntimeError(f"Python 中已加载其它 models 包，和 APTM 冲突: {loaded_path}")
        try:
            from models.model_retrieval import APTM_Retrieval
            from models.tokenization_bert import BertTokenizer
        except ImportError as exc:
            raise RuntimeError(f"无法从 {self.aptm_root} 导入官方 APTM") from exc

        with self.official_config.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config["vision_config"] = str(self.vision_config)
        config["text_config"] = str((self.aptm_root / "configs" / "config_bert.json").resolve())
        config["text_encoder"] = str(self.bert_path)
        config["h"] = self.image_height
        config["w"] = self.image_width
        # 官方 APTM_Retrieval 将 load_params 传给 build_vision_encoder。
        # 完整 checkpoint inference 时为 False；仅显式初始化模式才读取 swin_path。
        config["load_params"] = self.load_swin_pretrained
        # 此字段只由官方 Retrieval.py 主程序消费；adapter 在构建模型后主动加载 checkpoint。
        config["load_pretrained"] = False

        tokenizer = BertTokenizer.from_pretrained(
            str(self.bert_path), local_files_only=self.offline
        )
        temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        try:
            if self.load_swin_pretrained:
                temporary_directory = tempfile.TemporaryDirectory(prefix="tbps-aptm-")
                runtime_vision_config = self._write_runtime_vision_config(
                    Path(temporary_directory.name) / "config_swinB_384.json"
                )
                config["vision_config"] = str(runtime_vision_config)
            model = APTM_Retrieval(config=config)
        finally:
            if temporary_directory is not None:
                temporary_directory.cleanup()

        self._load_complete_checkpoint(torch, model)
        device = torch.device(self.device_name)
        model = model.to(device)
        model.eval()
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.image_height, self.image_width), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(self.normalization_mean, self.normalization_std),
            ]
        )
        self._runtime = (torch, functional, Image, (model, tokenizer, transform, config, device))
        return self._runtime

    def _cache_fingerprint(self, prompt_texts: Sequence[str]) -> str:
        checkpoint_stat = self.checkpoint.stat()
        payload = {
            "ontology_digest": self.ontology_digest,
            "prompts": list(prompt_texts),
            "checkpoint": str(self.checkpoint.resolve()),
            "checkpoint_size": checkpoint_stat.st_size,
            "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _get_prompt_features(self, prompt_texts: Sequence[str]) -> Any:
        torch, functional, _image_module, runtime = self._prepare_runtime(prompt_texts)
        model, tokenizer, _transform, config, device = runtime
        fingerprint = self._cache_fingerprint(prompt_texts)
        if self._prompt_features is not None:
            return self._prompt_features
        if self.prompt_feature_cache and self.prompt_feature_cache.exists():
            cached = torch.load(self.prompt_feature_cache, map_location="cpu")
            cached_features = cached.get("features") if isinstance(cached, dict) else None
            if (
                isinstance(cached, dict)
                and cached.get("fingerprint") == fingerprint
                and getattr(cached_features, "ndim", None) == 2
                and cached_features.shape[0] == len(prompt_texts)
            ):
                self._prompt_features = cached["features"].to(device)
                return self._prompt_features
        with torch.inference_mode():
            tokens = tokenizer(
                list(prompt_texts),
                padding="longest",
                truncation=True,
                max_length=config["max_tokens"],
                return_tensors="pt",
            ).to(device)
            text_embeds = model.get_text_embeds(tokens.input_ids, tokens.attention_mask)
            features = functional.normalize(model.get_features(text_embeds=text_embeds), dim=-1)
        if features.ndim != 2 or features.shape[0] != len(prompt_texts):
            raise RuntimeError(
                "APTM prompt feature shape 异常："
                f"expected ({len(prompt_texts)}, D), got {tuple(features.shape)}"
            )
        self._prompt_features = features
        if self.prompt_feature_cache:
            self.prompt_feature_cache.parent.mkdir(parents=True, exist_ok=True)
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self.prompt_feature_cache.name}.",
                suffix=".tmp",
                dir=self.prompt_feature_cache.parent,
            )
            os.close(file_descriptor)
            try:
                torch.save(
                    {
                        "schema_version": 1,
                        "fingerprint": fingerprint,
                        "features": features.detach().cpu(),
                    },
                    temporary_name,
                )
                Path(temporary_name).replace(self.prompt_feature_cache)
            except Exception:
                Path(temporary_name).unlink(missing_ok=True)
                raise
        return features

    def score(self, image_paths: Sequence[Path], prompt_texts: Sequence[str]) -> list[list[float]]:
        if len(prompt_texts) != 54:
            raise ValueError(f"APTM backend 需要 54 个 prompts，实际为 {len(prompt_texts)}")
        torch, functional, image_module, runtime = self._prepare_runtime(prompt_texts)
        model, _tokenizer, transform, _config, device = runtime
        prompt_features = self._get_prompt_features(prompt_texts)
        temperature = model.temp.detach()
        temperature_value = float(temperature.cpu().item())
        if not math.isfinite(temperature_value) or temperature_value <= 0.0:
            raise RuntimeError(f"APTM checkpoint temperature 必须是有限正数，实际为 {temperature_value}")
        rows: list[list[float]] = []
        with torch.inference_mode():
            for start in range(0, len(image_paths), self.batch_size):
                batch_paths = image_paths[start : start + self.batch_size]
                tensors = []
                for path in batch_paths:
                    with image_module.open(path) as image:
                        tensors.append(transform(image.convert("RGB")))
                images = torch.stack(tensors).to(device)
                self._last_image_device = str(images.device)
                image_embeds, _ = model.get_vision_embeds(images)
                image_features = functional.normalize(
                    model.get_features(image_embeds=image_embeds), dim=-1
                )
                logits = image_features @ prompt_features.t() / temperature
                rows.extend(logits.detach().cpu().tolist())
        return rows

    def diagnostics(self) -> dict[str, Any]:
        if self._runtime is None:
            raise RuntimeError("APTM runtime 尚未加载；请先执行 score()")
        _torch, _functional, _image_module, runtime = self._runtime
        model, _tokenizer, _transform, _config, _device = runtime
        model_device = str(next(model.parameters()).device)
        prompt_device = str(self._prompt_features.device) if self._prompt_features is not None else None
        return {
            "configured_device": self.device_name,
            "model_device": model_device,
            "prompt_features_device": prompt_device,
            "last_image_batch_device": self._last_image_device,
            "load_swin_pretrained": self.load_swin_pretrained,
            "checkpoint_load": self._checkpoint_load_report,
        }
