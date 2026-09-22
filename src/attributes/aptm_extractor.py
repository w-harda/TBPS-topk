from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from .config import write_json_atomic
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


class OfficialAPTMBackend:
    """官方 APTM 的薄封装；所有重依赖均延迟导入。"""

    def __init__(
        self,
        aptm_root: Path,
        official_config: Path,
        checkpoint: Path,
        bert_path: Path,
        vision_config: Path,
        swin_path: Path,
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
        self.official_config = official_config
        self.checkpoint = checkpoint
        self.bert_path = bert_path
        self.vision_config = vision_config
        self.swin_path = swin_path
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

    def _validate_files(self) -> None:
        required = {
            "APTM root": self.aptm_root,
            "APTM config": self.official_config,
            "APTM checkpoint": self.checkpoint,
            "BERT directory": self.bert_path,
            "Swin config": self.vision_config,
        }
        missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError("缺少 APTM 外部文件：\n" + "\n".join(missing))

    def _prepare_runtime(self) -> tuple[Any, Any, Any, Any]:
        if self._runtime is not None:
            return self._runtime
        self._validate_files()
        try:
            import torch
            import torch.nn.functional as functional
            import yaml
            from PIL import Image
            from torchvision import transforms
            from torchvision.transforms import InterpolationMode
        except ImportError as exc:
            raise RuntimeError("APTM 推理依赖未安装，请按 README 安装 GPU 环境") from exc
        if self.device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("配置要求 CUDA，但当前环境不可用；正式 APTM 提取请在 4090 服务器运行")

        aptm_root_string = str(self.aptm_root)
        if aptm_root_string not in sys.path:
            sys.path.insert(0, aptm_root_string)
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
        # 完整 APTM checkpoint 随后加载，无需先加载 Swin 初始化权重。
        config["load_params"] = False
        config["load_pretrained"] = False

        tokenizer = BertTokenizer.from_pretrained(str(self.bert_path))
        model = APTM_Retrieval(config=config)
        model.load_pretrained(str(self.checkpoint), config, is_eval=True)
        device = torch.device(self.device_name)
        model = model.to(device).eval()
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
        torch, functional, _image_module, runtime = self._prepare_runtime()
        model, tokenizer, _transform, config, device = runtime
        fingerprint = self._cache_fingerprint(prompt_texts)
        if self._prompt_features is not None:
            return self._prompt_features
        if self.prompt_feature_cache and self.prompt_feature_cache.exists():
            cached = torch.load(self.prompt_feature_cache, map_location="cpu")
            if cached.get("fingerprint") == fingerprint:
                self._prompt_features = cached["features"].to(device)
                return self._prompt_features
        with torch.no_grad():
            tokens = tokenizer(
                list(prompt_texts),
                padding="longest",
                truncation=True,
                max_length=config["max_tokens"],
                return_tensors="pt",
            ).to(device)
            text_embeds = model.get_text_embeds(tokens.input_ids, tokens.attention_mask)
            features = functional.normalize(model.get_features(text_embeds=text_embeds), dim=-1)
        self._prompt_features = features
        if self.prompt_feature_cache:
            self.prompt_feature_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"fingerprint": fingerprint, "features": features.detach().cpu()},
                self.prompt_feature_cache,
            )
        return features

    def score(self, image_paths: Sequence[Path], prompt_texts: Sequence[str]) -> list[list[float]]:
        torch, functional, image_module, runtime = self._prepare_runtime()
        model, _tokenizer, transform, _config, device = runtime
        prompt_features = self._get_prompt_features(prompt_texts)
        temperature = model.temp.detach().clamp(min=1e-6)
        rows: list[list[float]] = []
        with torch.no_grad():
            for start in range(0, len(image_paths), self.batch_size):
                batch_paths = image_paths[start : start + self.batch_size]
                tensors = []
                for path in batch_paths:
                    with image_module.open(path) as image:
                        tensors.append(transform(image.convert("RGB")))
                images = torch.stack(tensors).to(device)
                image_embeds, _ = model.get_vision_embeds(images)
                image_features = functional.normalize(
                    model.get_features(image_embeds=image_embeds), dim=-1
                )
                logits = image_features @ prompt_features.t() / temperature
                rows.extend(logits.detach().cpu().tolist())
        return rows
