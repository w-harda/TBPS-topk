# TBPS Dynamic Top-K：第一阶段属性模块

本仓库当前只实现以下边界内的功能：

```text
文本 Raw Attribute Extraction
→ Canonicalization（APTM/MALS 27 属性空间）
→ APTM gallery 属性预处理
→ Canonical C(A)
→ 属性重要性 S(a_i)
→ Dynamic Top-K
```

未实现 IRRA、AP-Attack、图像/文本扰动、Push-Pull Loss、TTA、迁移攻击或其他后续攻击模块。

## 已实现内容

- 只从配置指定的 train 标注构建名词短语词表，默认 `phrase length <= 3`、`min_freq=40`。
- 每个 raw phrase occurrence 保存原 caption、caption ID 和字符级 `[start, end)` span。
- 独立维护 APTM 官方 27 属性 / 54 prompt ontology，以及人工审核后的 alias map。
- 输出规则生成的 mapping candidates 和未映射高频 phrase，供一次人工审核后冻结映射。
- query 中未出现的属性保持 Unknown（即不输出），不会自动当作 Negative。
- APTM 作为第三方 adapter 隔离；54 个 prompt feature 一次计算并可缓存。
- gallery 每张图保存 27 个预测、每对 prompt logits、softmax 概率、置信度和实际选中 prompt。
- gallery cache 生成后，评分阶段只读缓存，不重新运行 APTM。
- 基于倒排索引计算 `C(A)`、`S(a_i)` 和熵驱动的 Dynamic Top-K。

## 官方 ontology 来源

`ontology/aptm_attributes.yaml` 严格按 APTM 官方仓库
`Shuyu-XJTU/APTM@b20025445471eb4b743164a6faec29a9eccfb293` 的
`trains.py::train_attr` prompt 顺序记录。

官方 pair 的顺序并不是统一的“negative, positive”。例如 hat/backpack 的 positive prompt 在前；gender 和 age 是语义二选一。因此本项目不通过下标猜语义，而是将每条官方 prompt 的 `label_value`、`semantic` 和 `canonical` 一起存入 ontology，并保存实际选中的 prompt 文本。

## 目录

```text
idea-TBPS/
├── configs/attributes.yaml
├── metadata/
├── ontology/
│   ├── alias_map.json
│   └── aptm_attributes.yaml
├── scripts/
│   ├── build_raw_attribute_vocab.py
│   ├── extract_gallery_attributes.py
│   └── score_query_attributes.py
├── src/attributes/
│   ├── aptm_extractor.py
│   ├── canonicalizer.py
│   ├── config.py
│   ├── dynamic_topk.py
│   ├── gallery_index.py
│   ├── ontology.py
│   ├── scorer.py
│   └── text_miner.py
├── tests/
├── third_party/APTM/README.md
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── requirements-aptm.txt
```

运行后才出现、且默认不进 Git 的目录为 `datasets/`、`checkpoints/`、`cache/` 和 `outputs/`。

## Windows：本地开发与 CPU 测试

建议 Python 3.9：

```powershell
cd E:\Git\idea-TBPS
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

NLTK POS tagger 数据需要准备一次：

```powershell
python -m nltk.downloader averaged_perceptron_tagger_eng
```

若服务器不能联网，可在 Windows 下载 NLTK data 后通过 XFTP 传输，并将
`configs/attributes.yaml` 中的 `text_mining.nltk_data_dir` 指向该目录。

CPU 测试不需要 CUHK-PEDES、APTM、checkpoint、BERT 或 GPU。

## 配置路径

所有项目路径都在 `configs/attributes.yaml` 中配置，并相对于项目根目录解析；业务代码没有 Windows 或 Linux 绝对路径。更换机器时只改配置，不改源码。

关键路径：

```yaml
dataset:
  root: ./datasets/CUHK-PEDES
  train_annotations: ./datasets/CUHK-PEDES/cuhk_train.json
  gallery_manifest: ./datasets/CUHK-PEDES/gallery.json

aptm:
  root: ./third_party/APTM
  checkpoint: ./checkpoints/aptm.pth
  bert_path: ./checkpoints/bert-base-uncased
  swin_path: ./checkpoints/swin_base_patch4_window7_224_22k.pth

cache:
  gallery_attributes: ./cache/gallery_attributes.json
```

`gallery.json` 是小型 manifest，可按以下格式准备；图片路径相对 `dataset.root`：

```json
[
  {"image_id": "cam_a_0001", "path": "imgs/cam_a/0001.jpg"},
  {"image_id": "cam_b_0002", "path": "imgs/cam_b/0002.jpg"}
]
```

如果 manifest 包含整个 gallery、因而体积较大，请将其与数据集一起留在 `datasets/`，不要提交。

## 1. 构建 Raw Attribute Vocabulary

输入必须是独立 train 标注文件，或包含 `train` 键的 JSON。若输入是混合 split 的记录列表，应把 `dataset.require_split_field` 设为 `true`，程序会拒绝缺少 split 的记录，避免 test caption 泄漏。

```powershell
python scripts/build_raw_attribute_vocab.py --config configs/attributes.yaml
```

输出：

- `outputs/raw_attribute_vocab.json`：频次、caption、span；
- `outputs/mapping_candidates.json`：自动规则候选，状态为 `pending_human_review`；
- `outputs/unmapped_high_frequency_phrases.json`：无法映射的高频 phrase。

审核候选后，人工更新并冻结 `ontology/alias_map.json`。当前版本不调用外部 LLM API。

## 2. Linux 4090：一次性提取 Gallery Attributes

建议创建独立环境。PyTorch 按服务器 CUDA 环境单独安装，示例：

```bash
conda create -n tbps-topk python=3.9 -y
conda activate tbps-topk
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -r requirements-aptm.txt
pip install -e .
```

然后准备外部文件并核对配置：

```text
third_party/APTM/                         APTM 官方源码
checkpoints/aptm.pth                      APTM 完整 checkpoint
checkpoints/bert-base-uncased/            BERT 本地目录
checkpoints/swin_base_patch4_window7_224_22k.pth  Swin 初始权重（可选；兼容/从基础模型重建时使用）
datasets/CUHK-PEDES/                      CUHK-PEDES 数据
datasets/CUHK-PEDES/cuhk_train.json       train captions
datasets/CUHK-PEDES/gallery.json          gallery manifest
```

执行：

```bash
python scripts/extract_gallery_attributes.py --config configs/attributes.yaml
```

结果写入 `cache/gallery_attributes.json`。该文件和 prompt embedding cache 都不提交 GitHub。

APTM adapter 复用官方 `APTM_Retrieval`、vision encoder、text encoder 和投影层；image/prompt feature 使用 L2 normalization，并计算：

```text
Z = normalized_image_feature @ normalized_prompt_feature.T / model.temp
```

每个属性只在其官方 `(2i, 2i+1)` prompt pair 内比较。

## 3. Query 评分与 Dynamic Top-K

gallery cache 生成后可在 CPU 上运行评分：

```powershell
python scripts/score_query_attributes.py `
  --config configs/attributes.yaml `
  --query "A man wearing a red shirt, black pants and carrying a backpack."
```

Linux shell 中去掉 PowerShell 的反引号即可。结构化输出默认保存为
`outputs/query_attribute_scores.json`。

评分严格使用 canonical attribute matching：

```text
C(A) = gallery 中包含 query 全部已知 canonical attributes 的图片集合
S(a_i) = ln((|C(A - {a_i})| + alpha) / (|C(A)| + alpha))
```

不读取 IRRA、CLIP、victim embedding 或 retrieval rank。

Dynamic Top-K：

```text
w_i = S(a_i) / sum_j S(a_j)
H = -sum_i w_i ln(w_i), 其中 0 ln 0 = 0
N_eff = exp(H)
k = clip(ceil(beta * N_eff), k_min, k_max)
```

一个有效属性时固定 `k=1`；全部分数为零时使用 `k_min`。`alpha`、`beta`、`k_min` 和 `k_max` 均在配置中管理。

## GitHub 与 XFTP 分工

GitHub 只保存可重建实验的内容：源码、配置、ontology、alias mapping、测试、依赖说明、小型 metadata 和必要的小型结果。

XFTP/本地存储负责大文件：原始数据集、图片、APTM/BERT/Swin 权重、checkpoint、gallery attribute cache、embedding cache 和大型实验输出。

`.gitignore` 已忽略：

- `datasets/`、`checkpoints/`、`weights/`、`cache/`、`outputs/`；
- `*.pth`、`*.pt`、`*.ckpt`、`*.bin`、`*.safetensors`；
- APTM 外部源码（仅保留版本说明）；
- Python cache、虚拟环境、测试覆盖文件、IDE 与系统临时文件。

## APTM 兼容性状态

当前没有修改 APTM 官方源码。所有项目逻辑都在 adapter 中：延迟导入官方代码、覆盖本地路径、加载 checkpoint、确定性预处理、缓存 prompt features 和输出统一 JSON。

APTM 官方环境是旧版 PyTorch/CUDA；目标服务器首先尝试 Python 3.9 + PyTorch 2.1.2 + CUDA 12.1 runtime。如果正式推理出现兼容错误，应记录原始报错并只做最小兼容修改，同时在本节记录改动。由于本地没有 checkpoint/GPU，目前尚未完成 4090 实机兼容验证。

## 版本控制状态

本地仓库已初始化，`origin` 配置为：

```text
https://github.com/w-harda/TBPS-topk.git
```

本实现不会自动 commit 或 push；请先检查测试和 `git status` 后再决定提交。
