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
│   ├── preflight_aptm.py
│   ├── smoke_test_aptm_gpu.py
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

## Windows：本地开发与 CPU/GPU 验证

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

如果 Windows 有可用 NVIDIA GPU，应在与服务器兼容的 Python 3.9 环境中继续执行
APTM preflight 和少量图片 smoke test，而不是把 import、路径或 adapter 错误留到服务器
调试。GPU smoke test 不属于普通 pytest，不会在缺少 checkpoint 的机器上自动运行：

```powershell
python scripts/preflight_aptm.py --config configs/attributes.yaml
python scripts/smoke_test_aptm_gpu.py `
  --config configs/attributes.yaml `
  --image C:\path\to\person.jpg
```

没有 CUDA 时，smoke script 会明确输出 `SKIP` 并正常退出；缺少 PyTorch、源码、BERT
或 checkpoint 属于环境未准备完成，会给出 `FAIL` 和具体路径。

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
  source_manifest: ./metadata/aptm_source.json
  checkpoint: ./checkpoints/aptm.pth
  bert_path: ./checkpoints/bert-base-uncased
  load_swin_pretrained: false
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

## 2. 准备与验证真实 APTM

建议创建独立环境。PyTorch 按服务器 CUDA 环境单独安装，示例：

```bash
conda create -n tbps-topk python=3.9 -y
conda activate tbps-topk
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -r requirements-aptm.txt
pip install -e .
```

当前服务器兼容组合为 Python 3.9、PyTorch 2.1.2+cu121、torchvision
0.16.2+cu121、NumPy 1.26.4 和 opencv-python 4.11.0.86。NumPy 与 OpenCV
版本被显式固定，是为了避免新版 opencv-python 5.x 强制要求 NumPy 2.x，
从而影响 PyTorch 2.1.2 / APTM 旧代码兼容性。

APTM 官方源码不能直接 clone 到 `third_party/APTM`，因为该目录已经包含本项目管理的
说明文件。请按 [third_party/APTM/README.md](third_party/APTM/README.md) 使用临时目录
clone 固定 commit 后复制，或在 Windows 准备后通过 XFTP 上传。

### 外部资源的准确结论

当前 adapter 的完整 checkpoint inference 调用链决定了：

```text
构建模型必须：
  third_party/APTM/                       固定 commit 的官方源码和 config
  checkpoints/bert-base-uncased/          config.json、vocab.txt、pytorch_model.bin

加载完整 inference 权重必须：
  checkpoints/aptm.pth                    完整 APTM checkpoint

完整 checkpoint inference 不需要：
  checkpoints/swin_base_patch4_window7_224_22k.pth

仅当 load_swin_pretrained: true 时必须：
  checkpoints/swin_base_patch4_window7_224_22k.pth

全量 gallery 才需要：
  datasets/CUHK-PEDES/                    图片与 gallery.json
```

原因是官方 `APTM_Retrieval` 构造期间始终通过
`BertForMaskedLM.from_pretrained(config['text_encoder'])` 构建文本编码器，所以即使稍后
加载完整 APTM checkpoint，BERT 本地目录仍是构建模型的必要输入。视觉侧在完整 checkpoint
模式中明确设置 `load_params=false`，先构造 Swin 结构、再用完整 APTM checkpoint 覆盖
vision encoder 和 projection 参数，因此不读取 Swin 初始化权重。adapter 会检查完整
checkpoint 是否缺少 vision/text encoder、projection 或 temperature 的关键参数。

只有需要按官方初始化流程先载入 Swin pretrained 权重时才把
`load_swin_pretrained` 改为 `true`；此时 adapter 会生成临时 vision config，将官方 JSON
中的 `ckpt` 明确重写为配置的 `swin_path`，再构建模型，之后仍加载配置的完整 APTM
checkpoint。此兼容/诊断模式不是当前 inference 默认路径，也不是新增训练入口。

### Preflight、GPU smoke 与全量提取

先运行 preflight，再使用一到两张普通行人图片执行真实 GPU smoke test：

```bash
python scripts/preflight_aptm.py --config configs/attributes.yaml
python scripts/smoke_test_aptm_gpu.py \
  --config configs/attributes.yaml \
  --image /path/to/person.jpg
```

smoke test 会加载真实 APTM/BERT/checkpoint、编码 54 prompts、输出每张图的 27 个属性，
并验证 model、prompt features 和 image tensor 都实际位于 CUDA device。

smoke test 通过后再执行全量 gallery preprocessing：

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

APTM 官方环境是旧版 PyTorch/CUDA；当前服务器已验证 Python 3.9.25、PyTorch
2.1.2+cu121、torchvision 0.16.2+cu121、NumPy 1.26.4、CUDA 可用和依赖无破损。
正式 APTM checkpoint inference 仍需通过新增的 GPU smoke test 验证。本地机器检测到 NVIDIA
GPU，但当前默认 Python 环境尚未安装 PyTorch，也没有项目外部权重，因此本次不能伪称已经
完成真实模型 smoke test。

## 推荐工作流

```text
Windows 本地开发
  → CPU pytest
  → 有 CUDA 时执行 APTM preflight + GPU smoke test
  → git commit / git push

GitHub
  → 同步源码、配置、ontology 和 metadata

Linux 4090 Server
  → git pull
  → 通过 XFTP 准备 dataset/checkpoint/BERT/可选 Swin
  → python scripts/preflight_aptm.py
  → python scripts/smoke_test_aptm_gpu.py --image ...
  → python scripts/extract_gallery_attributes.py
  → 后续只读取 gallery cache 计算 C(A)、S(a_i) 和 Dynamic Top-K
```

## 版本控制状态

本地仓库已初始化，`origin` 配置为：

```text
https://github.com/w-harda/TBPS-topk.git
```

本实现不会自动 commit 或 push；请先检查测试和 `git status` 后再决定提交。
