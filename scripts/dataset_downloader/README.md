# 数据集下载与预处理说明

这个目录是一个用 `uv` 管理的小工具，用来从 Hugging Face 抽样下载场景数据，并把数据整理成后续可预处理的本地目录结构。

当前工具主要负责两件事：

- 建立远端数据索引
- 按固定随机种子抽样下载并解压数据
- 生成下载清单，供后续预处理使用
- 将原始场景转换成 SceneViewer 更容易消费的预处理结果

## 支持的数据集

- `hsm` 对应 `3dlg-hcvc/hsm`
- `sage` 对应 `nvidia/SAGE-10k`
- `scenesmith` 对应 `nepfaff/scenesmith-example-scenes`
- `sceneweaver` 对应本地 `SceneWeaver` 输出目录
- `3dfront` 对应手动下载的 `3D-FRONT / 3D-FUTURE / 3D-FRONT-texture`

其中 `SceneSmith` 默认只处理 `Room` 和 `House` 两个子集，避免默认拉取 `NotGenerated` 这类需要额外留意上游许可的数据。

`3D-FRONT` 不走 Hugging Face 下载流程，需要先手动把以下文件放到仓库根目录 `assets/`：

- `3D-FRONT.zip`
- `3D-FUTURE-model.zip`
- `3D-FRONT-texture.zip`

## 输出目录

默认情况下，数据会写到仓库根目录下的 `assets/`，不是固定写到某台机器上的绝对路径。

默认目录结构如下：

```text
assets/
├── hsm/
│   ├── manifests/
│   └── source/
│       └── raw/
├── sage/
│   ├── manifests/
│   └── source/
│       ├── archives/
│       └── extracted/
├── scenesmith/
│   ├── manifests/
│   └── source/
│       ├── archives/
│       └── extracted/
├── sceneweaver/
│   ├── manifests/
│   └── source/
│       └── extracted/
└── preprocessed/
    ├── sage/
    ├── sceneweaver/
    ├── scenesmith/
    └── datasets.json
```

如果你不想使用默认位置，下载阶段可以通过 `--destination` 指定别的输出目录。

## 环境准备

先进入这个工具目录，再安装依赖：

```bash
cd scripts/dataset_downloader
uv sync
```

后面的命令默认都在 `scripts/dataset_downloader` 目录下执行。

`preprocess` 和 `renderable` 额外支持：

```bash
--limit N
```

用于只处理前 `N` 个场景，方便本地快速联调。

## 固定 Seed 以便复现

下载抽样时建议显式指定 `--seed`。这样不同人只要使用同样的参数，就能拿到一致的抽样结果，方便复现问题和对齐测试数据。

下面示例统一使用 `--seed 7`。

## HSM

### 1. 建立索引

```bash
uv run dataset-downloader index hsm
```

### 2. 下载数据

先预览抽样结果：

```bash
uv run dataset-downloader download hsm --sample-size 20 --seed 7 --dry-run
```

确认后执行真实下载：

```bash
uv run dataset-downloader download hsm --sample-size 20 --seed 7
```

默认会下载：

- `generated_scenes/scene_*.json`
- 这些样本场景中命中的 `support_region_dataset/annot/*.glb`
- 这些样本场景中命中的 `support_region_dataset/annot_surface/*.glb`

下载后的目录默认位于：

- `assets/hsm/source/raw/generated_scenes/`
- `assets/hsm/source/raw/support_region_dataset/`

### 3. 预处理数据

```bash
uv run dataset-downloader preprocess hsm
```

输出目录：

- `assets/preprocessed/hsm/`

### 4. 生成 renderable 数据

```bash
uv run dataset-downloader renderable hsm
```

输出目录：

- `assets/renderable/hsm/`

### 5. 额外依赖

HSM 的场景 JSON 只描述对象摆放，不直接携带完整家具几何。要让前端真正渲染对象，还需要本地准备：

- `assets/hsm/hssd-models/objects/<first-char>/<mesh-id>.glb`

这与 HSM 官方仓库使用的 HSSD 目录结构保持一致。

### 6. 自动下载 HSSD

如果你已经先下载了一批 `generated_scenes/*.json`，可以只按这些 scene 实际引用到的 mesh id 定向下载 HSSD objects：

```bash
uv run dataset-downloader hsm-hssd
```

如果还需要 decomposed part meshes：

```bash
uv run dataset-downloader hsm-hssd --include-decomposed
```

如果你确实要下载整个 HSSD objects 树：

```bash
uv run dataset-downloader hsm-hssd --full-objects
```

常用辅助参数：

```bash
uv run dataset-downloader hsm-hssd --dry-run
uv run dataset-downloader hsm-hssd --max-workers 16
uv run dataset-downloader hsm-hssd --manifest /tmp/hsm-hssd.json
```

注意：

- 这个命令依赖 Hugging Face 的 gated dataset 权限。
- 开始前请先接受 `hssd/hssd-models` 和 `hssd/hssd-hab` 的 license。
- 然后执行 `hf auth login`。

## SAGE

### 1. 建立索引

```bash
uv run dataset-downloader index sage
```

这一步只会读取远端文件列表，并在 `assets/sage/manifests/` 下生成索引清单，不会真正下载数据。

### 2. 下载数据

先预览抽样结果，不实际下载：

```bash
uv run dataset-downloader download sage --sample-size 15 --seed 7 --dry-run
```

确认无误后执行真实下载并解压：

```bash
uv run dataset-downloader download sage --sample-size 15 --seed 7
```

下载后的原始压缩包和解压内容默认分别位于：

- `assets/sage/source/archives/`
- `assets/sage/source/extracted/`

### 3. 预处理数据

```bash
uv run dataset-downloader preprocess sage
```

预处理结果会写到：

- `assets/preprocessed/sage/`

这个步骤会把下载得到的原始场景整理成统一格式，并刷新对应的 `index.json`。

### 4. 生成 renderable 数据

```bash
uv run dataset-downloader renderable sage
```

这一步会基于 `assets/preprocessed/sage/` 生成前端可直接消费的渲染数据，输出到：

- `assets/renderable/sage/`

## SceneSmith

如果你手上已经有本机跑出来的 SceneSmith 输出目录，也可以直接导入，不必重新下载官方样例。

### 1. 建立索引

```bash
uv run dataset-downloader index scenesmith
```

默认只索引 `Room` 和 `House` 两个子集。

### 2. 下载数据

先预览抽样结果，不实际下载：

```bash
uv run dataset-downloader download scenesmith --sample-size 10 --seed 7 --dry-run
```

确认后执行真实下载并解压：

```bash
uv run dataset-downloader download scenesmith --sample-size 10 --seed 7
```

如果你希望第一次只拉取更小的数据，便于本地快速验证，可以限制子集和压缩包大小：

```bash
uv run dataset-downloader download scenesmith \
  --subset Room \
  --sample-size 2 \
  --max-size-gib 1.0 \
  --seed 7
```

如果你想显式包含更多子集，可以重复传入 `--subset`：

```bash
uv run dataset-downloader download scenesmith \
  --subset Room \
  --subset House \
  --subset NoCritic \
  --sample-size 6 \
  --seed 7 \
  --dry-run
```

如果要包含所有子集：

```bash
uv run dataset-downloader download scenesmith --subset all --sample-size 8 --seed 7 --dry-run
```

下载后的原始压缩包和解压内容默认分别位于：

- `assets/scenesmith/source/archives/`
- `assets/scenesmith/source/extracted/`

### 3. 预处理数据

```bash
uv run dataset-downloader preprocess scenesmith
```

预处理结果会写到：

- `assets/preprocessed/scenesmith/`

如果某些场景下载不完整，预处理阶段会尽量跳过问题场景，并把跳过原因记录到结果清单里，而不是直接中断整个流程。

### 4. 生成 renderable 数据

```bash
uv run dataset-downloader renderable scenesmith
```

这一步会基于 `assets/preprocessed/scenesmith/` 生成前端可直接消费的渲染数据，输出到：

- `assets/renderable/scenesmith/`

### 5. 导入本机 SceneSmith 输出

如果你已经在本机 `scenesmith` 仓库里跑出了结果，例如：

```text
/home/xy3/ht/scenesmith/outputs/2026-05-18/12-41-05
```

可以直接导入这份结果并刷新 web 预览所需资产：

```bash
uv run dataset-downloader import-scenesmith-local \
  /home/xy3/ht/scenesmith/outputs/2026-05-18/12-41-05 \
  --build-preview
```

如果你本机别的地方已经下载好了原始 `SceneSmith` 场景包，也可以直接传单个 tar/zip 文件，哪怕文件名里保留了浏览器下载参数：

```bash
uv run dataset-downloader import-scenesmith-local \
  '/home/xy/proj/SceneBenchmark/local/scenesmith-example-data/scene_000.tar?download=true'
```

如果要把 `outputs` 下面所有本地实验一次性导入，可以直接传根目录：

```bash
uv run dataset-downloader import-scenesmith-local \
  /home/xy3/ht/scenesmith/outputs \
  --build-preview
```

这个命令会做三件事：

- 找到该目录下的 `scene_*` 子目录，直接接收单个 `scene_*` 目录，或递归发现 `outputs` 根目录下所有有效结果。
- 如果传入的是本地 tar/zip 场景包，会自动安全解压并校验是否包含 `package.xml` 和 `combined_house/`。
- 把它们接到 `assets/scenesmith/source/extracted/<subset>/` 下；批量导入时 subset 会按实验目录自动推断，例如 `local-2026-05-18-12-41-05`。
- 如果带了 `--build-preview`，会自动执行 `preprocess scenesmith` 和 `renderable scenesmith`。
- 如果 `assets/preprocessed/scenesmith/index.json` 或 `assets/renderable/scenesmith/index.json` 缺失，导入后也会自动补建一次预览资产，方便从误删目录中恢复。

常用参数：

```bash
# 指定 subset 名，避免不同本地实验都落到同一组目录
uv run dataset-downloader import-scenesmith-local /path/to/output --subset local-bedroom-run

# 对整个 outputs 根目录指定同一个 subset 时，目标 scene 会自动加实验名前缀以避免 scene_000 冲突
uv run dataset-downloader import-scenesmith-local /home/xy3/ht/scenesmith/outputs --subset all-local

# 改成真实复制，而不是软链接
uv run dataset-downloader import-scenesmith-local /path/to/output --mode copy

# 覆盖已存在的导入目标；这是默认行为
uv run dataset-downloader import-scenesmith-local /path/to/output --force

# 保留已存在的导入目标，重复项自动跳过
uv run dataset-downloader import-scenesmith-local /path/to/output --no-force

# 保留已存在的导入目标；更直白的别名，适合批量导入 outputs 根目录
uv run dataset-downloader import-scenesmith-local /home/xy3/ht/scenesmith/outputs --skip-existing
```

说明：

- 默认 `--mode link`，适合本机联调，速度快且不重复占磁盘。
- 传入 tar/zip 时会自动解压导入；`--mode` 只对目录输入生效。
- 无效或不完整的 `scene_*` 目录会记录为 skipped，不会中断其它结果导入。
- 默认 subset 会从路径自动推断，例如 `outputs/2026-05-18/12-41-05` 会生成类似 `local-2026-05-18-12-41-05` 的分组。
- 导入记录会写到 `assets/scenesmith/manifests/local_import_<subset>.json`，方便回看来源。

## 一次性预处理全部数据

如果两个数据集都已经下载好了，也可以统一执行：

```bash
uv run dataset-downloader preprocess all
```

这会同时刷新：

- `assets/preprocessed/sage/`
- `assets/preprocessed/scenesmith/`
- `assets/preprocessed/3dfront/`
- `assets/preprocessed/datasets.json`

## 3D-FRONT

### 1. 手动准备数据

把以下压缩包放到仓库根目录的 `assets/` 下：

```text
assets/
├── 3D-FRONT.zip
├── 3D-FUTURE-model.zip
└── 3D-FRONT-texture.zip
```

### 2. 预处理

```bash
uv run dataset-downloader preprocess 3dfront --limit 8
```

输出目录：

- `assets/preprocessed/3dfront/`

### 3. 生成 renderable 数据

```bash
uv run dataset-downloader renderable 3dfront --limit 8
```

输出目录：

- `assets/renderable/3dfront/`

### 4. 说明

- 房间壳层 mesh 直接来自 `3D-FRONT.zip` 中的场景网格。
- 家具 GLB 由 `3D-FUTURE-model.zip` 中的 `raw_model.obj + texture` 转换得到。
- 少量门窗类引用可能在 `3D-FUTURE-model.zip` 中没有对应模型；当前脚本会自动跳过这些对象，并保留房间结构预览。

## 补充说明

- `assets/preprocessed/*` 属于本地生成产物，应该按需重新生成，不建议手工维护。
- 下载阶段会生成 manifest，里面会记录数据集、子集、抽样 seed、远端文件路径和本地解压位置，便于后续复现与排查。
