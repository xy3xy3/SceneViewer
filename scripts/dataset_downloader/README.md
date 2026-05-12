# 数据集下载与预处理说明

这个目录是一个用 `uv` 管理的小工具，用来从 Hugging Face 抽样下载场景数据，并把数据整理成后续可预处理的本地目录结构。

当前工具主要负责两件事：

- 建立远端数据索引
- 按固定随机种子抽样下载并解压数据
- 生成下载清单，供后续预处理使用
- 将原始场景转换成 SceneViewer 更容易消费的预处理结果

## 支持的数据集

- `sage` 对应 `nvidia/SAGE-10k`
- `scenesmith` 对应 `nepfaff/scenesmith-example-scenes`

其中 `SceneSmith` 默认只处理 `Room` 和 `House` 两个子集，避免默认拉取 `NotGenerated` 这类需要额外留意上游许可的数据。

## 输出目录

默认情况下，数据会写到仓库根目录下的 `assets/`，不是固定写到某台机器上的绝对路径。

默认目录结构如下：

```text
assets/
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
└── preprocessed/
    ├── sage/
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

## 固定 Seed 以便复现

下载抽样时建议显式指定 `--seed`。这样不同人只要使用同样的参数，就能拿到一致的抽样结果，方便复现问题和对齐测试数据。

下面示例统一使用 `--seed 7`。

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

## 一次性预处理全部数据

如果两个数据集都已经下载好了，也可以统一执行：

```bash
uv run dataset-downloader preprocess all
```

这会同时刷新：

- `assets/preprocessed/sage/`
- `assets/preprocessed/scenesmith/`
- `assets/preprocessed/datasets.json`

## 补充说明

- `assets/preprocessed/*` 属于本地生成产物，应该按需重新生成，不建议手工维护。
- 下载阶段会生成 manifest，里面会记录数据集、子集、抽样 seed、远端文件路径和本地解压位置，便于后续复现与排查。
