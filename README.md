# SceneViewer

3D 场景查看器，包含数据集下载预处理脚本和 Web 前端渲染界面。

当前支持：

- `HSM`
- `SAGE`
- `SceneSmith`
- `3D-FRONT`

## 前置要求

开始之前，请确保已安装以下工具：

| 工具 | 用途 | 安装方式 |
|------|------|----------|
| [uv](https://docs.astral.sh/uv/) | Python 包管理与运行 | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [Node.js](https://nodejs.org/) | JavaScript 运行时 | 推荐通过 nvm 或包管理器安装 |
| [pnpm](https://pnpm.io/) | 前端包管理 | `npm i -g pnpm` |

## 快速开始

### 1. 下载与预处理数据集（可选）

如果你已有预处理好的数据，可跳过此步。

```bash
cd scripts/dataset_downloader
uv sync
```

以 SAGE 数据集为例：

```bash
# 建立远端索引
uv run dataset-downloader index sage

# 预览抽样（不下载）
uv run dataset-downloader download sage --sample-size 15 --seed 7 --dry-run

# 真实下载并解压
uv run dataset-downloader download sage --sample-size 15 --seed 7

# 预处理数据
uv run dataset-downloader preprocess sage

# 生成 renderable 数据
uv run dataset-downloader renderable sage
```

如果你已经知道要拉取的 SAGE 场景 id，也可以直接按 id 下载对应的
`scenes/<id>.zip`，不必先走随机抽样：

```bash
uv run dataset-downloader import-sage-remote \
  20251228_133527_layout_6b049b06 \
  --build-preview
```

说明：

- 这个命令会从 `nvidia/SAGE-10k` 下载 `scenes/20251228_133527_layout_6b049b06.zip`。
- 解压后场景会落到 `assets/sage/source/extracted/20251228_133527_layout_6b049b06/`。
- `--build-preview` 会顺手刷新 `assets/preprocessed/sage/` 和 `assets/renderable/sage/`。
- 也可以一次传多个 id；如果你手里是文件名，直接传 `*.zip` 也可以。

以 HSM 数据集为例：

```bash
# 建立远端索引
uv run dataset-downloader index hsm

# 抽样下载 HSM generated_scenes，并自动拉取样本中命中的 support-region 标注 glb
uv run dataset-downloader download hsm --sample-size 20 --seed 7

# 预处理 sceneState JSON
uv run dataset-downloader preprocess hsm

# 生成前端可渲染清单
uv run dataset-downloader renderable hsm
```

说明：

- HSM 的 `generated_scenes/*.json` 来自 `3dlg-hcvc/hsm`。
- 前端真正渲染对象几何时，还需要本地准备 `assets/hsm/hssd-models/objects/.../*.glb`。
- 这个目录结构与 HSM 官方仓库对 HSSD 的依赖保持一致。

如果你希望自动下载 HSSD 模型，也可以执行：

```bash
# 基于本地已经下载的 HSM scene json，只下载被这些 scene 引用到的 HSSD objects
uv run dataset-downloader hsm-hssd

# 如果还需要 decomposed parts
uv run dataset-downloader hsm-hssd --include-decomposed

# 如果要整库下载 objects（体量很大）
uv run dataset-downloader hsm-hssd --full-objects
```

注意：

- 运行前需要先在 Hugging Face 接受 `hssd/hssd-models` 与 `hssd/hssd-hab` 的 gated license。
- 然后执行 `hf auth login`。

如果你要接入 `3D-FRONT`，请先手动下载以下三个压缩包到仓库根目录的 `assets/` 下：

- `assets/3D-FRONT.zip`
- `assets/3D-FUTURE-model.zip`
- `assets/3D-FRONT-texture.zip`

然后执行：

```bash
cd scripts/dataset_downloader
uv sync
uv run dataset-downloader preprocess 3dfront --limit 8
uv run dataset-downloader renderable 3dfront --limit 8
```

说明：

- `3D-FRONT.zip` 提供房间布局和场景 JSON。
- `3D-FUTURE-model.zip` 提供家具几何和纹理。
- `3D-FRONT-texture.zip` 提供墙面和地面贴图。
- `--limit` 适合第一次本地联调时只生成一小批房间；去掉它即可跑更多场景。

如果你已经在本机跑出了 `SceneSmith` 的结果目录，也可以直接导入现有输出，不需要重新走 Hugging Face 下载：

```bash
cd scripts/dataset_downloader
uv sync
uv run dataset-downloader import-scenesmith-local \
  /path/to/scenesmith/outputs/2026-05-18/12-41-05 \
  --build-preview
```

也可以直接导入整个 `outputs` 根目录，命令会递归发现所有有效的 `scene_*` 结果：

```bash
uv run dataset-downloader import-scenesmith-local \
  /home/xy3/ht/scenesmith/outputs \
  --build-preview
```

说明：

- 这个命令会把本地 `scene_*` 输出接到 `assets/scenesmith/source/extracted/<subset>/` 下。
- 默认使用软链接，不会重复拷贝大文件；如需真正复制可加 `--mode copy`。
- 重复结果默认覆盖已有 view 导入；如需保留已有结果并跳过重复项，可加 `--no-force`。
- `--build-preview` 会顺手刷新 `assets/preprocessed/scenesmith/` 和 `assets/renderable/scenesmith/`，前端就能直接预览。

详细的数据集操作说明请查看 [scripts/dataset_downloader/README.md](scripts/dataset_downloader/README.md)。

### 2. 启动 Web 前端

```bash
cd web
pnpm install
pnpm dev
```

启动后在浏览器中访问提示的本地地址（默认为 `http://localhost:5173`）。

其他可用命令：

```bash
pnpm build    # 构建生产版本
pnpm preview  # 预览构建结果
pnpm lint     # 代码检查
```

## 项目结构

```
SceneViewer/
├── assets/                          # 数据目录（下载/预处理/渲染产物）
├── scripts/
│   └── dataset_downloader/          # 数据集下载与预处理工具
├── web/                             # 前端应用（React + Three.js + Vite）
├── refpaper/                        # 参考论文
└── refrepo/                         # 参考仓库
```
