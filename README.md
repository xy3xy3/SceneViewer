# SceneViewer

3D 场景查看器，包含数据集下载预处理脚本和 Web 前端渲染界面。

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
