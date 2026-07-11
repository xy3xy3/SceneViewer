#!/usr/bin/env bash
# 便携版 critic_probe 导入脚本
# 用法:
#   bash scripts/dataset_downloader/import_critic_probe.sh \
#     --critic-on  /path/to/critic_on \
#     --critic-off /path/to/critic_off \
#     [--mode copy|link] [--skip-existing] [--yes]
#
# --critic-on 和 --critic-off 至少指定一个，可分别指向不同目录
# --mode: link（默认，软链接节省空间）, copy（拷贝，源删除后仍可用）

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
DATASET_TOOL_DIR="$SCRIPT_DIR"

CRITIC_ON_DIR=""
CRITIC_OFF_DIR=""
MODE="link"
SKIP_EXISTING=0
ASSUME_YES=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/dataset_downloader/import_critic_probe.sh [options]

Options:
  --critic-on DIR    Path to critic_on batch directory (e.g. .../critic_on)
  --critic-off DIR   Path to critic_off batch directory (e.g. .../critic_off)
  --mode MODE        Import mode: link or copy. Default: link.
  --skip-existing    Skip scene dirs that already exist.
  --yes              Skip confirmation prompt.
  -h, --help

至少指定 --critic-on 和/或 --critic-off。两个可指向不同运行目录。
EOF
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing directory: $1" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --critic-on)  CRITIC_ON_DIR=$2;  shift 2 ;;
    --critic-off) CRITIC_OFF_DIR=$2; shift 2 ;;
    --mode)       MODE=$2;           shift 2 ;;
    --skip-existing) SKIP_EXISTING=1; shift ;;
    --yes)        ASSUME_YES=1;      shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "Unknown: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$CRITIC_ON_DIR" && -z "$CRITIC_OFF_DIR" ]]; then
  echo "必须指定 --critic-on 和/或 --critic-off" >&2
  usage >&2
  exit 1
fi
if [[ "$MODE" != "link" && "$MODE" != "copy" ]]; then
  echo "Unsupported --mode: $MODE (expected link or copy)" >&2
  exit 1
fi

# 检查源目录
HAS_WORK=0
if [[ -n "$CRITIC_ON_DIR" ]]; then
  require_dir "$CRITIC_ON_DIR"
  HAS_WORK=1
fi
if [[ -n "$CRITIC_OFF_DIR" ]]; then
  require_dir "$CRITIC_OFF_DIR"
  HAS_WORK=1
fi
if (( HAS_WORK == 0 )); then
  exit 1
fi

if (( ASSUME_YES == 0 )); then
  echo "将导入以下数据并重建 SceneSmith 预览:"
  [[ -n "$CRITIC_ON_DIR" ]]  && echo "  critic-on  <- $CRITIC_ON_DIR"
  [[ -n "$CRITIC_OFF_DIR" ]] && echo "  critic-off <- $CRITIC_OFF_DIR"
  read -r -p "确认导入？(yes/no): " reply
  if [[ "$reply" != "yes" ]]; then echo "Aborted."; exit 1; fi
fi

# 同步依赖
(
  cd "$DATASET_TOOL_DIR"
  uv sync
)

# 导入
import_group() {
  local label=$1 src=$2 subset=$3
  local -a targets=()
  mapfile -t targets < <(find "$src" -mindepth 1 -maxdepth 1 -type d -name 'batch_*' | sort)
  if [[ ${#targets[@]} -eq 0 ]]; then
    targets=("$src")
  fi
  local -a extra=()
  (( SKIP_EXISTING == 1 )) && extra+=(--skip-existing)
  echo "导入 $label ($subset, ${#targets[@]} batch)"
  for t in "${targets[@]}"; do
    echo "  - $(basename "$t")"
    (
      cd "$DATASET_TOOL_DIR"
      uv run dataset-downloader import-scenesmith-local \
        "$t" --subset "$subset" --mode "$MODE" "${extra[@]}"
    )
  done
}

[[ -n "$CRITIC_ON_DIR" ]]  && import_group "critic_on"  "$CRITIC_ON_DIR"  "critic-on"
[[ -n "$CRITIC_OFF_DIR" ]] && import_group "critic_off" "$CRITIC_OFF_DIR" "critic-off"

# 重建预览
echo "重建 SceneSmith preview 数据..."
(
  cd "$DATASET_TOOL_DIR"
  uv run dataset-downloader preprocess scenesmith
  uv run dataset-downloader renderable scenesmith
)

echo "完成！"
[[ -n "$CRITIC_ON_DIR" ]]  && echo "  critic-on  -> $CRITIC_ON_DIR"
[[ -n "$CRITIC_OFF_DIR" ]] && echo "  critic-off -> $CRITIC_OFF_DIR"
echo "Web 场景将出现在 SceneSmith 数据集下。"
