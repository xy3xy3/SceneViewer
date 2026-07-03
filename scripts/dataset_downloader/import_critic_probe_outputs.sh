#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
DATASET_TOOL_DIR="$REPO_ROOT/scripts/dataset_downloader"

DEFAULT_ROOT="/data/task3_2/L202500266_hrk/code/scenesmith/outputs/critic_probe/critic_probe_4rooms_2026-07-02_16-00-25"

SOURCE_ROOT="$DEFAULT_ROOT"
CRITIC_ON_DIR=""
CRITIC_OFF_DIR=""
MODE="link"
SKIP_EXISTING=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/dataset_downloader/import_critic_probe_outputs.sh [options]

Description:
  Import critic_probe SceneSmith outputs into SceneViewer/assets and rebuild
  SceneSmith preprocessed/renderable data so the web app can render them.

Options:
  --root DIR          Root directory containing critic_on/ and critic_off/.
                      Default:
                      /data/task3_2/L202500266_hrk/code/scenesmith/outputs/critic_probe/critic_probe_4rooms_2026-07-02_16-00-25
  --critic-on DIR     Override the critic_on source directory.
  --critic-off DIR    Override the critic_off source directory.
  --mode MODE         Import mode: link or copy. Default: link.
  --skip-existing     Keep already-imported targets and skip duplicates.
  -h, --help

Examples:
  bash scripts/dataset_downloader/import_critic_probe_outputs.sh
  bash scripts/dataset_downloader/import_critic_probe_outputs.sh --root /path/to/run
  bash scripts/dataset_downloader/import_critic_probe_outputs.sh \
    --critic-on /path/to/critic_on \
    --critic-off /path/to/critic_off \
    --mode copy
EOF
}

require_dir() {
  local path=$1
  if [[ ! -d "$path" ]]; then
    echo "Missing directory: $path" >&2
    exit 1
  fi
}

import_group() {
  local label=$1
  local group_dir=$2
  local subset_name=$3
  local -a import_targets=()
  local -a extra_args=()

  require_dir "$group_dir"

  mapfile -t import_targets < <(find "$group_dir" -mindepth 1 -maxdepth 1 -type d -name 'batch_*' | sort)
  if [[ ${#import_targets[@]} -eq 0 ]]; then
    import_targets=("$group_dir")
  fi

  if (( SKIP_EXISTING == 1 )); then
    extra_args+=(--skip-existing)
  fi

  echo "Importing $label into subset '$subset_name' (${#import_targets[@]} target(s))"
  for import_target in "${import_targets[@]}"; do
    echo "  - $(basename "$import_target")"
    (
      cd "$DATASET_TOOL_DIR"
      uv run dataset-downloader import-scenesmith-local \
        "$import_target" \
        --subset "$subset_name" \
        --mode "$MODE" \
        "${extra_args[@]}"
    )
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      SOURCE_ROOT=$2
      shift 2
      ;;
    --critic-on)
      CRITIC_ON_DIR=$2
      shift 2
      ;;
    --critic-off)
      CRITIC_OFF_DIR=$2
      shift 2
      ;;
    --mode)
      MODE=$2
      shift 2
      ;;
    --skip-existing)
      SKIP_EXISTING=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$MODE" != "link" && "$MODE" != "copy" ]]; then
  echo "Unsupported --mode: $MODE (expected link or copy)" >&2
  exit 1
fi

if [[ -z "$CRITIC_ON_DIR" ]]; then
  CRITIC_ON_DIR="$SOURCE_ROOT/critic_on"
fi

if [[ -z "$CRITIC_OFF_DIR" ]]; then
  CRITIC_OFF_DIR="$SOURCE_ROOT/critic_off"
fi

HAS_ANY_GROUP=0
if [[ -d "$CRITIC_ON_DIR" ]]; then
  HAS_ANY_GROUP=1
fi
if [[ -d "$CRITIC_OFF_DIR" ]]; then
  HAS_ANY_GROUP=1
fi

if (( HAS_ANY_GROUP == 0 )); then
  echo "Did not find critic_on or critic_off directories." >&2
  echo "Checked:" >&2
  echo "  $CRITIC_ON_DIR" >&2
  echo "  $CRITIC_OFF_DIR" >&2
  exit 1
fi

(
  cd "$DATASET_TOOL_DIR"
  uv sync
)

if [[ -d "$CRITIC_ON_DIR" ]]; then
  import_group "critic_on" "$CRITIC_ON_DIR" "critic-on"
fi

if [[ -d "$CRITIC_OFF_DIR" ]]; then
  import_group "critic_off" "$CRITIC_OFF_DIR" "critic-off"
fi

echo "Rebuilding SceneSmith preview data"
(
  cd "$DATASET_TOOL_DIR"
  uv run dataset-downloader preprocess scenesmith
  uv run dataset-downloader renderable scenesmith
)

echo "Done."
echo "Imported sources:"
echo "  critic_on  -> $CRITIC_ON_DIR"
echo "  critic_off -> $CRITIC_OFF_DIR"
echo "Web scenes will appear under the SceneSmith dataset."
