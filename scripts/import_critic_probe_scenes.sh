#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
DATASET_TOOL_DIR="$REPO_ROOT/scripts/dataset_downloader"

SOURCE_ROOT=${1:-/data/task3_2/L202500266_hrk/code/scenesmith/outputs/critic_probe/2026-06-24_21-21-36}

declare -A SUBSET_NAMES=(
  [critic_off]=critic-off
  [critic_on]=critic-on
)

require_dir() {
  local path=$1
  if [[ ! -d "$path" ]]; then
    echo "Missing directory: $path" >&2
    exit 1
  fi
}

import_probe_group() {
  local group_name=$1
  local subset_name=$2
  local group_dir="$SOURCE_ROOT/$group_name"

  require_dir "$group_dir"

  mapfile -t batch_dirs < <(find "$group_dir" -mindepth 1 -maxdepth 1 -type d -name 'batch_*' | sort)
  if [[ ${#batch_dirs[@]} -eq 0 ]]; then
    echo "No batch directories found under: $group_dir" >&2
    exit 1
  fi

  echo "Importing $group_name into subset '$subset_name' (${#batch_dirs[@]} batches)"
  for batch_dir in "${batch_dirs[@]}"; do
    echo "  - $(basename "$batch_dir")"
    (
      cd "$DATASET_TOOL_DIR"
      uv run dataset-downloader import-scenesmith-local \
        "$batch_dir" \
        --subset "$subset_name"
    )
  done
}

main() {
  require_dir "$SOURCE_ROOT"

  (
    cd "$DATASET_TOOL_DIR"
    uv sync
  )

  import_probe_group "critic_off" "${SUBSET_NAMES[critic_off]}"
  import_probe_group "critic_on" "${SUBSET_NAMES[critic_on]}"

  echo "Rebuilding SceneSmith preview data"
  (
    cd "$DATASET_TOOL_DIR"
    uv run dataset-downloader preprocess scenesmith
    uv run dataset-downloader renderable scenesmith
  )

  echo "Done. Imported critic probe scenes from: $SOURCE_ROOT"
  echo "Web scenes will appear under the SceneSmith dataset with subsets:"
  echo "  - ${SUBSET_NAMES[critic_off]}"
  echo "  - ${SUBSET_NAMES[critic_on]}"
}

main "$@"
