#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
ASSETS_DIR="$REPO_ROOT/assets"
ASSUME_YES=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/dataset_downloader/clear_assets.sh [--yes]

Description:
  Remove everything under SceneViewer/assets and recreate assets/.gitkeep.

Options:
  --yes     Skip the confirmation prompt.
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes)
      ASSUME_YES=1
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

if (( ASSUME_YES == 0 )); then
  echo "This will permanently remove everything under:"
  echo "  $ASSETS_DIR"
  read -r -p "Type 'yes' to continue: " reply
  if [[ "$reply" != "yes" ]]; then
    echo "Aborted."
    exit 1
  fi
fi

mkdir -p "$ASSETS_DIR"
find "$ASSETS_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
touch "$ASSETS_DIR/.gitkeep"

echo "Cleared assets directory:"
echo "  $ASSETS_DIR"
