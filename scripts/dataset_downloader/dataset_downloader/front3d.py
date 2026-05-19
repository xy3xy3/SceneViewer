from __future__ import annotations

import re
from pathlib import Path

from .config import ASSETS_ROOT, REPO_ROOT


FRONT3D_LAYOUT_ZIP = ASSETS_ROOT / "3D-FRONT.zip"
FRONT3D_MODEL_ZIP = ASSETS_ROOT / "3D-FUTURE-model.zip"
FRONT3D_TEXTURE_ZIP = ASSETS_ROOT / "3D-FRONT-texture.zip"

FRONT3D_DATASET_KEY = "3dfront"
FRONT3D_TEXTURE_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def front3d_required_archives() -> list[Path]:
    return [FRONT3D_LAYOUT_ZIP, FRONT3D_MODEL_ZIP, FRONT3D_TEXTURE_ZIP]


def ensure_front3d_archives() -> None:
    missing = [path for path in front3d_required_archives() if not path.exists()]
    if missing:
        missing_paths = ", ".join(path.name for path in missing)
        raise SystemExit(
            "Missing required 3D-FRONT archives in assets/: "
            f"{missing_paths}. Download them manually before running preprocess/renderable."
        )


def repo_relative_path(path: Path) -> str:
    absolute_path = path if path.is_absolute() else path.absolute()
    try:
        return absolute_path.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return absolute_path.resolve().as_posix()


def safe_front3d_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "item"


def front3d_texture_member_candidates(texture_jid: str) -> list[str]:
    base = f"3D-FRONT-texture/{texture_jid}"
    return [f"{base}/texture{suffix}" for suffix in FRONT3D_TEXTURE_IMAGE_SUFFIXES]


def map_front3d_shell_category(raw_type: str | None) -> str:
    if raw_type in {"Floor", "SlabBottom"}:
        return "floor"
    if raw_type == "Window":
        return "window"
    if raw_type == "Door":
        return "door"
    if raw_type == "Ceiling":
        return "ceiling"
    if raw_type in {"Baseboard", "Pocket", "Back"}:
        return "trim"
    if raw_type in {"CustomizedFeatureWall", "ExtrusionCustomizedBackgroundWall"}:
        return "feature"
    return "wall"
