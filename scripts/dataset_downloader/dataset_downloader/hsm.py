from __future__ import annotations

import csv
import io
import json
import math
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .config import ASSETS_ROOT, REPO_ROOT
from . import env_config


HSM_DATASET_KEY = "hsm"

def _resolve_hsm_hssd_root() -> Path:
    """Resolve HSSD models root from env/config, falling back to assets/hsm/hssd-models."""
    configured = env_config.get_path("SCENEVIEWER_HSSD_ROOT")
    if configured is not None and configured.exists():
        return configured
    return ASSETS_ROOT / HSM_DATASET_KEY / "hssd-models"

def get_hsm_hssd_root() -> Path:
    return _resolve_hsm_hssd_root()

HSM_HSSD_ROOT = _resolve_hsm_hssd_root()
HSM_METADATA_ROOT = ASSETS_ROOT / HSM_DATASET_KEY / "metadata"
HSM_HSSD_MODELS_REPO_ID = "hssd/hssd-models"
HSM_HSSD_DECOMPOSED_REPO_ID = "hssd/hssd-hab"
HSM_RELEASE_TAG = "1.0.0"
HSM_RELEASE_DATA_URL = (
    f"https://github.com/3dlg-hcvc/hsm/releases/download/{HSM_RELEASE_TAG}/data.zip"
)
HSM_RELEASE_METADATA_MEMBERS = {
    "object_categories": "data/preprocessed/object_categories.json",
    "hssd_index": "data/preprocessed/hssd_wnsynsetkey_index.json",
}
SCENEVAL_ANNOTATIONS_URL = (
    "https://github.com/3dlg-hcvc/SceneEval/releases/download/"
    "SceneEval-500_v250610/SceneEval-500_v250610.zip"
)
HSM_EXPORT_FIX_MATRIX = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)


def ensure_hsm_hssd_models() -> None:
    hssd_root = get_hsm_hssd_root()
    required_root = hssd_root / "objects"
    if required_root.exists():
        return
    raise SystemExit(
        "Missing HSSD model assets for HSM renderable export. "
        "Either:\n"
        "  1. Place the HSSD dataset under assets/hsm/hssd-models/.\n"
        "  2. Set SCENEVIEWER_HSSD_ROOT in .env or config.yml to point to an existing HSSD checkout.\n"
        "     Example: SCENEVIEWER_HSSD_ROOT=/home/user/proj/scenesmith/data/hssd-models"
    )


def repo_relative_path(path: Path) -> str:
    # Use absolute() instead of resolve() to avoid following symlinks.
    # This keeps paths inside the repo tree even when assets are symlinked
    # to external directories (e.g. HSSD models).
    abs_path = path.absolute()
    try:
        return abs_path.relative_to(REPO_ROOT.absolute()).as_posix()
    except ValueError:
        return abs_path.as_posix()


def hsm_repo_relative_path(path: Path) -> str:
    """Return a stable repo-local path for HSM assets when a repo alias exists.

    If HSM models are sourced from an external checkout via ``SCENEVIEWER_HSSD_ROOT``
    but also exposed under ``assets/hsm/hssd-models`` (commonly as a symlink),
    prefer exporting that repo-local alias so the web app can serve the file.
    """

    alias_root = ASSETS_ROOT / HSM_DATASET_KEY / "hssd-models"
    try:
        relative_to_hssd_root = path.resolve().relative_to(get_hsm_hssd_root().resolve())
    except ValueError:
        return repo_relative_path(path)

    alias_path = alias_root / relative_to_hssd_root
    return repo_relative_path(alias_path)


def hsm_generated_scenes_root(destination: Path) -> Path:
    return destination / "source" / "raw" / "generated_scenes"


def hsm_support_region_root(destination: Path) -> Path:
    return destination / "source" / "raw" / "support_region_dataset"


def hsm_hssd_glb_path(mesh_id: str) -> Path:
    hssd_root = get_hsm_hssd_root()
    if "_part_" in mesh_id:
        base_id = mesh_id.split("_part_", 1)[0]
        return hssd_root / "objects" / "decomposed" / base_id / f"{mesh_id}.glb"
    return hssd_root / "objects" / mesh_id[0] / f"{mesh_id}.glb"


def _hsm_metadata_path(kind: str) -> Path:
    return HSM_METADATA_ROOT / Path(HSM_RELEASE_METADATA_MEMBERS[kind]).name


def ensure_hsm_metadata() -> None:
    missing = [kind for kind in HSM_RELEASE_METADATA_MEMBERS if not _hsm_metadata_path(kind).exists()]
    if not missing:
        return

    HSM_METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    archive_path = HSM_METADATA_ROOT / f"hsm_release_{HSM_RELEASE_TAG}_data.zip"
    urllib.request.urlretrieve(HSM_RELEASE_DATA_URL, archive_path)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for kind in missing:
                member = HSM_RELEASE_METADATA_MEMBERS[kind]
                destination = _hsm_metadata_path(kind)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
    finally:
        archive_path.unlink(missing_ok=True)


_SCENEAVAL_ANNOTATIONS_PATH = HSM_METADATA_ROOT / "annotations.csv"


def ensure_sceneval_annotations() -> None:
    """Download SceneEval-500 annotations.csv if not already present."""
    if _SCENEAVAL_ANNOTATIONS_PATH.exists():
        return
    HSM_METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    archive_path = HSM_METADATA_ROOT / "SceneEval-500.zip"
    urllib.request.urlretrieve(SCENEVAL_ANNOTATIONS_URL, archive_path)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            with archive.open("annotations.csv") as source, _SCENEAVAL_ANNOTATIONS_PATH.open("wb") as target:
                shutil.copyfileobj(source, target)
    finally:
        archive_path.unlink(missing_ok=True)


def load_sceneval_annotations() -> dict[int, dict[str, str]]:
    """Load SceneEval-500 annotations as a dict mapping scene ID to annotation fields."""
    ensure_sceneval_annotations()
    result: dict[int, dict[str, str]] = {}
    if not _SCENEAVAL_ANNOTATIONS_PATH.exists():
        return result
    with _SCENEAVAL_ANNOTATIONS_PATH.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                scene_id = int(row["ID"])
            except (KeyError, ValueError):
                continue
            result[scene_id] = dict(row)
    return result


def _iter_hssd_index_entries(data: object):
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                yield None, entry
        return

    if isinstance(data, dict):
        for category, value in data.items():
            if isinstance(value, list):
                for entry in value:
                    if isinstance(entry, dict):
                        yield category, entry
            elif isinstance(value, dict):
                yield category, value


def _humanize_wnsynsetkey(wnsynsetkey: str | None) -> str | None:
    if not isinstance(wnsynsetkey, str) or not wnsynsetkey:
        return None
    token = wnsynsetkey.split(",", 1)[0].strip()
    token = token.split(".n.", 1)[0]
    token = token.replace("_", " ").strip()
    return token or None


def load_hsm_hssd_metadata() -> dict[str, dict[str, object]]:
    try:
        ensure_hsm_metadata()
        index_path = _hsm_metadata_path("hssd_index")
        payload = json.loads(index_path.read_text())
    except Exception:
        return {}
    result: dict[str, dict[str, object]] = {}
    for category, entry in _iter_hssd_index_entries(payload):
        mesh_id = entry.get("id")
        if not isinstance(mesh_id, str) or not mesh_id:
            continue
        wnsynsetkey = category if isinstance(category, str) else None
        result[mesh_id] = {
            "name": entry.get("name") if isinstance(entry.get("name"), str) else None,
            "wnsynsetkey": wnsynsetkey,
            "semantic_label": _humanize_wnsynsetkey(wnsynsetkey),
            "support_region": bool(entry.get("support_region")),
            "up": entry.get("up") if isinstance(entry.get("up"), str) else None,
            "front": entry.get("front") if isinstance(entry.get("front"), str) else None,
        }
    return result


def hsm_downloaded_support_asset(destination: Path, mesh_id: str, *, surface_only: bool = False) -> Path | None:
    roots = ["annot_surface"] if surface_only else ["annot", "annot_surface"]
    base_root = hsm_support_region_root(destination)
    for child in roots:
        direct_candidate = base_root / child / f"{mesh_id}.glb"
        if direct_candidate.exists():
            return direct_candidate
        nested_matches = sorted(base_root.rglob(f"{child}/{mesh_id}.glb"))
        if nested_matches:
            return nested_matches[0]
    return None


def hsm_scene_id_from_remote_path(path: str) -> str:
    return Path(path).stem


def hsm_object_model_id(raw_model_id: object) -> str | None:
    if not isinstance(raw_model_id, str):
        return None
    if "." in raw_model_id:
        _, _, suffix = raw_model_id.partition(".")
        return suffix or None
    return raw_model_id or None


def hsm_transform_matrix(transform_block: object) -> list[float] | None:
    if not isinstance(transform_block, dict):
        return None
    data = transform_block.get("data")
    if not isinstance(data, list) or len(data) < 16:
        return None
    values: list[float] = []
    for value in data[:16]:
        if not isinstance(value, (int, float)):
            return None
        values.append(float(value))
    return values


def hsm_position_from_transform(transform_block: object) -> dict[str, float] | None:
    values = hsm_transform_matrix(transform_block)
    if not values:
        return None
    return {
        "x": values[12],
        "y": values[13],
        "z": values[14],
    }


def hsm_scale_from_transform(transform_block: object) -> list[float] | None:
    values = hsm_transform_matrix(transform_block)
    if not values:
        return None
    columns = (
        (values[0], values[1], values[2]),
        (values[4], values[5], values[6]),
        (values[8], values[9], values[10]),
    )
    return [math.sqrt(sum(component * component for component in column)) for column in columns]


def _parse_hssd_vec(vec_str: str | None) -> np.ndarray | None:
    if not isinstance(vec_str, str) or not vec_str.strip():
        return None
    cleaned = vec_str.strip().replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    parts = [part for part in cleaned.replace(",", " ").split() if part]
    if len(parts) != 3:
        return None
    try:
        return np.array([float(part) for part in parts], dtype=float)
    except ValueError:
        return None


def hsm_alignment_rotation_matrix(up: str | None, front: str | None) -> np.ndarray | None:
    up_array = _parse_hssd_vec(up)
    front_array = _parse_hssd_vec(front)
    if up_array is None or front_array is None:
        return None

    target_front = np.array([0.0, 0.0, 1.0], dtype=float)
    target_up = np.array([0.0, 1.0, 0.0], dtype=float)
    if np.allclose(up_array, target_up) and np.allclose(front_array, target_front):
        return None

    up_norm = np.linalg.norm(up_array)
    front_norm = np.linalg.norm(front_array)
    if up_norm < 1e-6 or front_norm < 1e-6:
        return None

    norm_up = up_array / up_norm
    norm_front = front_array / front_norm
    if np.allclose(norm_front, norm_up) or np.allclose(norm_front, -norm_up):
        return None

    right = np.cross(norm_front, norm_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        return None

    norm_right = right / right_norm
    target_right = np.cross(target_front, target_up)
    source_matrix = np.column_stack((norm_right, norm_front, norm_up))
    target_matrix = np.column_stack((target_right, target_front, target_up))
    return target_matrix @ source_matrix.T


def hsm_rotation_matrix_from_transform(transform_block: object) -> np.ndarray | None:
    values = hsm_transform_matrix(transform_block)
    if not values:
        return None

    matrix = np.array(values, dtype=float).reshape((4, 4), order="F")
    rotation_scale = matrix[:3, :3]
    column_norms = np.linalg.norm(rotation_scale, axis=0)
    safe_norms = np.where(column_norms > 1e-8, column_norms, 1.0)
    normalized = rotation_scale / safe_norms
    u, _, vh = np.linalg.svd(normalized)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    # HSM exports sceneState transforms after applying stk_utils.sv_fix_coordinates().
    # Undo that fixed basis so we can recover the original scene-space yaw.
    return HSM_EXPORT_FIX_MATRIX.T @ rotation


def _quaternion_from_rotation_matrix(rotation: np.ndarray) -> list[float]:
    trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s

    length = math.sqrt(x * x + y * y + z * z + w * w)
    if length <= 1e-8:
        return [0.0, 0.0, 0.0, 1.0]
    return [x / length, y / length, z / length, w / length]


def hsm_three_quaternion_from_transform(
    transform_block: object,
    *,
    up: str | None = None,
    front: str | None = None,
) -> list[float] | None:
    alignment = hsm_alignment_rotation_matrix(up, front)
    if alignment is None:
        return None

    yaw_rad = math.radians(hsm_yaw_deg_from_transform(transform_block))
    yaw_rotation = np.array(
        [
            [math.cos(yaw_rad), 0.0, math.sin(yaw_rad)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw_rad), 0.0, math.cos(yaw_rad)],
        ],
        dtype=float,
    )
    return _quaternion_from_rotation_matrix(yaw_rotation @ alignment)


def hsm_yaw_deg_from_transform(transform_block: object) -> float:
    rotation = hsm_rotation_matrix_from_transform(transform_block)
    if rotation is None:
        return 0.0
    return math.degrees(math.atan2(rotation[2, 0], rotation[0, 0]))


def hsm_scene_model_ids(scene_payload: dict[str, object]) -> set[str]:
    result: set[str] = set()
    scene = scene_payload.get("scene")
    if not isinstance(scene, dict):
        return result
    for obj in scene.get("object", []):
        if not isinstance(obj, dict):
            continue
        model_id = hsm_object_model_id(obj.get("modelId"))
        if model_id:
            result.add(model_id)
    return result


def load_hsm_scene(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())
