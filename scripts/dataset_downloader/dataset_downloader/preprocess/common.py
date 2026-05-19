from __future__ import annotations

import json
import math
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import yaml

from ..config import DATASETS, PREPROCESSED_ROOT, REPO_ROOT
from ..front3d import (
    FRONT3D_DATASET_KEY,
    FRONT3D_LAYOUT_ZIP,
    FRONT3D_MODEL_ZIP,
    FRONT3D_TEXTURE_ZIP,
    ensure_front3d_archives,
    map_front3d_shell_category,
    repo_relative_path,
    safe_front3d_name,
)
from ..hsm import (
    HSM_DATASET_KEY,
    hsm_downloaded_support_asset,
    hsm_generated_scenes_root,
    load_hsm_hssd_metadata,
    hsm_object_model_id,
    hsm_position_from_transform,
    hsm_scale_from_transform,
    hsm_scene_id_from_remote_path,
    hsm_yaw_deg_from_transform,
    load_hsm_scene,
)


SCHEMA_VERSION = 1
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _repo_path(path: Path) -> str:
    return repo_relative_path(path)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def _sorted_files(directory: Path, suffixes: set[str]) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _collect_preview_images(scene_dir: Path) -> list[str]:
    preview_dir = scene_dir / "preview"
    return [_repo_path(path) for path in _sorted_files(preview_dir, IMAGE_SUFFIXES)]


def _scene_manifest_base(
    *,
    dataset: str,
    scene_id: str,
    scene_uid: str,
    subset: str | None,
    scene_dir: Path,
    description: str | None,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _now_utc(),
        "dataset": dataset,
        "scene_id": scene_id,
        "scene_uid": scene_uid,
        "subset": subset,
        "description": description,
        "source": {
            "extracted_dir": _repo_path(scene_dir),
        },
    }


def _sage_scene_output_dir(scene_id: str) -> Path:
    return PREPROCESSED_ROOT / "sage" / scene_id


def _scenesmith_scene_output_dir(subset: str, scene_id: str) -> Path:
    return PREPROCESSED_ROOT / "scenesmith" / subset / scene_id


def _front3d_scene_output_dir(house_id: str, room_id: str) -> Path:
    return PREPROCESSED_ROOT / FRONT3D_DATASET_KEY / safe_front3d_name(house_id) / safe_front3d_name(room_id)


def _front3d_scalar_triplet(values: object) -> list[float] | None:
    if not isinstance(values, list) or len(values) < 3:
        return None
    result: list[float] = []
    for value in values[:3]:
        if not isinstance(value, (int, float)):
            return None
        result.append(float(value))
    return result


def _front3d_scalar_quaternion(values: object) -> list[float] | None:
    if not isinstance(values, list) or len(values) < 4:
        return None
    result: list[float] = []
    for value in values[:4]:
        if not isinstance(value, (int, float)):
            return None
        result.append(float(value))
    return result


def _front3d_position(values: object) -> dict[str, float] | None:
    triplet = _front3d_scalar_triplet(values)
    if not triplet:
        return None
    return {
        "x": triplet[0],
        "y": triplet[1],
        "z": triplet[2],
    }


def _front3d_dimensions(size_values: object) -> dict[str, float] | None:
    triplet = _front3d_scalar_triplet(size_values)
    if not triplet:
        return None
    return {
        "width": triplet[0],
        "length": triplet[1],
        "height": triplet[2],
    }


def _front3d_bbox_from_position_and_size(
    *,
    position: dict[str, float] | None,
    dimensions: dict[str, float] | None,
) -> tuple[list[float], list[float]] | tuple[None, None]:
    if not position or not dimensions:
        return None, None
    half_width = max(dimensions["width"], 0.0) / 2.0
    half_length = max(dimensions["length"], 0.0) / 2.0
    height = max(dimensions["height"], 0.0)
    return (
        [
            position["x"] - half_width,
            position["y"],
            position["z"] - half_length,
        ],
        [
            position["x"] + half_width,
            position["y"] + height,
            position["z"] + half_length,
        ],
    )


def _front3d_room_bounds(
    room_children: list[dict[str, object]],
    mesh_by_uid: dict[str, dict[str, object]],
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    coords: list[float] = []
    for child in room_children:
        ref = child.get("ref")
        if not isinstance(ref, str):
            continue
        mesh = mesh_by_uid.get(ref)
        if not mesh:
            continue
        xyz = mesh.get("xyz")
        if isinstance(xyz, list):
            coords.extend(value for value in xyz if isinstance(value, (int, float)))

    if len(coords) < 3:
        return None, None

    xs = coords[0::3]
    ys = coords[1::3]
    zs = coords[2::3]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    return (
        {
            "x": float((min_x + max_x) / 2.0),
            "y": float(min_y),
            "z": float((min_z + max_z) / 2.0),
        },
        {
            "width": float(max_x - min_x),
            "length": float(max_z - min_z),
            "height": float(max_y - min_y),
        },
    )


def _front3d_shell_ref(
    child: dict[str, object],
    mesh: dict[str, object],
) -> dict[str, object]:
    return {
        "id": child.get("instanceid") or mesh.get("uid"),
        "mesh_uid": mesh.get("uid"),
        "mesh_type": mesh.get("type"),
        "category": map_front3d_shell_category(mesh.get("type")),
        "material_uid": mesh.get("material"),
        "position": _front3d_position(child.get("pos")),
        "scale": _front3d_scalar_triplet(child.get("scale")),
        "rotation_quaternion": _front3d_scalar_quaternion(child.get("rot")),
    }


def _front3d_object_entry(
    room_id: str,
    child: dict[str, object],
    furniture: dict[str, object],
) -> dict[str, object]:
    dimensions = _front3d_dimensions(furniture.get("size"))
    position = _front3d_position(child.get("pos"))
    bbox_min, bbox_max = _front3d_bbox_from_position_and_size(
        position=position,
        dimensions=dimensions,
    )
    return {
        "id": child.get("instanceid") or furniture.get("uid"),
        "room_id": room_id,
        "type": furniture.get("category") or furniture.get("type"),
        "name": furniture.get("title"),
        "description": furniture.get("title"),
        "position": position,
        "dimensions": dimensions,
        "quaternion": _front3d_scalar_quaternion(child.get("rot")),
        "scale": _front3d_scalar_triplet(child.get("scale")),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "source_id": furniture.get("uid"),
        "metadata": {
            "jid": furniture.get("jid"),
            "valid": bool(furniture.get("valid", True)),
            "type": furniture.get("type"),
            "source_category_id": furniture.get("sourceCategoryId"),
            "bbox": furniture.get("bbox"),
            "component_modifiers": child.get("componentModifiers") or [],
            "source_ref": child.get("ref"),
        },
    }


def _normalize_sage_object(scene_dir: Path, room_id: str, obj: dict[str, object]) -> dict[str, object]:
    source_id = obj.get("source_id")
    mesh_path = scene_dir / "objects" / f"{source_id}.ply" if source_id else None
    texture_path = (
        scene_dir / "objects" / f"{source_id}_texture.png" if source_id else None
    )
    return {
        "id": obj.get("id"),
        "room_id": room_id,
        "type": obj.get("type"),
        "description": obj.get("description"),
        "position": obj.get("position"),
        "rotation": obj.get("rotation"),
        "dimensions": obj.get("dimensions"),
        "source": obj.get("source"),
        "source_id": source_id,
        "place_id": obj.get("place_id"),
        "place_guidance": obj.get("place_guidance"),
        "mass": obj.get("mass"),
        "pbr_parameters": obj.get("pbr_parameters"),
        "mesh": {
            "ply": _repo_path(mesh_path) if mesh_path and mesh_path.exists() else None,
            "texture": (
                _repo_path(texture_path)
                if texture_path and texture_path.exists()
                else None
            ),
        },
    }



__all__ = [name for name in globals() if not name.startswith("__")]
