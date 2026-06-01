from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

from .config import REPO_ROOT


HSSD_DATASET_KEY = "hssd"
HSSD_HAB_REPO_ID = "hssd/hssd-hab"
HSSD_HAB_SCENE_DATASET_CONFIG = "hssd-hab.scene_dataset_config.json"
HSSD_TARGET_UP = np.array([0.0, 1.0, 0.0], dtype=float)
HSSD_TARGET_FRONT = np.array([0.0, 0.0, -1.0], dtype=float)


def repo_relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def hssd_extracted_root(destination: Path) -> Path:
    return destination / "source" / "extracted"


def hssd_habitat_root(destination: Path) -> Path:
    return destination / "source" / "habitat"


def hssd_stage_root(destination: Path) -> Path:
    return hssd_extracted_root(destination) / "stages"


def hssd_stage_glb_path(destination: Path, scene_id: str) -> Path:
    return hssd_stage_root(destination) / f"{scene_id}.glb"


def hssd_scene_instance_path(destination: Path, scene_id: str) -> Path:
    return hssd_habitat_root(destination) / "scenes" / f"{scene_id}.scene_instance.json"


def hssd_stage_config_path(destination: Path, scene_id: str) -> Path:
    return hssd_habitat_root(destination) / "stages" / f"{scene_id}.stage_config.json"


def hssd_objects_metadata_path(destination: Path) -> Path:
    return hssd_habitat_root(destination) / "metadata" / "objects.json"


def _hssd_template_object_config_relative_path(template_name: str) -> str:
    if "-" in template_name and all(part.isdigit() for part in template_name.split("-")):
        return f"objects/openings/{template_name}.object_config.json"
    if "_part_" in template_name:
        base_id = template_name.split("_part_", 1)[0]
        return f"objects/decomposed/{base_id}/{template_name}.object_config.json"
    if "/" in template_name:
        return f"{template_name}.object_config.json"
    return f"objects/{template_name[0]}/{template_name}.object_config.json"


def hssd_object_config_path(destination: Path, template_name: str) -> Path:
    return hssd_habitat_root(destination) / _hssd_template_object_config_relative_path(template_name)


def hssd_object_glb_path_from_config(
    destination: Path,
    *,
    template_name: str,
    render_asset: str | None,
) -> Path:
    object_config_path = hssd_object_config_path(destination, template_name)
    if isinstance(render_asset, str) and render_asset:
        config_parent = object_config_path.parent.relative_to(hssd_habitat_root(destination))
        extracted_candidate = hssd_extracted_root(destination) / config_parent / render_asset
        if extracted_candidate.exists():
            return extracted_candidate
        habitat_candidate = hssd_habitat_root(destination) / config_parent / render_asset
        if habitat_candidate.exists():
            return habitat_candidate
        return extracted_candidate
    if "/" in template_name:
        return hssd_extracted_root(destination) / template_name
    if "-" in template_name and all(part.isdigit() for part in template_name.split("-")):
        extracted_candidate = hssd_extracted_root(destination) / "objects" / "openings" / f"{template_name}.glb"
        if extracted_candidate.exists():
            return extracted_candidate
        return hssd_habitat_root(destination) / "objects" / "openings" / f"{template_name}.glb"
    if "_part_" in template_name:
        base_id = template_name.split("_part_", 1)[0]
        extracted_candidate = (
            hssd_extracted_root(destination)
            / "objects"
            / "decomposed"
            / base_id
            / f"{template_name}.glb"
        )
        if extracted_candidate.exists():
            return extracted_candidate
        return (
            hssd_habitat_root(destination)
            / "objects"
            / "decomposed"
            / base_id
            / f"{template_name}.glb"
        )
    return hssd_extracted_root(destination) / "objects" / template_name[0] / f"{template_name}.glb"


def load_hssd_scene_instance(destination: Path, scene_id: str) -> dict[str, object]:
    return json.loads(hssd_scene_instance_path(destination, scene_id).read_text())


def load_hssd_object_config(destination: Path, template_name: str) -> dict[str, object]:
    return json.loads(hssd_object_config_path(destination, template_name).read_text())


def load_hssd_objects_metadata(destination: Path) -> dict[str, dict[str, object]]:
    metadata_path = hssd_objects_metadata_path(destination)
    if not metadata_path.exists():
        return {}
    payload = json.loads(metadata_path.read_text())
    return payload if isinstance(payload, dict) else {}


def _numeric_triplet(values: object) -> np.ndarray | None:
    if not isinstance(values, list) or len(values) < 3:
        return None
    result = []
    for value in values[:3]:
        if not isinstance(value, (int, float)):
            return None
        result.append(float(value))
    return np.asarray(result, dtype=float)


def _normalized_direction(values: object) -> np.ndarray | None:
    vector = _numeric_triplet(values)
    if vector is None:
        return None
    length = float(np.linalg.norm(vector))
    if length <= 1e-8:
        return None
    return vector / length


def hssd_alignment_rotation_matrix(
    up: object,
    front: object,
) -> np.ndarray | None:
    source_up = _normalized_direction(up)
    source_front = _normalized_direction(front)
    if source_up is None or source_front is None:
        return None

    if np.allclose(source_up, HSSD_TARGET_UP, atol=1e-6) and np.allclose(
        source_front, HSSD_TARGET_FRONT, atol=1e-6
    ):
        return None

    if np.allclose(source_front, source_up, atol=1e-6) or np.allclose(
        source_front, -source_up, atol=1e-6
    ):
        return None

    source_right = np.cross(source_front, source_up)
    source_right_norm = float(np.linalg.norm(source_right))
    if source_right_norm <= 1e-8:
        return None
    source_right = source_right / source_right_norm

    target_right = np.cross(HSSD_TARGET_FRONT, HSSD_TARGET_UP)
    target_right = target_right / np.linalg.norm(target_right)

    source_basis = np.column_stack((source_right, source_up, source_front))
    target_basis = np.column_stack((target_right, HSSD_TARGET_UP, HSSD_TARGET_FRONT))
    return target_basis @ np.linalg.inv(source_basis)


def hssd_rotation_matrix_from_wxyz(quaternion: object) -> np.ndarray | None:
    if not isinstance(quaternion, list) or len(quaternion) < 4:
        return None

    if not all(isinstance(value, (int, float)) for value in quaternion[:4]):
        return None

    w, x, y, z = (float(value) for value in quaternion[:4])
    length = math.sqrt(w * w + x * x + y * y + z * z)
    if length <= 1e-8:
        return np.identity(3, dtype=float)

    w /= length
    x /= length
    y /= length
    z /= length
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=float,
    )


def hssd_three_quaternion_from_instance(
    rotation: object,
    *,
    up: object,
    front: object,
) -> list[float] | None:
    scene_rotation = hssd_rotation_matrix_from_wxyz(rotation)
    alignment = hssd_alignment_rotation_matrix(up, front)
    if scene_rotation is None and alignment is None:
        return None

    if scene_rotation is None:
        combined = alignment
    elif alignment is None:
        combined = scene_rotation
    else:
        combined = scene_rotation @ alignment

    if combined is None:
        return None

    trace = float(combined[0, 0] + combined[1, 1] + combined[2, 2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (combined[2, 1] - combined[1, 2]) / scale
        y = (combined[0, 2] - combined[2, 0]) / scale
        z = (combined[1, 0] - combined[0, 1]) / scale
    elif combined[0, 0] > combined[1, 1] and combined[0, 0] > combined[2, 2]:
        scale = math.sqrt(1.0 + combined[0, 0] - combined[1, 1] - combined[2, 2]) * 2.0
        w = (combined[2, 1] - combined[1, 2]) / scale
        x = 0.25 * scale
        y = (combined[0, 1] + combined[1, 0]) / scale
        z = (combined[0, 2] + combined[2, 0]) / scale
    elif combined[1, 1] > combined[2, 2]:
        scale = math.sqrt(1.0 + combined[1, 1] - combined[0, 0] - combined[2, 2]) * 2.0
        w = (combined[0, 2] - combined[2, 0]) / scale
        x = (combined[0, 1] + combined[1, 0]) / scale
        y = 0.25 * scale
        z = (combined[1, 2] + combined[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + combined[2, 2] - combined[0, 0] - combined[1, 1]) * 2.0
        w = (combined[1, 0] - combined[0, 1]) / scale
        x = (combined[0, 2] + combined[2, 0]) / scale
        y = (combined[1, 2] + combined[2, 1]) / scale
        z = 0.25 * scale

    length = math.sqrt(x * x + y * y + z * z + w * w)
    if length <= 1e-8:
        return None
    return [x / length, y / length, z / length, w / length]


def download_hssd_habitat_metadata(
    *,
    destination_root: Path,
    scene_ids: list[str],
    force_download: bool,
    max_workers: int,
) -> dict[str, object]:
    habitat_root = hssd_habitat_root(destination_root)
    habitat_root.mkdir(parents=True, exist_ok=True)

    scene_ids = sorted(set(scene_ids))
    base_patterns = [
        HSSD_HAB_SCENE_DATASET_CONFIG,
        "metadata/objects.json",
        "metadata/hssd_obj_semantics_condensed.csv",
    ]
    scene_patterns = [f"scenes/{scene_id}.scene_instance.json" for scene_id in scene_ids]
    stage_patterns = [f"stages/{scene_id}.stage_config.json" for scene_id in scene_ids]

    first_pass = snapshot_download(
        repo_id=HSSD_HAB_REPO_ID,
        repo_type="dataset",
        local_dir=habitat_root,
        allow_patterns=base_patterns + scene_patterns + stage_patterns,
        force_download=force_download,
        max_workers=max_workers,
    )

    template_names: set[str] = set()
    missing_scene_ids: list[str] = []
    for scene_id in scene_ids:
        scene_path = hssd_scene_instance_path(destination_root, scene_id)
        if not scene_path.exists():
            missing_scene_ids.append(scene_id)
            continue
        payload = json.loads(scene_path.read_text())
        for key in ("object_instances", "articulated_object_instances"):
            for entry in payload.get(key, []):
                if not isinstance(entry, dict):
                    continue
                template_name = entry.get("template_name")
                if isinstance(template_name, str) and template_name:
                    template_names.add(template_name)

    object_patterns = sorted(
        {
            pattern
            for template_name in template_names
            for pattern in (
                [
                    _hssd_template_object_config_relative_path(template_name),
                    (
                        f"objects/decomposed/{template_name.split('_part_', 1)[0]}/{template_name}.glb"
                        if "_part_" in template_name
                        else None
                    ),
                ]
            )
            if isinstance(pattern, str)
        }
    )
    second_pass: list[str] | str = []
    if object_patterns:
        object_result = snapshot_download(
            repo_id=HSSD_HAB_REPO_ID,
            repo_type="dataset",
            local_dir=habitat_root,
            allow_patterns=object_patterns,
            force_download=force_download,
            max_workers=max_workers,
        )
        second_pass = (
            [item.filename for item in object_result]
            if isinstance(object_result, list)
            else str(object_result)
        )

    return {
        "repo_id": HSSD_HAB_REPO_ID,
        "target_dir": str(habitat_root),
        "scene_count": len(scene_ids),
        "scene_instance_count": len(scene_patterns) - len(missing_scene_ids),
        "stage_config_count": len(stage_patterns),
        "object_template_count": len(template_names),
        "object_config_count": len(object_patterns),
        "missing_scene_ids": missing_scene_ids,
        "base_result": (
            [item.filename for item in first_pass]
            if isinstance(first_pass, list)
            else str(first_pass)
        ),
        "object_result": second_pass,
    }
