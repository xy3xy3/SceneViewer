from __future__ import annotations

import csv

from .common import *
from ..hssd import (
    HSSD_DATASET_KEY,
    download_hssd_habitat_metadata,
    hssd_object_config_path,
    hssd_object_glb_path_from_config,
    hssd_objects_metadata_path,
    hssd_scene_instance_path,
    hssd_stage_config_path,
)


def _hssd_scene_output_dir(scene_id: str) -> Path:
    return PREPROCESSED_ROOT / "hssd" / scene_id


def _hssd_semantic_labels(destination_root: Path) -> dict[str, str]:
    csv_path = destination_root / "source" / "habitat" / "metadata" / "hssd_obj_semantics_condensed.csv"
    if not csv_path.exists():
        return {}

    with csv_path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        result: dict[str, str] = {}
        for row in rows:
            mesh_id = row.get("Object Hash")
            condensed = row.get("Semantic Category:\nCONDENSED")
            if isinstance(mesh_id, str) and mesh_id and isinstance(condensed, str) and condensed:
                result[mesh_id] = condensed
        return result


def _hssd_object_metadata_by_id(destination_root: Path) -> dict[str, dict[str, object]]:
    metadata_path = hssd_objects_metadata_path(destination_root)
    if not metadata_path.exists():
        return {}
    payload = json.loads(metadata_path.read_text())
    return payload if isinstance(payload, dict) else {}


def _hssd_stage_ids(stage_root: Path) -> list[str]:
    return sorted(path.stem for path in stage_root.glob("*.glb"))


def _hssd_object_entry(
    *,
    destination_root: Path,
    scene_id: str,
    object_index: int,
    payload: dict[str, object],
    metadata_by_id: dict[str, dict[str, object]],
    semantic_labels: dict[str, str],
) -> dict[str, object] | None:
    template_name = payload.get("template_name")
    if not isinstance(template_name, str) or not template_name:
        return None

    config_path = hssd_object_config_path(destination_root, template_name)
    if not config_path.exists():
        return None
    object_config = json.loads(config_path.read_text())
    render_asset = (
        object_config.get("render_asset") if isinstance(object_config.get("render_asset"), str) else None
    )
    glb_path = hssd_object_glb_path_from_config(
        destination_root,
        template_name=template_name,
        render_asset=render_asset,
    )
    translation = payload.get("translation") if isinstance(payload.get("translation"), list) else []
    scale = payload.get("non_uniform_scale") if isinstance(payload.get("non_uniform_scale"), list) else []
    metadata = metadata_by_id.get(template_name, {})
    semantic_label = semantic_labels.get(template_name)
    object_type = (
        semantic_label
        or (metadata.get("type") if isinstance(metadata.get("type"), str) else None)
        or "object"
    )
    name = metadata.get("name") if isinstance(metadata.get("name"), str) else template_name

    return {
        "id": f"{scene_id}__obj_{object_index:04d}",
        "room_id": "stage_0",
        "type": object_type,
        "name": name,
        "description": name,
        "position": {
            "x": float(translation[0]) if len(translation) >= 1 else 0.0,
            "y": float(translation[1]) if len(translation) >= 2 else 0.0,
            "z": float(translation[2]) if len(translation) >= 3 else 0.0,
        },
        "object_type": object_type,
        "gltf_path": _repo_path(glb_path) if glb_path.exists() else None,
        "quaternion": (
            [
                float(payload["rotation"][1]),
                float(payload["rotation"][2]),
                float(payload["rotation"][3]),
                float(payload["rotation"][0]),
            ]
            if isinstance(payload.get("rotation"), list)
            and len(payload["rotation"]) >= 4
            and all(isinstance(value, (int, float)) for value in payload["rotation"][:4])
            else None
        ),
        "scale": [
            float(scale[0]) if len(scale) >= 1 else 1.0,
            float(scale[1]) if len(scale) >= 2 else 1.0,
            float(scale[2]) if len(scale) >= 3 else 1.0,
        ],
        "metadata": {
            "source_id": template_name,
            "hssd_name": name,
            "hssd_type": metadata.get("type"),
            "hssd_semantic_label": semantic_label,
            "hssd_motion_type": payload.get("motion_type"),
            "hssd_rotation_wxyz": payload.get("rotation"),
            "hssd_up": object_config.get("up"),
            "hssd_front": object_config.get("front"),
            "hssd_render_asset": render_asset,
            "hssd_object_config": _repo_path(config_path),
        },
    }


def preprocess_hssd_dataset(scene_limit: int | None = None) -> dict[str, object]:
    destination_root = DATASETS[HSSD_DATASET_KEY].destination_root
    source_root = destination_root / "source" / "extracted"
    stage_root = source_root / "stages"
    output_root = PREPROCESSED_ROOT / "hssd"
    scenes: list[dict[str, object]] = []
    skipped_scenes: list[dict[str, object]] = []

    if output_root.exists():
        shutil.rmtree(output_root)

    if stage_root.exists():
        stage_ids = _hssd_stage_ids(stage_root)
        if stage_ids and any(
            not hssd_scene_instance_path(destination_root, scene_id).exists()
            for scene_id in stage_ids
        ):
            download_hssd_habitat_metadata(
                destination_root=destination_root,
                scene_ids=stage_ids,
                force_download=False,
                max_workers=8,
            )

        metadata_by_id = _hssd_object_metadata_by_id(destination_root)
        semantic_labels = _hssd_semantic_labels(destination_root)
        for stage_path in sorted(path for path in stage_root.iterdir() if path.is_file()):
            if scene_limit is not None and len(scenes) >= scene_limit:
                break

            if stage_path.suffix.lower() != ".glb":
                skipped_scenes.append(
                    {
                        "scene_id": stage_path.stem,
                        "scene_uid": f"hssd/{stage_path.stem}",
                        "reason": "unsupported_stage_file",
                        "source_path": _repo_path(stage_path),
                    }
                )
                continue

            scene_id = stage_path.stem
            scene_uid = f"hssd/{scene_id}"
            scene_instance_path = hssd_scene_instance_path(destination_root, scene_id)
            if not scene_instance_path.exists():
                skipped_scenes.append(
                    {
                        "scene_id": scene_id,
                        "scene_uid": scene_uid,
                        "reason": "missing_scene_instance_json",
                        "source_path": _repo_path(stage_path),
                    }
                )
                continue
            scene_instance = json.loads(scene_instance_path.read_text())
            raw_objects = (
                scene_instance.get("object_instances")
                if isinstance(scene_instance.get("object_instances"), list)
                else []
            )
            normalized_objects = [
                object_entry
                for index, payload in enumerate(raw_objects)
                if isinstance(payload, dict)
                for object_entry in [
                    _hssd_object_entry(
                        destination_root=destination_root,
                        scene_id=scene_id,
                        object_index=index,
                        payload=payload,
                        metadata_by_id=metadata_by_id,
                        semantic_labels=semantic_labels,
                    )
                ]
                if object_entry is not None
            ]
            stage_config = hssd_stage_config_path(destination_root, scene_id)
            manifest = _scene_manifest_base(
                dataset="hssd",
                scene_id=scene_id,
                scene_uid=scene_uid,
                subset=None,
                scene_dir=stage_path.parent,
                description=None,
            )
            manifest.update(
                {
                    "status": "ready",
                    "display": {
                        "title": scene_id,
                        "subtitle": "HSSD stage scene",
                        "preview_images": [],
                    },
                    "stats": {
                        "room_count": 1,
                        "object_count": len(normalized_objects),
                        "stage_glb_count": 1,
                    },
                    "assets": {
                        "stage_glb": _repo_path(stage_path),
                        "scene_instance_json": _repo_path(scene_instance_path),
                        "stage_config_json": _repo_path(stage_config) if stage_config.exists() else None,
                    },
                    "normalized": {
                        "kind": "hssd_scene_instance",
                        "rooms": [
                            {
                                "id": "stage_0",
                                "room_type": "stage",
                                "object_count": len(normalized_objects),
                                "object_ids": [obj["id"] for obj in normalized_objects],
                            }
                        ],
                        "objects": normalized_objects,
                    },
                }
            )

            scene_manifest_path = _hssd_scene_output_dir(scene_id) / "scene.json"
            _write_json(scene_manifest_path, manifest)
            scenes.append(
                {
                    "scene_id": scene_id,
                    "scene_uid": scene_uid,
                    "description": None,
                    "title": scene_id,
                    "preview_image": None,
                    "scene_manifest": _repo_path(scene_manifest_path),
                    "stats": manifest["stats"],
                }
            )

    index = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _now_utc(),
        "dataset": "hssd",
        "source_root": _repo_path(source_root),
        "output_root": _repo_path(output_root),
        "scene_count": len(scenes),
        "skipped_count": len(skipped_scenes),
        "scenes": scenes,
        "skipped_scenes": skipped_scenes,
    }
    _write_json(output_root / "index.json", index)
    return index
