from __future__ import annotations

import json

from datetime import UTC, datetime
from pathlib import Path

from .config import DATASETS, PREPROCESSED_ROOT, REPO_ROOT


SCHEMA_VERSION = 1
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


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


def preprocess_sage_dataset() -> dict[str, object]:
    source_root = DATASETS["sage"].destination_root / "source" / "extracted"
    output_root = PREPROCESSED_ROOT / "sage"
    scenes: list[dict[str, object]] = []
    skipped_scenes: list[dict[str, object]] = []

    if source_root.exists():
        for scene_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
            layout_files = sorted(scene_dir.glob("layout_*.json"))
            if not layout_files:
                skipped_scenes.append(
                    {
                        "scene_id": scene_dir.name,
                        "scene_uid": f"sage/{scene_dir.name}",
                        "reason": "missing_layout_json",
                        "source_dir": _repo_path(scene_dir),
                    }
                )
                continue

            layout_path = layout_files[0]
            layout = json.loads(layout_path.read_text())
            rooms = layout.get("rooms", [])
            preview_images = _collect_preview_images(scene_dir)

            normalized_rooms: list[dict[str, object]] = []
            normalized_objects: list[dict[str, object]] = []
            wall_count = 0
            door_count = 0
            window_count = 0

            for room in rooms:
                room_id = room.get("id")
                room_objects = room.get("objects", [])
                room_walls = room.get("walls", [])
                room_doors = room.get("doors", [])
                room_windows = room.get("windows", [])
                wall_count += len(room_walls)
                door_count += len(room_doors)
                window_count += len(room_windows)

                normalized_rooms.append(
                    {
                        "id": room_id,
                        "room_type": room.get("room_type"),
                        "position": room.get("position"),
                        "dimensions": room.get("dimensions"),
                        "ceiling_height": room.get("ceiling_height"),
                        "floor_material": room.get("floor_material"),
                        "walls": room_walls,
                        "doors": room_doors,
                        "windows": room_windows,
                        "object_ids": [obj.get("id") for obj in room_objects],
                    }
                )

                for obj in room_objects:
                    normalized_objects.append(
                        _normalize_sage_object(scene_dir, room_id, obj)
                    )

            scene_id = scene_dir.name
            scene_uid = f"sage/{scene_id}"
            manifest = _scene_manifest_base(
                dataset="sage",
                scene_id=scene_id,
                scene_uid=scene_uid,
                subset=None,
                scene_dir=scene_dir,
                description=layout.get("description"),
            )
            manifest.update(
                {
                    "status": "ready",
                    "display": {
                        "title": layout.get("id", scene_id),
                        "subtitle": layout.get("building_style"),
                        "preview_images": preview_images,
                    },
                    "stats": {
                        "room_count": len(normalized_rooms),
                        "object_count": len(normalized_objects),
                        "wall_count": wall_count,
                        "door_count": door_count,
                        "window_count": window_count,
                    },
                    "assets": {
                        "layout_json": _repo_path(layout_path),
                        "preview_images": preview_images,
                        "materials_dir": _repo_path(scene_dir / "materials"),
                        "objects_dir": _repo_path(scene_dir / "objects"),
                    },
                    "normalized": {
                        "kind": "sage_layout",
                        "layout_id": layout.get("id"),
                        "total_area": layout.get("total_area"),
                        "building_style": layout.get("building_style"),
                        "created_from_text": layout.get("created_from_text"),
                        "policy_analysis": layout.get("policy_analysis"),
                        "rooms": normalized_rooms,
                        "objects": normalized_objects,
                    },
                }
            )

            output_dir = _sage_scene_output_dir(scene_id)
            scene_manifest_path = output_dir / "scene.json"
            _write_json(scene_manifest_path, manifest)

            scenes.append(
                {
                    "scene_id": scene_id,
                    "scene_uid": scene_uid,
                    "description": layout.get("description"),
                    "title": layout.get("id", scene_id),
                    "preview_image": preview_images[0] if preview_images else None,
                    "scene_manifest": _repo_path(scene_manifest_path),
                    "stats": manifest["stats"],
                }
            )

    index = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _now_utc(),
        "dataset": "sage",
        "source_root": _repo_path(source_root),
        "output_root": _repo_path(output_root),
        "scene_count": len(scenes),
        "skipped_count": len(skipped_scenes),
        "scenes": scenes,
        "skipped_scenes": skipped_scenes,
    }
    _write_json(output_root / "index.json", index)
    return index


def _resolve_scenesmith_sdf(scene_dir: Path, room_id: str, sdf_path: str | None) -> Path | None:
    if not sdf_path:
        return None
    candidate = Path(sdf_path)
    if candidate.is_absolute():
        return candidate
    return scene_dir / f"room_{room_id}" / candidate


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _normalize_scenesmith_room(
    scene_dir: Path,
    room_id: str,
    room_data: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    room_geometry = room_data.get("room_geometry", {})
    objects = room_data.get("objects", {})
    normalized_objects: list[dict[str, object]] = []

    for object_id, obj_data in objects.items():
        metadata = obj_data.get("metadata", {}) if isinstance(obj_data, dict) else {}
        resolved_sdf = _resolve_scenesmith_sdf(
            scene_dir,
            room_id,
            obj_data.get("sdf_path") if isinstance(obj_data, dict) else None,
        )
        gltf_path = None
        if resolved_sdf and resolved_sdf.exists():
            gltf_candidates = sorted(resolved_sdf.parent.glob("*.gltf"))
            if gltf_candidates:
                gltf_path = gltf_candidates[0]

        normalized_objects.append(
            {
                "id": object_id,
                "room_id": room_id,
                "name": obj_data.get("name") if isinstance(obj_data, dict) else None,
                "description": (
                    obj_data.get("description") if isinstance(obj_data, dict) else None
                ),
                "object_type": (
                    obj_data.get("object_type") if isinstance(obj_data, dict) else None
                )
                or metadata.get("object_type")
                or metadata.get("asset_type"),
                "transform": (
                    obj_data.get("transform") if isinstance(obj_data, dict) else None
                ),
                "bbox_min": (
                    obj_data.get("bbox_min") if isinstance(obj_data, dict) else None
                ),
                "bbox_max": (
                    obj_data.get("bbox_max") if isinstance(obj_data, dict) else None
                ),
                "sdf_path": _repo_path(resolved_sdf) if resolved_sdf else None,
                "gltf_path": _repo_path(gltf_path) if gltf_path else None,
                "metadata": metadata,
            }
        )

    room_manifest = {
        "id": room_id,
        "object_count": len(normalized_objects),
        "dimensions": {
            "length": room_geometry.get("length"),
            "width": room_geometry.get("width"),
            "height": room_geometry.get("height"),
        },
        "walls": room_geometry.get("walls", []),
        "floor": room_geometry.get("floor"),
        "objects": [obj["id"] for obj in normalized_objects],
    }
    return room_manifest, normalized_objects


def preprocess_scenesmith_dataset() -> dict[str, object]:
    source_root = DATASETS["scenesmith"].destination_root / "source" / "extracted"
    output_root = PREPROCESSED_ROOT / "scenesmith"
    scenes: list[dict[str, object]] = []
    skipped_scenes: list[dict[str, object]] = []

    if source_root.exists():
        for subset_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
            subset = subset_dir.name
            for scene_dir in sorted(path for path in subset_dir.iterdir() if path.is_dir()):
                scene_id = scene_dir.name
                scene_uid = f"scenesmith/{subset}/{scene_id}"
                combined_dir = scene_dir / "combined_house"
                house_state_path = combined_dir / "house_state.json"
                dmd_path = combined_dir / "house.dmd.yaml"
                package_xml_path = scene_dir / "package.xml"

                missing = [
                    label
                    for label, path in (
                        ("combined_house/house_state.json", house_state_path),
                        ("combined_house/house.dmd.yaml", dmd_path),
                        ("package.xml", package_xml_path),
                    )
                    if not path.exists()
                ]
                if missing:
                    skipped_scenes.append(
                        {
                            "scene_id": scene_id,
                            "scene_uid": scene_uid,
                            "subset": subset,
                            "reason": "missing_required_files",
                            "missing": missing,
                            "source_dir": _repo_path(scene_dir),
                        }
                    )
                    continue

                house_state = json.loads(house_state_path.read_text())
                raw_rooms = house_state.get("rooms", {})
                normalized_rooms: list[dict[str, object]] = []
                normalized_objects: list[dict[str, object]] = []
                for room_id, room_data in raw_rooms.items():
                    room_manifest, room_objects = _normalize_scenesmith_room(
                        scene_dir,
                        room_id,
                        room_data,
                    )
                    normalized_rooms.append(room_manifest)
                    normalized_objects.extend(room_objects)

                gltf_files = sorted(scene_dir.rglob("*.gltf"))
                obj_files = sorted(scene_dir.rglob("*.obj"))
                preview_images = [
                    _repo_path(path)
                    for path in sorted(scene_dir.rglob("*"))
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                ][:24]

                layout = house_state.get("layout", {})
                placed_rooms = layout.get("placed_rooms", [])
                description = (
                    layout.get("description")
                    or house_state.get("description")
                    or f"{subset} {scene_id}"
                )

                manifest = _scene_manifest_base(
                    dataset="scenesmith",
                    scene_id=scene_id,
                    scene_uid=scene_uid,
                    subset=subset,
                    scene_dir=scene_dir,
                    description=description,
                )
                manifest.update(
                    {
                        "status": "ready",
                        "display": {
                            "title": f"{subset} {scene_id}",
                            "subtitle": layout.get("style") or layout.get("category"),
                            "preview_images": preview_images,
                        },
                        "stats": {
                            "room_count": len(normalized_rooms),
                            "object_count": len(normalized_objects),
                            "placed_room_count": len(placed_rooms),
                            "gltf_count": len(gltf_files),
                            "obj_count": len(obj_files),
                        },
                        "assets": {
                            "package_xml": _repo_path(package_xml_path),
                            "house_state_json": _repo_path(house_state_path),
                            "house_dmd_yaml": _repo_path(dmd_path),
                            "house_blend": (
                                _repo_path(combined_dir / "house.blend")
                                if (combined_dir / "house.blend").exists()
                                else None
                            ),
                            "preview_images": preview_images,
                        },
                        "normalized": {
                            "kind": "scenesmith_house_state",
                            "layout": layout,
                            "rooms": normalized_rooms,
                            "objects": normalized_objects,
                        },
                    }
                )

                scene_output_dir = _scenesmith_scene_output_dir(subset, scene_id)
                scene_manifest_path = scene_output_dir / "scene.json"
                _write_json(scene_manifest_path, manifest)

                scenes.append(
                    {
                        "scene_id": scene_id,
                        "scene_uid": scene_uid,
                        "subset": subset,
                        "description": description,
                        "title": f"{subset} {scene_id}",
                        "preview_image": preview_images[0] if preview_images else None,
                        "scene_manifest": _repo_path(scene_manifest_path),
                        "stats": manifest["stats"],
                    }
                )

    index = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _now_utc(),
        "dataset": "scenesmith",
        "source_root": _repo_path(source_root),
        "output_root": _repo_path(output_root),
        "scene_count": len(scenes),
        "skipped_count": len(skipped_scenes),
        "scenes": scenes,
        "skipped_scenes": skipped_scenes,
    }
    _write_json(output_root / "index.json", index)
    return index


def write_dataset_catalog(indices: list[dict[str, object]]) -> dict[str, object]:
    catalog = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _now_utc(),
        "datasets": [
            {
                "dataset": index["dataset"],
                "scene_count": index["scene_count"],
                "skipped_count": index["skipped_count"],
                "index_path": _repo_path(PREPROCESSED_ROOT / index["dataset"] / "index.json"),
            }
            for index in indices
        ],
    }
    _write_json(PREPROCESSED_ROOT / "datasets.json", catalog)
    return catalog
