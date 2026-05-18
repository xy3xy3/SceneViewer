from __future__ import annotations

import json
import math
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import yaml

from .config import DATASETS, PREPROCESSED_ROOT, REPO_ROOT
from .front3d import (
    FRONT3D_DATASET_KEY,
    FRONT3D_LAYOUT_ZIP,
    FRONT3D_MODEL_ZIP,
    FRONT3D_TEXTURE_ZIP,
    ensure_front3d_archives,
    map_front3d_shell_category,
    repo_relative_path,
    safe_front3d_name,
)
from .hsm import (
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


def _hsm_wall_entry(element: dict[str, object]) -> dict[str, object] | None:
    points = element.get("points")
    if not isinstance(points, list) or len(points) < 2:
        return None
    start = points[0]
    end = points[1]
    if not all(isinstance(point, list) and len(point) >= 2 for point in (start, end)):
        return None
    height = float(element.get("height") or 2.5)
    thickness = float(element.get("depth") or 0.1)
    materials = element.get("materials") if isinstance(element.get("materials"), list) else []
    inside_material = materials[0].get("texture") if materials and isinstance(materials[0], dict) else None
    return {
        "id": str(element.get("id") or "wall"),
        "start_point": {"x": float(start[0]), "y": float(start[1])},
        "end_point": {"x": float(end[0]), "y": float(end[1])},
        "height": height,
        "thickness": thickness,
        "material": inside_material,
    }


def _hsm_openings_from_wall(element: dict[str, object], wall: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    holes = element.get("holes")
    if not isinstance(holes, list):
        return [], []

    start = wall["start_point"]
    end = wall["end_point"]
    wall_length = math.hypot(end["x"] - start["x"], end["y"] - start["y"])
    if wall_length <= 1e-8:
        return [], []

    doors: list[dict[str, object]] = []
    windows: list[dict[str, object]] = []
    for hole in holes:
        if not isinstance(hole, dict):
            continue
        box = hole.get("box")
        if not isinstance(box, dict):
            continue
        mins = box.get("min")
        maxs = box.get("max")
        if not (
            isinstance(mins, list)
            and isinstance(maxs, list)
            and len(mins) >= 2
            and len(maxs) >= 2
        ):
            continue
        start_distance = float(mins[0])
        end_distance = float(maxs[0])
        center_distance = (start_distance + end_distance) / 2.0
        position_on_wall = max(0.0, min(1.0, center_distance / wall_length))
        entry = {
            "id": str(hole.get("id") or f"{wall['id']}-opening"),
            "wall_id": wall["id"],
            "position_on_wall": position_on_wall,
            "width": max(0.0, end_distance - start_distance),
            "height": max(0.0, float(maxs[1]) - float(mins[1])),
        }
        if hole.get("type") == "Door":
            doors.append(entry)
        elif hole.get("type") == "Window":
            windows.append(entry)
    return doors, windows


def _hsm_room_dimensions(floor_points: list[list[float]]) -> dict[str, float] | None:
    if not floor_points:
        return None
    xs = [float(point[0]) for point in floor_points if len(point) >= 2]
    ys = [float(point[1]) for point in floor_points if len(point) >= 2]
    if not xs or not ys:
        return None
    return {
        "width": max(xs) - min(xs),
        "length": max(ys) - min(ys),
    }


def preprocess_hsm_dataset(scene_limit: int | None = None) -> dict[str, object]:
    source_root = hsm_generated_scenes_root(DATASETS[HSM_DATASET_KEY].destination_root)
    output_root = PREPROCESSED_ROOT / HSM_DATASET_KEY
    hssd_metadata_by_id = load_hsm_hssd_metadata()
    scenes: list[dict[str, object]] = []
    skipped_scenes: list[dict[str, object]] = []

    if output_root.exists():
        shutil.rmtree(output_root)

    if source_root.exists():
        scene_paths = sorted(source_root.rglob("*.json"))
        for scene_path in scene_paths:
            if scene_limit is not None and len(scenes) >= scene_limit:
                break

            raw_scene = load_hsm_scene(scene_path)
            scene_state = raw_scene.get("scene")
            if not isinstance(scene_state, dict):
                skipped_scenes.append(
                    {
                        "scene_id": scene_path.stem,
                        "scene_uid": f"{HSM_DATASET_KEY}/{scene_path.stem}",
                        "reason": "missing_scene_block",
                        "source_path": _repo_path(scene_path),
                    }
                )
                continue

            arch = scene_state.get("arch")
            arch_elements = arch.get("elements") if isinstance(arch, dict) else []
            if not isinstance(arch_elements, list):
                arch_elements = []

            walls: list[dict[str, object]] = []
            doors: list[dict[str, object]] = []
            windows: list[dict[str, object]] = []
            floor_points: list[list[float]] = []
            room_height = 2.5
            for element in arch_elements:
                if not isinstance(element, dict):
                    continue
                element_type = element.get("type")
                if element_type == "Wall":
                    wall = _hsm_wall_entry(element)
                    if wall is None:
                        continue
                    walls.append(wall)
                    room_height = max(room_height, float(wall["height"]))
                    wall_doors, wall_windows = _hsm_openings_from_wall(element, wall)
                    doors.extend(wall_doors)
                    windows.extend(wall_windows)
                elif element_type == "Floor":
                    points = element.get("points")
                    if isinstance(points, list):
                        floor_points = [point for point in points if isinstance(point, list) and len(point) >= 2]

            room_dimensions = _hsm_room_dimensions(floor_points) or {"width": 4.0, "length": 4.0}
            room_id = "room_0"
            normalized_objects: list[dict[str, object]] = []
            raw_objects = scene_state.get("object")
            if not isinstance(raw_objects, list):
                raw_objects = []

            for raw_object in raw_objects:
                if not isinstance(raw_object, dict):
                    continue
                model_id = hsm_object_model_id(raw_object.get("modelId"))
                if not model_id:
                    continue
                hssd_metadata = hssd_metadata_by_id.get(model_id, {})
                detailed_name = hssd_metadata.get("name") if isinstance(hssd_metadata.get("name"), str) else None
                semantic_label = (
                    hssd_metadata.get("semantic_label")
                    if isinstance(hssd_metadata.get("semantic_label"), str)
                    else None
                )
                annot_path = hsm_downloaded_support_asset(DATASETS[HSM_DATASET_KEY].destination_root, model_id)
                annot_surface_path = hsm_downloaded_support_asset(
                    DATASETS[HSM_DATASET_KEY].destination_root,
                    model_id,
                    surface_only=True,
                )
                normalized_objects.append(
                    {
                        "id": str(raw_object.get("id") or model_id),
                        "room_id": room_id,
                        "type": semantic_label or "hssd_object",
                        "name": detailed_name or model_id,
                        "description": detailed_name or semantic_label or model_id,
                        "object_type": semantic_label or "hssd_object",
                        "position": hsm_position_from_transform(raw_object.get("transform")),
                        "rotation": {
                            "z": hsm_yaw_deg_from_transform(raw_object.get("transform")),
                        },
                        "scale": hsm_scale_from_transform(raw_object.get("transform")),
                        "source_id": model_id,
                        "metadata": {
                            "model_id": model_id,
                            "raw_model_id": raw_object.get("modelId"),
                            "hssd_name": detailed_name,
                            "hssd_wnsynsetkey": hssd_metadata.get("wnsynsetkey"),
                            "hssd_semantic_label": semantic_label,
                            "hssd_up": hssd_metadata.get("up"),
                            "hssd_front": hssd_metadata.get("front"),
                            "hssd_support_region": hssd_metadata.get("support_region"),
                            "transform": raw_object.get("transform"),
                            "support_region_asset": _repo_path(annot_path) if annot_path else None,
                            "support_region_surface_asset": (
                                _repo_path(annot_surface_path) if annot_surface_path else None
                            ),
                        },
                    }
                )

            scene_id = hsm_scene_id_from_remote_path(scene_path.name)
            scene_uid = f"{HSM_DATASET_KEY}/{scene_id}"
            manifest = _scene_manifest_base(
                dataset=HSM_DATASET_KEY,
                scene_id=scene_id,
                scene_uid=scene_uid,
                subset=None,
                scene_dir=scene_path.parent,
                description=f"HSM generated scene {scene_id}",
            )
            manifest["source"] = {
                "scene_json": _repo_path(scene_path),
            }
            manifest.update(
                {
                    "status": "ready",
                    "display": {
                        "title": scene_id,
                        "subtitle": "HSM generated scene",
                        "preview_images": [],
                    },
                    "stats": {
                        "room_count": 1,
                        "object_count": len(normalized_objects),
                        "wall_count": len(walls),
                        "door_count": len(doors),
                        "window_count": len(windows),
                    },
                    "assets": {
                        "scene_json": _repo_path(scene_path),
                    },
                    "normalized": {
                        "kind": "hsm_scene_state",
                        "format": raw_scene.get("format"),
                        "rooms": [
                            {
                                "id": room_id,
                                "room_type": "generated_room",
                                "dimensions": {
                                    **room_dimensions,
                                    "height": room_height,
                                },
                                "ceiling_height": room_height,
                                "walls": walls,
                                "doors": doors,
                                "windows": windows,
                                "object_ids": [obj["id"] for obj in normalized_objects],
                            }
                        ],
                        "objects": normalized_objects,
                    },
                }
            )

            output_dir = PREPROCESSED_ROOT / HSM_DATASET_KEY / scene_id
            scene_manifest_path = output_dir / "scene.json"
            _write_json(scene_manifest_path, manifest)
            scenes.append(
                {
                    "scene_id": scene_id,
                    "scene_uid": scene_uid,
                    "description": manifest["description"],
                    "title": scene_id,
                    "preview_image": None,
                    "scene_manifest": _repo_path(scene_manifest_path),
                    "stats": manifest["stats"],
                }
            )
    index = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _now_utc(),
        "dataset": HSM_DATASET_KEY,
        "source_root": _repo_path(source_root),
        "output_root": _repo_path(output_root),
        "scene_count": len(scenes),
        "skipped_count": len(skipped_scenes),
        "scenes": scenes,
        "skipped_scenes": skipped_scenes,
    }
    _write_json(output_root / "index.json", index)
    return index


def preprocess_sage_dataset(scene_limit: int | None = None) -> dict[str, object]:
    source_root = DATASETS["sage"].destination_root / "source" / "extracted"
    output_root = PREPROCESSED_ROOT / "sage"
    scenes: list[dict[str, object]] = []
    skipped_scenes: list[dict[str, object]] = []

    if output_root.exists():
        shutil.rmtree(output_root)

    if source_root.exists():
        for scene_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
            if scene_limit is not None and len(scenes) >= scene_limit:
                break
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

            if scene_limit is not None and len(scenes) >= scene_limit:
                break

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


def preprocess_3dfront_dataset(scene_limit: int | None = None) -> dict[str, object]:
    ensure_front3d_archives()

    output_root = PREPROCESSED_ROOT / FRONT3D_DATASET_KEY
    scenes: list[dict[str, object]] = []
    skipped_scenes: list[dict[str, object]] = []

    if output_root.exists():
        shutil.rmtree(output_root)

    with zipfile.ZipFile(FRONT3D_LAYOUT_ZIP) as archive:
        layout_members = sorted(
            name
            for name in archive.namelist()
            if name.startswith("3D-FRONT/") and name.endswith(".json")
        )

        stop_requested = False
        for member_name in layout_members:
            if stop_requested:
                break

            layout = json.loads(archive.read(member_name))
            house_id = str(layout.get("uid") or Path(member_name).stem)
            furniture_by_uid = {
                item.get("uid"): item
                for item in layout.get("furniture", [])
                if isinstance(item, dict) and item.get("uid")
            }
            mesh_by_uid = {
                item.get("uid"): item
                for item in layout.get("mesh", [])
                if isinstance(item, dict) and item.get("uid")
            }
            snapshots = (layout.get("extension") or {}).get("snapshots") or []

            for room in (layout.get("scene") or {}).get("room", []):
                if scene_limit is not None and len(scenes) >= scene_limit:
                    stop_requested = True
                    break

                if not isinstance(room, dict):
                    continue

                room_id = str(room.get("instanceid") or room.get("type") or f"room-{len(scenes)}")
                room_children = [
                    child
                    for child in room.get("children", [])
                    if isinstance(child, dict)
                ]
                room_position, room_dimensions = _front3d_room_bounds(room_children, mesh_by_uid)
                room_shells: list[dict[str, object]] = []
                room_objects: list[dict[str, object]] = []

                for child in room_children:
                    ref = child.get("ref")
                    if not isinstance(ref, str):
                        continue

                    mesh = mesh_by_uid.get(ref)
                    if mesh:
                        room_shells.append(_front3d_shell_ref(child, mesh))
                        continue

                    furniture = furniture_by_uid.get(ref)
                    if furniture:
                        room_objects.append(_front3d_object_entry(room_id, child, furniture))
                        continue

                if not room_shells and not room_objects:
                    skipped_scenes.append(
                        {
                            "scene_id": room_id,
                            "scene_uid": f"{FRONT3D_DATASET_KEY}/{house_id}/{room_id}",
                            "reason": "room_has_no_supported_children",
                            "source_layout_entry": member_name,
                        }
                    )
                    continue

                scene_uid = f"{FRONT3D_DATASET_KEY}/{house_id}/{room_id}"
                manifest = _scene_manifest_base(
                    dataset=FRONT3D_DATASET_KEY,
                    scene_id=room_id,
                    scene_uid=scene_uid,
                    subset=house_id,
                    scene_dir=FRONT3D_LAYOUT_ZIP.parent,
                    description=room.get("type"),
                )
                manifest["source"] = {
                    "layout_zip": _repo_path(FRONT3D_LAYOUT_ZIP),
                    "layout_entry": member_name,
                    "model_zip": _repo_path(FRONT3D_MODEL_ZIP),
                    "texture_zip": _repo_path(FRONT3D_TEXTURE_ZIP),
                }
                manifest.update(
                    {
                        "status": "ready",
                        "display": {
                            "title": room_id,
                            "subtitle": room.get("type"),
                            "preview_images": [],
                        },
                        "stats": {
                            "room_count": 1,
                            "object_count": len(room_objects),
                            "shell_count": len(room_shells),
                            "renderable_object_count": sum(
                                1
                                for item in room_objects
                                if (item.get("metadata") or {}).get("valid", False)
                            ),
                            "snapshot_count": len(snapshots),
                        },
                        "assets": {
                            "layout_zip": _repo_path(FRONT3D_LAYOUT_ZIP),
                            "layout_entry": member_name,
                            "model_zip": _repo_path(FRONT3D_MODEL_ZIP),
                            "texture_zip": _repo_path(FRONT3D_TEXTURE_ZIP),
                        },
                        "normalized": {
                            "kind": "3dfront_room",
                            "house_id": house_id,
                            "house_layout_entry": member_name,
                            "north_vector": layout.get("north_vector"),
                            "snapshots": snapshots,
                            "rooms": [
                                {
                                    "id": room_id,
                                    "room_type": room.get("type"),
                                    "position": room_position,
                                    "dimensions": room_dimensions,
                                    "object_ids": [obj.get("id") for obj in room_objects],
                                    "object_count": len(room_objects),
                                    "shell_refs": room_shells,
                                }
                            ],
                            "objects": room_objects,
                        },
                    }
                )

                output_dir = _front3d_scene_output_dir(house_id, room_id)
                scene_manifest_path = output_dir / "scene.json"
                _write_json(scene_manifest_path, manifest)
                scenes.append(
                    {
                        "scene_id": room_id,
                        "scene_uid": scene_uid,
                        "subset": house_id,
                        "description": room.get("type"),
                        "title": room_id,
                        "preview_image": None,
                        "scene_manifest": _repo_path(scene_manifest_path),
                        "stats": manifest["stats"],
                    }
                )

    index = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _now_utc(),
        "dataset": FRONT3D_DATASET_KEY,
        "source_root": _repo_path(FRONT3D_LAYOUT_ZIP),
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
    if candidate.exists():
        return candidate
    repo_candidate = REPO_ROOT / candidate
    if repo_candidate.exists():
        return repo_candidate
    if candidate.parts and candidate.parts[0].startswith("room_"):
        return scene_dir / candidate
    return scene_dir / f"room_{room_id}" / candidate


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


class _DmdLoader(yaml.SafeLoader):
    pass


def _construct_unknown_yaml_tag(
    loader: _DmdLoader,
    tag_suffix: str,
    node: yaml.Node,
) -> object:
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


_DmdLoader.add_multi_constructor("!", _construct_unknown_yaml_tag)


def _load_dmd_yaml(path: Path) -> dict[str, object]:
    return yaml.load(path.read_text(), Loader=_DmdLoader)


def _extract_pose_data(pose_block: object) -> dict[str, object] | None:
    if not isinstance(pose_block, dict) or not pose_block:
        return None
    first_value = next(iter(pose_block.values()))
    if not isinstance(first_value, dict):
        return None
    return {
        "translation": first_value.get("translation"),
        "rotation": first_value.get("rotation"),
        "base_frame": first_value.get("base_frame"),
    }


def _extract_angle_axis(rotation: object) -> dict[str, object] | None:
    if not isinstance(rotation, dict):
        return None
    return {
        "angle_deg": rotation.get("angle_deg"),
        "axis": rotation.get("axis"),
    }


def _package_scene_uri_to_path(scene_dir: Path, uri: str | None) -> Path | None:
    if not uri:
        return None
    prefix = "package://scene/"
    if uri.startswith(prefix):
        return scene_dir / uri[len(prefix) :]
    candidate = Path(uri)
    if candidate.is_absolute():
        return candidate
    return scene_dir / candidate


def _scene_room_dirs(scene_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in scene_dir.iterdir()
        if path.is_dir() and path.name.startswith("room_") and path.name != "room_geometry"
    )


def _scan_room_generated_assets(room_dir: Path) -> dict[str, list[str]]:
    generated_root = room_dir / "generated_assets"
    result: dict[str, list[str]] = {}
    if not generated_root.exists():
        return result

    for category_dir in sorted(path for path in generated_root.iterdir() if path.is_dir()):
        gltfs = sorted(category_dir.rglob("*.gltf"))
        if gltfs:
            result[category_dir.name] = [_repo_path(path) for path in gltfs]
    return result


def _scan_floor_plan_assets(scene_dir: Path, room_id: str) -> dict[str, object]:
    floor_root = scene_dir / "floor_plans" / room_id
    floor_gltf = _first_existing([floor_root / "floors" / "floor.gltf"])
    wall_gltfs = sorted((floor_root / "walls").rglob("wall.gltf")) if (floor_root / "walls").exists() else []
    window_gltfs = sorted((floor_root / "windows").rglob("window.gltf")) if (floor_root / "windows").exists() else []
    return {
        "floor_gltf": _repo_path(floor_gltf) if floor_gltf else None,
        "wall_gltfs": [_repo_path(path) for path in wall_gltfs],
        "window_gltfs": [_repo_path(path) for path in window_gltfs],
    }


def _object_type_from_package_path(path: str | None) -> str | None:
    if not path:
        return None
    parts = Path(path).parts
    for marker in (
        "furniture",
        "manipulands",
        "manipuland",
        "wall_mounted",
        "ceiling_mounted",
    ):
        if marker in parts:
            return marker
    if "room_geometry" in parts:
        return "room_geometry"
    return None


def _room_id_from_room_dir(room_dir_name: str) -> str:
    return room_dir_name[len("room_") :]


def _infer_room_id_from_model_name(model_name: str, room_ids: list[str]) -> str | None:
    for room_id in sorted(room_ids, key=len, reverse=True):
        if model_name == room_id or model_name.startswith(f"{room_id}_"):
            return room_id
    return None


def _build_scenesmith_fallback_state(
    scene_dir: Path,
    dmd_path: Path,
) -> dict[str, object]:
    dmd = _load_dmd_yaml(dmd_path)
    directives = dmd.get("directives", [])
    room_dirs = _scene_room_dirs(scene_dir)
    room_ids = [_room_id_from_room_dir(path.name) for path in room_dirs]

    room_frames: dict[str, dict[str, object]] = {}
    models: dict[str, dict[str, object]] = {}
    welds: dict[str, dict[str, object]] = {}

    for directive in directives:
        if not isinstance(directive, dict):
            continue
        if "add_frame" in directive:
            add_frame = directive["add_frame"]
            if not isinstance(add_frame, dict):
                continue
            frame_name = add_frame.get("name")
            x_pf = add_frame.get("X_PF", {})
            if (
                isinstance(frame_name, str)
                and frame_name.startswith("room_")
                and frame_name.endswith("_frame")
            ):
                room_id = frame_name[len("room_") : -len("_frame")]
                room_frames[room_id] = {
                    "frame_name": frame_name,
                    "translation": x_pf.get("translation"),
                    "base_frame": x_pf.get("base_frame"),
                }
        elif "add_model" in directive:
            add_model = directive["add_model"]
            if not isinstance(add_model, dict):
                continue
            model_name = add_model.get("name")
            if not isinstance(model_name, str):
                continue
            models[model_name] = {
                "name": model_name,
                "file": add_model.get("file"),
                "pose": _extract_pose_data(add_model.get("default_free_body_pose")),
            }
        elif "add_weld" in directive:
            add_weld = directive["add_weld"]
            if not isinstance(add_weld, dict):
                continue
            child = add_weld.get("child")
            if not isinstance(child, str):
                continue
            model_name = child.split("::", 1)[0]
            welds[model_name] = {
                "parent": add_weld.get("parent"),
                "pose": {
                    "translation": add_weld.get("X_PC", {}).get("translation")
                    if isinstance(add_weld.get("X_PC"), dict)
                    else None,
                    "rotation": add_weld.get("X_PC", {}).get("rotation")
                    if isinstance(add_weld.get("X_PC"), dict)
                    else None,
                    "base_frame": add_weld.get("parent"),
                },
            }

    rooms: dict[str, dict[str, object]] = {}
    for room_dir in room_dirs:
        room_id = _room_id_from_room_dir(room_dir.name)
        floor_assets = _scan_floor_plan_assets(scene_dir, room_id)
        generated_assets = _scan_room_generated_assets(room_dir)
        room_geometry_sdf = scene_dir / "room_geometry" / f"room_geometry_{room_id}.sdf"
        rooms[room_id] = {
            "room_id": room_id,
            "frame": room_frames.get(room_id, {}),
            "room_geometry_sdf": _repo_path(room_geometry_sdf)
            if room_geometry_sdf.exists()
            else None,
            "floor_plan_assets": floor_assets,
            "generated_assets": generated_assets,
            "objects": {},
        }

    for model_name, model in models.items():
        file_uri = model.get("file")
        if not isinstance(file_uri, str):
            continue
        resolved_path = _package_scene_uri_to_path(scene_dir, file_uri)
        if model_name.startswith("room_geometry_"):
            continue

        room_id = None
        if isinstance(resolved_path, Path):
            parts = resolved_path.parts
            for part in parts:
                if part.startswith("room_"):
                    room_id = _room_id_from_room_dir(part)
                    break
        if room_id is None:
            room_id = _infer_room_id_from_model_name(model_name, room_ids)
        if room_id is None:
            continue

        pose = model.get("pose") or welds.get(model_name, {}).get("pose")
        sdf_path = resolved_path if resolved_path and resolved_path.exists() else None
        gltf_path = None
        if sdf_path is not None:
            candidates = sorted(sdf_path.parent.glob("*.gltf"))
            if candidates:
                gltf_path = candidates[0]

        rooms.setdefault(
            room_id,
            {
                "room_id": room_id,
                "frame": room_frames.get(room_id, {}),
                "room_geometry_sdf": None,
                "floor_plan_assets": _scan_floor_plan_assets(scene_dir, room_id),
                "generated_assets": {},
                "objects": {},
            },
        )
        rooms[room_id]["objects"][model_name] = {
            "object_id": model_name,
            "name": model_name,
            "description": None,
            "object_type": _object_type_from_package_path(file_uri),
            "transform": {
                "translation": pose.get("translation") if isinstance(pose, dict) else None,
                "rotation_angle_axis": _extract_angle_axis(
                    pose.get("rotation") if isinstance(pose, dict) else None
                ),
                "base_frame": pose.get("base_frame") if isinstance(pose, dict) else None,
            },
            "sdf_path": _repo_path(sdf_path) if sdf_path else None,
            "gltf_path": _repo_path(gltf_path) if gltf_path else None,
            "metadata": {
                "source_format": "dmd_fallback",
                "model_file_uri": file_uri,
                "welded": model_name in welds,
            },
        }

    return {
        "layout": {
            "source_format": "dmd_fallback",
            "room_count": len(rooms),
        },
        "rooms": rooms,
        "_source": "dmd_fallback",
    }


def _normalize_scenesmith_room(
    scene_dir: Path,
    room_id: str,
    room_data: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    room_geometry = room_data.get("room_geometry", {})
    objects = room_data.get("objects", {})
    normalized_objects: list[dict[str, object]] = []
    room_dir = scene_dir / f"room_{room_id}"
    scanned_floor_plan_assets = _scan_floor_plan_assets(scene_dir, room_id)
    scanned_generated_assets = _scan_room_generated_assets(room_dir)
    raw_room_geometry_sdf = room_data.get("room_geometry_sdf")
    resolved_room_geometry_sdf = _resolve_scenesmith_sdf(scene_dir, room_id, raw_room_geometry_sdf)
    if resolved_room_geometry_sdf is None:
        fallback_room_geometry_sdf = scene_dir / "room_geometry" / f"room_geometry_{room_id}.sdf"
        if fallback_room_geometry_sdf.exists():
            resolved_room_geometry_sdf = fallback_room_geometry_sdf

    for object_id, obj_data in objects.items():
        metadata = obj_data.get("metadata", {}) if isinstance(obj_data, dict) else {}
        raw_sdf_path = obj_data.get("sdf_path") if isinstance(obj_data, dict) else None
        raw_gltf_path = obj_data.get("gltf_path") if isinstance(obj_data, dict) else None
        raw_geometry_path = obj_data.get("geometry_path") if isinstance(obj_data, dict) else None
        resolved_sdf = _resolve_scenesmith_sdf(scene_dir, room_id, raw_sdf_path)
        resolved_gltf = _resolve_scenesmith_sdf(
            scene_dir,
            room_id,
            raw_gltf_path or raw_geometry_path,
        )
        if resolved_gltf is None and resolved_sdf and resolved_sdf.exists():
            gltf_candidates = sorted(resolved_sdf.parent.glob("*.gltf"))
            if gltf_candidates:
                resolved_gltf = gltf_candidates[0]

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
                "gltf_path": _repo_path(resolved_gltf) if resolved_gltf else None,
                "metadata": metadata,
            }
        )

    room_manifest = {
        "id": room_id,
        "object_count": len(normalized_objects),
        "dimensions": {
            "length": room_geometry.get("length"),
            "width": room_geometry.get("width"),
            "height": room_geometry.get("height") or room_geometry.get("wall_height"),
        },
        "frame": room_data.get("frame"),
        "room_geometry_sdf": (
            _repo_path(resolved_room_geometry_sdf) if resolved_room_geometry_sdf else None
        ),
        "floor_plan_assets": room_data.get("floor_plan_assets") or scanned_floor_plan_assets,
        "generated_assets": room_data.get("generated_assets") or scanned_generated_assets,
        "walls": room_geometry.get("walls", []),
        "floor": room_geometry.get("floor"),
        "objects": [obj["id"] for obj in normalized_objects],
    }
    return room_manifest, normalized_objects


def preprocess_scenesmith_dataset(scene_limit: int | None = None) -> dict[str, object]:
    source_root = DATASETS["scenesmith"].destination_root / "source" / "extracted"
    output_root = PREPROCESSED_ROOT / "scenesmith"
    scenes: list[dict[str, object]] = []
    skipped_scenes: list[dict[str, object]] = []

    if output_root.exists():
        shutil.rmtree(output_root)

    if source_root.exists():
        for subset_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
            subset = subset_dir.name
            for scene_dir in sorted(path for path in subset_dir.iterdir() if path.is_dir()):
                if scene_limit is not None and len(scenes) >= scene_limit:
                    break
                scene_id = scene_dir.name
                scene_uid = f"scenesmith/{subset}/{scene_id}"
                combined_dir = scene_dir / "combined_house"
                house_state_path = combined_dir / "house_state.json"
                dmd_path = combined_dir / "house.dmd.yaml"
                package_xml_path = scene_dir / "package.xml"
                mujoco_scene_path = scene_dir / "mujoco" / "scene.xml"

                missing = [
                    label
                    for label, path in (
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

                scene_state_mode = "house_state_json" if house_state_path.exists() else "dmd_fallback"
                if house_state_path.exists():
                    house_state = json.loads(house_state_path.read_text())
                else:
                    house_state = _build_scenesmith_fallback_state(scene_dir, dmd_path)
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
                            "house_state_json": (
                                _repo_path(house_state_path)
                                if house_state_path.exists()
                                else None
                            ),
                            "house_dmd_yaml": _repo_path(dmd_path),
                            "house_blend": (
                                _repo_path(combined_dir / "house.blend")
                                if (combined_dir / "house.blend").exists()
                                else None
                            ),
                            "mujoco_scene_xml": (
                                _repo_path(mujoco_scene_path)
                                if mujoco_scene_path.exists()
                                else None
                            ),
                            "preview_images": preview_images,
                        },
                        "normalized": {
                            "kind": "scenesmith_house_state"
                            if scene_state_mode == "house_state_json"
                            else "scenesmith_dmd_fallback",
                            "scene_state_mode": scene_state_mode,
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

            if scene_limit is not None and len(scenes) >= scene_limit:
                break

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


def write_dataset_catalog() -> dict[str, object]:
    datasets: list[dict[str, object]] = []
    for dataset in sorted(DATASETS):
        index_path = PREPROCESSED_ROOT / dataset / "index.json"
        if not index_path.exists():
            continue
        index = json.loads(index_path.read_text())
        datasets.append(
            {
                "dataset": index["dataset"],
                "scene_count": index["scene_count"],
                "skipped_count": index["skipped_count"],
                "index_path": _repo_path(index_path),
            }
        )

    catalog = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _now_utc(),
        "datasets": datasets,
    }
    _write_json(PREPROCESSED_ROOT / "datasets.json", catalog)
    return catalog
