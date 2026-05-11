from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import yaml

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

    if output_root.exists():
        shutil.rmtree(output_root)

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

    for object_id, obj_data in objects.items():
        metadata = obj_data.get("metadata", {}) if isinstance(obj_data, dict) else {}
        raw_sdf_path = obj_data.get("sdf_path") if isinstance(obj_data, dict) else None
        raw_gltf_path = obj_data.get("gltf_path") if isinstance(obj_data, dict) else None
        resolved_sdf = _resolve_scenesmith_sdf(scene_dir, room_id, raw_sdf_path)
        resolved_gltf = _resolve_scenesmith_sdf(scene_dir, room_id, raw_gltf_path)
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
            "height": room_geometry.get("height"),
        },
        "frame": room_data.get("frame"),
        "room_geometry_sdf": room_data.get("room_geometry_sdf"),
        "floor_plan_assets": room_data.get("floor_plan_assets"),
        "generated_assets": room_data.get("generated_assets"),
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

    if output_root.exists():
        shutil.rmtree(output_root)

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
