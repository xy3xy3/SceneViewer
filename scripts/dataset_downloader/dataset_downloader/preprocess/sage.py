from __future__ import annotations

from .common import *

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

