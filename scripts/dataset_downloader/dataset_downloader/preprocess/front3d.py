from __future__ import annotations

from .common import *

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

