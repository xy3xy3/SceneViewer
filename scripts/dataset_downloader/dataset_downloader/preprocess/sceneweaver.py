from __future__ import annotations

from .common import *


def _sceneweaver_latest_iteration(run_dir: Path) -> int | None:
    args_path = run_dir / "args.json"
    if args_path.exists():
        args_data = json.loads(args_path.read_text())
        if isinstance(args_data, dict):
            iteration = args_data.get("iter")
            if isinstance(iteration, int):
                return iteration

    iterations: list[int] = []
    for path in (run_dir / "record_scene").glob("layout_*.json"):
        match = re.fullmatch(r"layout_(\d+)\.json", path.name)
        if match:
            iterations.append(int(match.group(1)))
    return max(iterations) if iterations else None


def _sceneweaver_title(scene_id: str) -> str:
    parts = [part for part in re.split(r"[_-]+", scene_id) if part]
    return " ".join(part.capitalize() for part in parts) or scene_id


def _sceneweaver_preview_images(run_dir: Path, iteration: int) -> list[str]:
    candidates = [
        run_dir / "record_scene" / f"render_{iteration}_perspective.jpg",
        run_dir / "record_scene" / f"render_{iteration}.jpg",
        run_dir / "record_scene" / f"render_{iteration}_marked.jpg",
        run_dir / "record_scene" / f"render_{iteration}_bbox.png",
    ]
    return [_repo_path(path) for path in candidates if path.exists()]


def _sceneweaver_room_height(
    room_size: list[float] | None,
    objects: dict[str, object],
) -> float:
    max_height = 2.8
    for payload in objects.values():
        if not isinstance(payload, dict):
            continue
        location = payload.get("location")
        size = payload.get("size")
        if (
            isinstance(location, list)
            and len(location) >= 3
            and isinstance(size, list)
            and len(size) >= 3
        ):
            max_height = max(max_height, float(location[2]) + float(size[2]) + 0.4)
    return max_height


def _sceneweaver_normalized_object(
    *,
    object_id: str,
    room_id: str,
    payload: dict[str, object],
) -> dict[str, object]:
    location = payload.get("location") if isinstance(payload.get("location"), list) else []
    rotation = payload.get("rotation") if isinstance(payload.get("rotation"), list) else []
    size = payload.get("size") if isinstance(payload.get("size"), list) else []
    seed = None
    factory_name = object_id
    if "_" in object_id:
        prefix, suffix = object_id.split("_", 1)
        if prefix.isdigit():
            seed = int(prefix)
            factory_name = suffix

    parent_specs = []
    raw_parent = payload.get("parent")
    if isinstance(raw_parent, list):
        for relation in raw_parent:
            if isinstance(relation, list) and len(relation) >= 2:
                parent_specs.append({"target": relation[0], "relation": relation[1]})

    return {
        "id": object_id,
        "room_id": room_id,
        "type": factory_name.removesuffix("Factory"),
        "name": factory_name,
        "description": factory_name.removesuffix("Factory"),
        "position": {
            "x": float(location[0]) if len(location) >= 1 else 0.0,
            "y": float(location[2]) if len(location) >= 3 else 0.0,
            "z": -float(location[1]) if len(location) >= 2 else 0.0,
        },
        "rotation": {
            "z": float(rotation[2]) if len(rotation) >= 3 else 0.0,
        },
        "dimensions": {
            "width": float(size[0]) if len(size) >= 1 else None,
            "length": float(size[1]) if len(size) >= 2 else None,
            "height": float(size[2]) if len(size) >= 3 else None,
        },
        "metadata": {
            "factory_name": factory_name,
            "seed": seed,
            "source_location": location[:3],
            "source_rotation": rotation[:3],
            "source_size": size[:3],
            "parent_specs": parent_specs,
        },
    }


def preprocess_sceneweaver_dataset(scene_limit: int | None = None) -> dict[str, object]:
    source_root = DATASETS["sceneweaver"].destination_root / "source" / "extracted"
    output_root = PREPROCESSED_ROOT / "sceneweaver"
    scenes: list[dict[str, object]] = []
    skipped_scenes: list[dict[str, object]] = []

    if output_root.exists():
        shutil.rmtree(output_root)

    if source_root.exists():
        for subset_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
            subset = subset_dir.name
            for run_dir in sorted(path for path in subset_dir.iterdir() if path.is_dir()):
                if scene_limit is not None and len(scenes) >= scene_limit:
                    break

                scene_id = run_dir.name
                scene_uid = f"sceneweaver/{subset}/{scene_id}"
                iteration = _sceneweaver_latest_iteration(run_dir)
                if iteration is None:
                    skipped_scenes.append(
                        {
                            "scene_id": scene_id,
                            "scene_uid": scene_uid,
                            "subset": subset,
                            "reason": "missing_iteration_files",
                            "source_dir": _repo_path(run_dir),
                        }
                    )
                    continue

                layout_path = run_dir / "record_scene" / f"layout_{iteration}.json"
                blend_path = run_dir / "record_files" / f"scene_{iteration}.blend"
                roominfo_path = run_dir / "roominfo.json"
                args_path = run_dir / "args.json"
                missing = [
                    label
                    for label, path in (
                        (f"record_scene/layout_{iteration}.json", layout_path),
                        (f"record_files/scene_{iteration}.blend", blend_path),
                        ("roominfo.json", roominfo_path),
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
                            "source_dir": _repo_path(run_dir),
                        }
                    )
                    continue

                layout_data = json.loads(layout_path.read_text())
                roominfo = json.loads(roominfo_path.read_text())
                if not isinstance(layout_data, dict) or not isinstance(roominfo, dict):
                    skipped_scenes.append(
                        {
                            "scene_id": scene_id,
                            "scene_uid": scene_uid,
                            "subset": subset,
                            "reason": "invalid_json_payload",
                            "source_dir": _repo_path(run_dir),
                        }
                    )
                    continue

                room_size = layout_data.get("roomsize")
                if not isinstance(room_size, list) or len(room_size) < 2:
                    room_size = roominfo.get("roomsize")
                width = float(room_size[0]) if isinstance(room_size, list) and len(room_size) >= 1 else 4.0
                length = float(room_size[1]) if isinstance(room_size, list) and len(room_size) >= 2 else 4.0

                raw_objects = layout_data.get("objects", {})
                if not isinstance(raw_objects, dict):
                    raw_objects = {}
                room_id = "room_0"
                normalized_objects = [
                    _sceneweaver_normalized_object(
                        object_id=object_id,
                        room_id=room_id,
                        payload=payload,
                    )
                    for object_id, payload in raw_objects.items()
                    if isinstance(payload, dict)
                ]
                room_height = _sceneweaver_room_height(room_size if isinstance(room_size, list) else None, raw_objects)
                normalized_rooms = [
                    {
                        "id": room_id,
                        "room_type": roominfo.get("roomtype"),
                        "object_count": len(normalized_objects),
                        "object_ids": [obj["id"] for obj in normalized_objects],
                        "dimensions": {
                            "width": width,
                            "length": length,
                            "height": room_height,
                        },
                    }
                ]
                preview_images = _sceneweaver_preview_images(run_dir, iteration)
                description = roominfo.get("ideas") if isinstance(roominfo.get("ideas"), str) else None

                metric_path = run_dir / "pipeline" / f"metric_{iteration}.json"
                final_metric = json.loads(metric_path.read_text()) if metric_path.exists() else None

                manifest = _scene_manifest_base(
                    dataset="sceneweaver",
                    scene_id=scene_id,
                    scene_uid=scene_uid,
                    subset=subset,
                    scene_dir=run_dir,
                    description=description,
                )
                manifest.update(
                    {
                        "status": "ready",
                        "display": {
                            "title": _sceneweaver_title(scene_id),
                            "subtitle": roominfo.get("roomtype"),
                            "preview_images": preview_images,
                        },
                        "stats": {
                            "object_count": len(normalized_objects),
                            "room_count": len(normalized_rooms),
                            "latest_iteration": iteration,
                            "preview_image_count": len(preview_images),
                        },
                        "assets": {
                            "run_args": _repo_path(args_path) if args_path.exists() else None,
                            "roominfo": _repo_path(roominfo_path),
                            "layout_json": _repo_path(layout_path),
                            "blend_file": _repo_path(blend_path),
                            "metric_json": _repo_path(metric_path) if metric_path.exists() else None,
                            "preview_images": preview_images,
                        },
                        "normalized": {
                            "kind": "sceneweaver_layout",
                            "created_from_text": description,
                            "layout": layout_data,
                            "rooms": normalized_rooms,
                            "objects": normalized_objects,
                            "snapshots": [final_metric] if isinstance(final_metric, dict) else [],
                        },
                    }
                )

                scene_output_dir = _sceneweaver_scene_output_dir(subset, scene_id)
                scene_manifest_path = scene_output_dir / "scene.json"
                _write_json(scene_manifest_path, manifest)

                scenes.append(
                    {
                        "scene_id": scene_id,
                        "scene_uid": scene_uid,
                        "subset": subset,
                        "description": description,
                        "title": _sceneweaver_title(scene_id),
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
        "dataset": "sceneweaver",
        "source_root": _repo_path(source_root),
        "output_root": _repo_path(output_root),
        "scene_count": len(scenes),
        "skipped_count": len(skipped_scenes),
        "scenes": scenes,
        "skipped_scenes": skipped_scenes,
    }
    _write_json(output_root / "index.json", index)
    return index
