from __future__ import annotations

from ..hsm import load_sceneval_annotations

from .common import *

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
    sceneval_annotations = load_sceneval_annotations()
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
            scene_id = hsm_scene_id_from_remote_path(scene_path.name)
            asset_annotations = _load_asset_annotation_index(
                scene_path.parent / scene_id / "asset_annotations",
                DATASETS[HSM_DATASET_KEY].destination_root
                / "source"
                / "benchmark_annotations"
                / scene_id
                / "asset_annotations",
            )
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
                object_id = str(raw_object.get("id") or model_id)
                hssd_metadata = hssd_metadata_by_id.get(model_id, {})
                detailed_name = hssd_metadata.get("name") if isinstance(hssd_metadata.get("name"), str) else None
                semantic_label = (
                    hssd_metadata.get("semantic_label")
                    if isinstance(hssd_metadata.get("semantic_label"), str)
                    else None
                )
                annotation_fields = _asset_annotation_label_fields(
                    asset_annotations.get(object_id)
                )
                preferred_label = annotation_fields.get("preferred_label")
                category_label = annotation_fields.get("category_label")
                annot_path = hsm_downloaded_support_asset(DATASETS[HSM_DATASET_KEY].destination_root, model_id)
                annot_surface_path = hsm_downloaded_support_asset(
                    DATASETS[HSM_DATASET_KEY].destination_root,
                    model_id,
                    surface_only=True,
                )
                normalized_objects.append(
                    {
                        "id": object_id,
                        "room_id": room_id,
                        "type": category_label or semantic_label or "hssd_object",
                        "name": preferred_label or detailed_name or model_id,
                        "description": preferred_label or detailed_name or semantic_label or model_id,
                        "object_type": category_label or semantic_label or "hssd_object",
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
                            "vlm_canonical_name": annotation_fields.get("canonical_name"),
                            "vlm_category_norm": annotation_fields.get("category_norm"),
                            "vlm_benchmark_relevance": annotation_fields.get("benchmark_relevance"),
                            "vlm_annotation_source": annotation_fields.get("annotation_source"),
                            "vlm_annotator_model": annotation_fields.get("annotator_model"),
                            "transform": raw_object.get("transform"),
                            "support_region_asset": _repo_path(annot_path) if annot_path else None,
                            "support_region_surface_asset": (
                                _repo_path(annot_surface_path) if annot_surface_path else None
                            ),
                        },
                    }
                )
            scene_uid = f"{HSM_DATASET_KEY}/{scene_id}"
            # Look up prompt from SceneEval-500 annotations
            scene_id_num = scene_id.removeprefix("scene_")
            annotation = None
            try:
                annotation = sceneval_annotations.get(int(scene_id_num))
            except (ValueError, TypeError):
                pass
            prompt_text = annotation.get("Description") if annotation else None
            description = prompt_text or f"HSM generated scene {scene_id}"
            manifest = _scene_manifest_base(
                dataset=HSM_DATASET_KEY,
                scene_id=scene_id,
                scene_uid=scene_uid,
                subset=None,
                scene_dir=scene_path.parent,
                description=description,
            )
            manifest["source"] = {
                "scene_json": _repo_path(scene_path),
            }
            if annotation:
                manifest["sceneeval_annotation"] = annotation
            manifest.update(
                {
                    "status": "ready",
                    "display": {
                        "title": scene_id,
                        "subtitle": prompt_text[:120] if prompt_text else "HSM generated scene",
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
                    "description": description,
                    "title": scene_id,
                    "subtitle": prompt_text[:120] if prompt_text else None,
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
