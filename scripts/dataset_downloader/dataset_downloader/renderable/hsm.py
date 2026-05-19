from __future__ import annotations

from .common import *

def build_hsm_renderables(scene_limit: int | None = None) -> dict[str, object]:
    ensure_hsm_hssd_models()

    dataset = HSM_DATASET_KEY
    source_index_path = _preprocessed_index_path(dataset)
    if not source_index_path.exists():
        raise SystemExit(
            "Missing assets/preprocessed/hsm/index.json. Run `dataset-downloader preprocess hsm` first."
        )

    output_root = RENDERABLE_ROOT / dataset
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    source_index = _read_json(source_index_path)
    scenes: list[dict[str, object]] = []

    scene_summaries = list(source_index.get("scenes", []))
    if scene_limit is not None:
        scene_summaries = scene_summaries[:scene_limit]

    _write_renderable_progress(
        dataset=dataset,
        scenes=scenes,
        source_scene_count=len(scene_summaries),
        status="in_progress",
    )

    progress = _scene_progress(dataset, scene_summaries)
    for scene_summary in progress:
        summary = dict(scene_summary)
        progress.set_postfix_str(summary.get("scene_uid", summary.get("scene_id", "")), refresh=False)
        scene_manifest_path = REPO_ROOT / summary["scene_manifest"]
        scene_manifest = _read_json(scene_manifest_path)
        scene_id = scene_manifest["scene_id"]

        renderable_objects: list[dict[str, object]] = []
        skipped_object_count = 0
        for obj in (scene_manifest.get("normalized") or {}).get("objects", []):
            source_id = obj.get("source_id")
            if not isinstance(source_id, str):
                skipped_object_count += 1
                continue
            glb_path = hsm_hssd_glb_path(source_id)
            if not glb_path.exists():
                skipped_object_count += 1
                continue
            position = obj.get("position") or {}
            scale = obj.get("scale") or [1.0, 1.0, 1.0]
            metadata = obj.get("metadata") or {}
            renderable_objects.append(
                {
                    "id": obj["id"],
                    "asset_path": hsm_repo_relative_path(glb_path),
                    "position": [
                        float(position.get("x", 0.0)),
                        float(position.get("z", 0.0)),
                        float(position.get("y", 0.0)),
                    ],
                    "rotation_y_deg": float((obj.get("rotation") or {}).get("z", 0.0)),
                    "quaternion": hsm_three_quaternion_from_transform(
                        metadata.get("transform"),
                        up=metadata.get("hssd_up") if isinstance(metadata.get("hssd_up"), str) else None,
                        front=metadata.get("hssd_front") if isinstance(metadata.get("hssd_front"), str) else None,
                    ),
                    "scale": [
                        float(scale[0]) if len(scale) >= 1 else 1.0,
                        float(scale[2]) if len(scale) >= 3 else 1.0,
                        float(scale[1]) if len(scale) >= 2 else 1.0,
                    ],
                    "source_id": source_id,
                    "name": obj.get("name"),
                    "category": metadata.get("hssd_wnsynsetkey"),
                    "semantic_label": metadata.get("hssd_semantic_label"),
                    "description": obj.get("description"),
                    "type": obj.get("type"),
                    "object_type": obj.get("object_type") or obj.get("type"),
                    "support_region_asset": metadata.get("support_region_asset"),
                    "support_region_surface_asset": metadata.get("support_region_surface_asset"),
                }
            )

        renderable_rooms: list[dict[str, object]] = []
        for room in (scene_manifest.get("normalized") or {}).get("rooms", []):
            renderable_rooms.append(
                {
                    "id": room["id"],
                    "room_type": room.get("room_type"),
                    "dimensions": room.get("dimensions"),
                    "ceiling_height": room.get("ceiling_height"),
                    "floor_texture_path": None,
                    "wall_texture_path": None,
                    "walls": room.get("walls") or [],
                    "doors": room.get("doors") or [],
                    "windows": room.get("windows") or [],
                }
            )

        render_manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": _now_utc(),
            "dataset": dataset,
            "scene_id": scene_id,
            "scene_uid": scene_manifest["scene_uid"],
            "source_scene_manifest": summary["scene_manifest"],
            "objects": renderable_objects,
            "rooms": renderable_rooms,
            "skipped_object_count": skipped_object_count,
        }
        output_dir = RENDERABLE_ROOT / dataset / scene_id
        render_manifest_path = output_dir / "scene.json"
        _write_json(render_manifest_path, render_manifest)
        scenes.append(
            {
                "scene_id": scene_id,
                "scene_uid": scene_manifest["scene_uid"],
                "render_manifest": _repo_path(render_manifest_path),
                "object_count": len(renderable_objects),
                "room_count": len(renderable_rooms),
                "skipped_object_count": skipped_object_count,
            }
        )
        _write_renderable_progress(
            dataset=dataset,
            scenes=scenes,
            source_scene_count=len(scene_summaries),
            status="in_progress",
        )

    return _write_renderable_progress(
        dataset=dataset,
        scenes=scenes,
        source_scene_count=len(scene_summaries),
        status="ready",
    )

