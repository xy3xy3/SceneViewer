from __future__ import annotations

from .common import *
from ..hssd import hssd_three_quaternion_from_instance


def _hssd_renderable_scene_output(scene_id: str) -> Path:
    return RENDERABLE_ROOT / "hssd" / scene_id


def build_hssd_renderables(scene_limit: int | None = None) -> dict[str, object]:
    dataset = "hssd"
    source_index_path = _preprocessed_index_path(dataset)
    if not source_index_path.exists():
        raise SystemExit(
            "Missing assets/preprocessed/hssd/index.json. "
            "Run `dataset-downloader preprocess hssd` first."
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
        progress.set_postfix_str(
            summary.get("scene_uid", summary.get("scene_id", "")),
            refresh=False,
        )
        scene_manifest_path = REPO_ROOT / summary["scene_manifest"]
        scene_manifest = _read_json(scene_manifest_path)
        scene_id = scene_manifest["scene_id"]
        stage_glb = ((scene_manifest.get("assets") or {}).get("stage_glb"))
        if not isinstance(stage_glb, str):
            continue

        normalized = scene_manifest.get("normalized") or {}
        room = ((normalized.get("rooms") or [{}])[0]) if isinstance(normalized.get("rooms"), list) else {}
        room_dimensions = room.get("dimensions") if isinstance(room, dict) else {}

        output_dir = _hssd_renderable_scene_output(scene_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        renderable_objects = []
        skipped_object_count = 0
        for source_object in normalized.get("objects", []):
            if not isinstance(source_object, dict):
                continue
            asset_path = source_object.get("gltf_path")
            position = source_object.get("position") if isinstance(source_object.get("position"), dict) else {}
            scale = source_object.get("scale") if isinstance(source_object.get("scale"), list) else []
            metadata = source_object.get("metadata") if isinstance(source_object.get("metadata"), dict) else {}
            if not isinstance(asset_path, str):
                skipped_object_count += 1
                continue
            if "/source/habitat/objects/decomposed/" in asset_path:
                skipped_object_count += 1
                continue
            renderable_objects.append(
                {
                    "id": source_object["id"],
                    "asset_path": asset_path,
                    "position": [
                        float(position.get("x", 0.0)),
                        float(position.get("y", 0.0)),
                        float(position.get("z", 0.0)),
                    ],
                    "size": [0.25, 0.25, 0.25],
                    "rotation_y_deg": 0.0,
                    "quaternion": hssd_three_quaternion_from_instance(
                        metadata.get("hssd_rotation_wxyz"),
                        up=metadata.get("hssd_up"),
                        front=metadata.get("hssd_front"),
                    ),
                    "scale": [
                        float(scale[0]) if len(scale) >= 1 else 1.0,
                        float(scale[1]) if len(scale) >= 2 else 1.0,
                        float(scale[2]) if len(scale) >= 3 else 1.0,
                    ],
                    "source_id": metadata.get("source_id"),
                    "object_type": source_object.get("object_type") or source_object.get("type"),
                    "description": source_object.get("description"),
                }
            )
        render_manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": _now_utc(),
            "dataset": dataset,
            "scene_id": scene_id,
            "scene_uid": scene_manifest["scene_uid"],
            "source_scene_manifest": summary["scene_manifest"],
            "scene_glb": stage_glb,
            "room": {
                "id": room.get("id"),
                "room_type": room.get("room_type"),
                "dimensions": room_dimensions,
            },
            "objects": renderable_objects,
            "skipped_object_count": skipped_object_count,
        }
        render_manifest_path = output_dir / "scene.json"
        _write_json(render_manifest_path, render_manifest)
        scenes.append(
            {
                "scene_id": scene_id,
                "scene_uid": scene_manifest["scene_uid"],
                "render_manifest": _repo_path(render_manifest_path),
                "object_count": len(renderable_objects),
                "room_count": 1,
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
