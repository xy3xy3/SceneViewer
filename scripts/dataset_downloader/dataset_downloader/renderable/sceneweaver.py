from __future__ import annotations

import os
import subprocess

from .common import *


def _sceneweaver_blender_bin() -> str:
    return os.environ.get("SCENEVIEWER_BLENDER_BIN", "blender")


def _sceneweaver_repo_root() -> Path:
    candidate_env = os.environ.get("SCENEVIEWER_SCENEWEAVER_REPO")
    candidates = [Path(candidate_env).expanduser()] if candidate_env else []
    candidates.extend(
        [
            REPO_ROOT.parent / "SceneWeaver",
            Path("/home/xy/proj/SceneWeaver"),
        ]
    )
    for candidate in candidates:
        if (candidate / "infinigen" / "tools" / "export.py").exists():
            return candidate
    raise SystemExit(
        "Could not locate the SceneWeaver repository. "
        "Set SCENEVIEWER_SCENEWEAVER_REPO to a checkout containing infinigen/tools/export.py."
    )


def _sceneweaver_bake_resolution() -> int:
    raw_value = os.environ.get("SCENEVIEWER_SCENEWEAVER_BAKE_RESOLUTION", "1024")
    try:
        return max(int(raw_value), 256)
    except ValueError:
        return 1024


def _sceneweaver_bake_device() -> str:
    return os.environ.get("SCENEVIEWER_SCENEWEAVER_BAKE_DEVICE", "CPU")


def _sceneweaver_export_glb(*, blend_path: Path, output_path: Path) -> None:
    export_script = Path(__file__).with_name("sceneweaver_blender_export.py")
    subprocess.run(
        [
            _sceneweaver_blender_bin(),
            "-b",
            "--python-exit-code",
            "1",
            "--python",
            str(export_script),
            "--",
            "--blend",
            str(blend_path),
            "--output",
            str(output_path),
            "--sceneweaver-repo",
            str(_sceneweaver_repo_root()),
            "--resolution",
            str(_sceneweaver_bake_resolution()),
            "--device",
            _sceneweaver_bake_device(),
        ],
        check=True,
    )
    if not output_path.exists():
        raise RuntimeError(f"SceneWeaver export did not produce {output_path}")


def _sceneweaver_renderable_object(
    source_object: dict[str, object],
) -> dict[str, object]:
    metadata = source_object.get("metadata") or {}
    source_location = metadata.get("source_location") if isinstance(metadata, dict) else None
    source_rotation = metadata.get("source_rotation") if isinstance(metadata, dict) else None
    source_size = metadata.get("source_size") if isinstance(metadata, dict) else None

    loc_x = float(source_location[0]) if isinstance(source_location, list) and len(source_location) >= 1 else 0.0
    loc_y = float(source_location[1]) if isinstance(source_location, list) and len(source_location) >= 2 else 0.0
    loc_z = float(source_location[2]) if isinstance(source_location, list) and len(source_location) >= 3 else 0.0
    size_x = float(source_size[0]) if isinstance(source_size, list) and len(source_size) >= 1 else 0.2
    size_y = float(source_size[1]) if isinstance(source_size, list) and len(source_size) >= 2 else 0.2
    size_z = float(source_size[2]) if isinstance(source_size, list) and len(source_size) >= 3 else 0.2
    rotation_y_deg = 0.0
    if isinstance(source_rotation, list) and len(source_rotation) >= 3:
        rotation_y_deg = math.degrees(-float(source_rotation[2]))

    return {
        "id": source_object["id"],
        "position": [loc_x, loc_z + size_z / 2.0, -loc_y],
        "size": [size_x, size_z, size_y],
        "rotation_y_deg": rotation_y_deg,
        "object_type": source_object.get("type") or source_object.get("name"),
        "description": source_object.get("description"),
    }


def build_sceneweaver_renderables(scene_limit: int | None = None) -> dict[str, object]:
    dataset = "sceneweaver"
    source_index_path = _preprocessed_index_path(dataset)
    if not source_index_path.exists():
        raise SystemExit(
            "Missing assets/preprocessed/sceneweaver/index.json. "
            "Run `dataset-downloader preprocess sceneweaver` first."
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
        subset = scene_manifest["subset"]
        scene_id = scene_manifest["scene_id"]
        blend_rel = ((scene_manifest.get("assets") or {}).get("blend_file"))
        if not isinstance(blend_rel, str):
            continue

        blend_path = REPO_ROOT / blend_rel
        output_dir = _sceneweaver_renderable_scene_output(subset, scene_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        scene_glb_path = output_dir / "scene.glb"
        _sceneweaver_export_glb(blend_path=blend_path, output_path=scene_glb_path)

        normalized = scene_manifest.get("normalized") or {}
        room = ((normalized.get("rooms") or [{}])[0]) if isinstance(normalized.get("rooms"), list) else {}
        room_dimensions = room.get("dimensions") if isinstance(room, dict) else {}
        render_manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": _now_utc(),
            "dataset": dataset,
            "scene_id": scene_id,
            "scene_uid": scene_manifest["scene_uid"],
            "subset": subset,
            "source_scene_manifest": summary["scene_manifest"],
            "scene_glb": _repo_path(scene_glb_path),
            "room": {
                "id": room.get("id"),
                "room_type": room.get("room_type"),
                "dimensions": room_dimensions,
            },
            "objects": [
                _sceneweaver_renderable_object(source_object)
                for source_object in normalized.get("objects", [])
                if isinstance(source_object, dict)
            ],
        }
        render_manifest_path = output_dir / "scene.json"
        _write_json(render_manifest_path, render_manifest)
        scenes.append(
            {
                "scene_id": scene_id,
                "scene_uid": scene_manifest["scene_uid"],
                "subset": subset,
                "render_manifest": _repo_path(render_manifest_path),
                "object_count": len(render_manifest["objects"]),
                "room_count": 1,
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
