from __future__ import annotations

from .common import *


def _scenesmith_rotation_y_deg(object_data: dict[str, object]) -> float:
    rotation_three = _scenesmith_source_rotation_to_three(
        _scenesmith_source_rotation_matrix(object_data)
    )
    return math.degrees(math.atan2(float(rotation_three[0, 2]), float(rotation_three[0, 0])))


def _scenesmith_three_quaternion(object_data: dict[str, object]) -> list[float] | None:
    rotation_three = _scenesmith_source_rotation_to_three(
        _scenesmith_source_rotation_matrix(object_data)
    )
    return _quaternion_from_rotation_matrix(rotation_three)


def _scenesmith_resolve_repo_relative_path(base_path: str, relative_path: str) -> str:
    base_parts = base_path.split("/")
    if base_parts:
        base_parts.pop()
    for segment in relative_path.split("/"):
        if not segment or segment == ".":
            continue
        if segment == "..":
            if base_parts:
                base_parts.pop()
            continue
        base_parts.append(segment)
    return "/".join(base_parts)


def _parse_scenesmith_mesh_scale(scale_text: str | None) -> list[float] | None:
    if not scale_text:
        return None
    values = [float(value) for value in scale_text.split() if value.strip()]
    if len(values) < 3:
        return None
    return [values[0], values[1], values[2]]


def _scenesmith_object_scale(
    object_data: dict[str, object],
    cache: dict[tuple[str, str], list[float]],
) -> list[float]:
    sdf_path = object_data.get("sdf_path")
    gltf_path = object_data.get("gltf_path")
    if not isinstance(sdf_path, str) or not isinstance(gltf_path, str):
        return [1.0, 1.0, 1.0]

    cache_key = (sdf_path, gltf_path)
    if cache_key in cache:
        return cache[cache_key]

    resolved_sdf_path = REPO_ROOT / sdf_path
    if not resolved_sdf_path.exists():
        cache[cache_key] = [1.0, 1.0, 1.0]
        return cache[cache_key]

    try:
        document = ET.fromstring(resolved_sdf_path.read_text())
    except ET.ParseError:
        cache[cache_key] = [1.0, 1.0, 1.0]
        return cache[cache_key]

    target_name = Path(gltf_path).name
    for visual in document.findall(".//visual"):
        uri = visual.findtext("./geometry/mesh/uri")
        if not uri:
            continue
        resolved_gltf_path = _scenesmith_resolve_repo_relative_path(sdf_path, uri.strip())
        if resolved_gltf_path != gltf_path and Path(resolved_gltf_path).name != target_name:
            continue
        scale = _parse_scenesmith_mesh_scale(visual.findtext("./geometry/mesh/scale"))
        cache[cache_key] = scale or [1.0, 1.0, 1.0]
        return cache[cache_key]

    cache[cache_key] = [1.0, 1.0, 1.0]
    return cache[cache_key]


def _scenesmith_room_geometry_visual_asset_paths(scene_manifest: dict[str, object]) -> set[str]:
    result: set[str] = set()
    rooms = ((scene_manifest.get("normalized") or {}).get("rooms") or [])
    for room in rooms:
        if not isinstance(room, dict):
            continue
        room_geometry_path = room.get("room_geometry_sdf")
        if not isinstance(room_geometry_path, str):
            continue
        sdf_path = REPO_ROOT / room_geometry_path
        if not sdf_path.exists():
            continue
        try:
            document = ET.fromstring(sdf_path.read_text())
        except ET.ParseError:
            continue
        for visual in document.findall(".//visual"):
            uri = visual.findtext("./geometry/mesh/uri")
            if not uri:
                continue
            result.add(_scenesmith_resolve_repo_relative_path(room_geometry_path, uri.strip()))
    return result


def _scenesmith_gltf_bounds(asset_path: str, cache: dict[str, dict[str, float] | None]) -> dict[str, float] | None:
    if asset_path in cache:
        return cache[asset_path]

    path = REPO_ROOT / asset_path
    if not path.exists() or path.suffix.lower() != ".gltf":
        cache[asset_path] = None
        return None

    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        cache[asset_path] = None
        return None

    accessors = payload.get("accessors") or []
    meshes = payload.get("meshes") or []
    accessor_indices: set[int] = set()
    for mesh in meshes:
        if not isinstance(mesh, dict):
            continue
        for primitive in mesh.get("primitives") or []:
            if not isinstance(primitive, dict):
                continue
            attributes = primitive.get("attributes") or {}
            position_accessor = attributes.get("POSITION")
            if isinstance(position_accessor, int):
                accessor_indices.add(position_accessor)

    min_corner = np.array([math.inf, math.inf, math.inf], dtype=float)
    max_corner = np.array([-math.inf, -math.inf, -math.inf], dtype=float)
    for accessor_index in accessor_indices:
        if accessor_index < 0 or accessor_index >= len(accessors):
            continue
        accessor = accessors[accessor_index]
        if not isinstance(accessor, dict):
            continue
        mins = accessor.get("min")
        maxs = accessor.get("max")
        if not isinstance(mins, list) or not isinstance(maxs, list) or len(mins) < 3 or len(maxs) < 3:
            continue
        min_corner = np.minimum(min_corner, np.asarray(mins[:3], dtype=float))
        max_corner = np.maximum(max_corner, np.asarray(maxs[:3], dtype=float))

    if not np.isfinite(min_corner).all() or not np.isfinite(max_corner).all():
        cache[asset_path] = None
        return None

    result = {
        "size_x": float(max_corner[0] - min_corner[0]),
        "size_y": float(max_corner[1] - min_corner[1]),
        "size_z": float(max_corner[2] - min_corner[2]),
        "min_y": float(min_corner[1]),
        "max_y": float(max_corner[1]),
    }
    cache[asset_path] = result
    return result


def _scenesmith_object_text(object_data: dict[str, object]) -> str:
    parts = [
        object_data.get("id"),
        object_data.get("object_type"),
        object_data.get("description"),
    ]
    return " ".join(str(part) for part in parts if part).lower()


def _is_scenesmith_seat(object_data: dict[str, object]) -> bool:
    text = _scenesmith_object_text(object_data)
    return "chair" in text or "stool" in text or "bench" in text


def _is_scenesmith_surface_anchor(object_data: dict[str, object]) -> bool:
    text = _scenesmith_object_text(object_data)
    if not any(token in text for token in ("table", "desk", "worktable")):
        return False
    return not any(token in text for token in ("coffee_table", "side_table", "bedside_table", "tv_console"))


def _rotate_xz(vector_x: float, vector_z: float, yaw_rad: float) -> tuple[float, float]:
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    return (
        cosine * vector_x + sine * vector_z,
        -sine * vector_x + cosine * vector_z,
    )


def _scenesmith_resolve_seat_surface_penetrations(renderable_objects: list[dict[str, object]]) -> None:
    bounds_cache: dict[str, dict[str, float] | None] = {}
    objects_by_room: dict[str, list[dict[str, object]]] = {}
    for obj in renderable_objects:
        room_id = obj.get("room_id")
        if isinstance(room_id, str):
            objects_by_room.setdefault(room_id, []).append(obj)

    gap = 0.08
    for room_objects in objects_by_room.values():
        surfaces = [obj for obj in room_objects if _is_scenesmith_surface_anchor(obj)]
        seats = [obj for obj in room_objects if _is_scenesmith_seat(obj)]
        for seat in seats:
            seat_asset_path = seat.get("asset_path")
            if not isinstance(seat_asset_path, str):
                continue
            seat_bounds = _scenesmith_gltf_bounds(seat_asset_path, bounds_cache)
            seat_position = seat.get("position")
            if seat_bounds is None or not isinstance(seat_position, list) or len(seat_position) < 3:
                continue

            for surface in surfaces:
                surface_asset_path = surface.get("asset_path")
                surface_position = surface.get("position")
                if not isinstance(surface_asset_path, str):
                    continue
                if not isinstance(surface_position, list) or len(surface_position) < 3:
                    continue

                surface_bounds = _scenesmith_gltf_bounds(surface_asset_path, bounds_cache)
                if surface_bounds is None:
                    continue

                surface_yaw_deg = float(surface.get("rotation_y_deg") or 0.0)
                surface_yaw_rad = math.radians(surface_yaw_deg)
                dx = float(seat_position[0]) - float(surface_position[0])
                dz = float(seat_position[2]) - float(surface_position[2])
                local_x, local_z = _rotate_xz(dx, dz, -surface_yaw_rad)

                relative_yaw_rad = math.radians(
                    float(seat.get("rotation_y_deg") or 0.0) - surface_yaw_deg
                )
                seat_half_x = 0.5 * (
                    abs(math.cos(relative_yaw_rad)) * seat_bounds["size_x"]
                    + abs(math.sin(relative_yaw_rad)) * seat_bounds["size_z"]
                )
                seat_half_z = 0.5 * (
                    abs(math.sin(relative_yaw_rad)) * seat_bounds["size_x"]
                    + abs(math.cos(relative_yaw_rad)) * seat_bounds["size_z"]
                )
                target_x = surface_bounds["size_x"] / 2.0 + seat_half_x + gap
                target_z = surface_bounds["size_z"] / 2.0 + seat_half_z + gap
                if abs(local_x) >= target_x or abs(local_z) >= target_z:
                    continue

                push_x = target_x - abs(local_x)
                push_z = target_z - abs(local_z)
                if push_x <= push_z:
                    axis = "x"
                    current_value = local_x
                    target_value = target_x
                else:
                    axis = "z"
                    current_value = local_z
                    target_value = target_z

                direction = 1.0 if current_value >= 0.0 else -1.0
                if abs(current_value) <= 1e-4:
                    direction = 1.0

                if axis == "x":
                    local_x = direction * target_value
                else:
                    local_z = direction * target_value

                world_dx, world_dz = _rotate_xz(local_x, local_z, surface_yaw_rad)
                seat_position[0] = float(surface_position[0]) + world_dx
                seat_position[2] = float(surface_position[2]) + world_dz


def build_scenesmith_renderables(scene_limit: int | None = None) -> dict[str, object]:
    dataset = "scenesmith"
    source_index_path = _preprocessed_index_path(dataset)
    if not source_index_path.exists():
        raise SystemExit(
            "Missing assets/preprocessed/scenesmith/index.json. Run `dataset-downloader preprocess scenesmith` first."
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
        room_geometry_visual_assets = _scenesmith_room_geometry_visual_asset_paths(scene_manifest)

        room_frames = {
            room["id"]: ((room.get("frame") or {}).get("translation") or [0.0, 0.0, 0.0])
            for room in scene_manifest["normalized"]["rooms"]
        }

        room_shells: list[dict[str, object]] = []
        object_scale_cache: dict[tuple[str, str], list[float]] = {}
        for room in scene_manifest["normalized"]["rooms"]:
            room_translation = _axis_swap_source_to_three(room_frames.get(room["id"], [0.0, 0.0, 0.0]))
            floor_plan = room.get("floor_plan_assets") or {}
            for category, asset_paths in (
                ("floor", [floor_plan.get("floor_gltf")] if floor_plan.get("floor_gltf") else []),
                ("wall", floor_plan.get("wall_gltfs") or []),
                ("window", floor_plan.get("window_gltfs") or []),
            ):
                for asset_path in asset_paths:
                    if category == "window" and asset_path not in room_geometry_visual_assets:
                        continue
                    room_shells.append(
                        {
                            "id": f"{room['id']}::{category}::{_scenesmith_shell_id(asset_path)}",
                            "category": category,
                            "asset_path": asset_path,
                            "position": room_translation,
                            "rotation_y_deg": 0.0,
                            "scale": [1.0, 1.0, 1.0],
                            "room_id": room["id"],
                        }
                    )

        renderable_objects: list[dict[str, object]] = []
        for obj in scene_manifest["normalized"]["objects"]:
            asset_path = obj.get("gltf_path")
            if not asset_path:
                continue
            room_translation = room_frames.get(obj["room_id"], [0.0, 0.0, 0.0])
            local_translation = (obj.get("transform") or {}).get("translation") or [0.0, 0.0, 0.0]
            world_translation = [
                float(room_translation[0]) + float(local_translation[0]),
                float(room_translation[1]) + float(local_translation[1]),
                float(room_translation[2]) + float(local_translation[2]),
            ]
            renderable_objects.append(
                {
                    "id": obj["id"],
                    "asset_path": asset_path,
                    "position": _axis_swap_source_to_three(world_translation),
                    "rotation_y_deg": _scenesmith_rotation_y_deg(obj),
                    "quaternion": _scenesmith_three_quaternion(obj),
                    "scale": _scenesmith_object_scale(obj, object_scale_cache),
                    "room_id": obj["room_id"],
                    "object_type": obj.get("object_type"),
                    "description": obj.get("description"),
                }
            )

        render_manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": _now_utc(),
            "dataset": dataset,
            "scene_id": scene_id,
            "scene_uid": scene_manifest["scene_uid"],
            "subset": subset,
            "source_scene_manifest": summary["scene_manifest"],
            "room_shells": room_shells,
            "objects": renderable_objects,
        }
        output_dir = _scenesmith_renderable_scene_output(subset, scene_id)
        render_manifest_path = output_dir / "scene.json"
        _write_json(render_manifest_path, render_manifest)
        scenes.append(
            {
                "scene_id": scene_id,
                "scene_uid": scene_manifest["scene_uid"],
                "subset": subset,
                "render_manifest": _repo_path(render_manifest_path),
                "object_count": len(renderable_objects),
                "room_shell_count": len(room_shells),
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
