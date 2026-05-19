from __future__ import annotations

from .common import *

def _front3d_texture_image(
    texture_archive: zipfile.ZipFile,
    texture_jid: str | None,
) -> Image.Image | None:
    if not texture_jid:
        return None
    for member_name in front3d_texture_member_candidates(texture_jid):
        try:
            return Image.open(BytesIO(texture_archive.read(member_name))).convert("RGB")
        except KeyError:
            continue
    return None


def _front3d_base_color(material: dict[str, object] | None) -> list[float]:
    color = material.get("color") if material else None
    if isinstance(color, list) and len(color) >= 4:
        rgba = [float(value) / 255.0 for value in color[:4]]
        return rgba
    if isinstance(color, list) and len(color) >= 3:
        rgb = [float(value) / 255.0 for value in color[:3]]
        return [*rgb, 1.0]
    return [1.0, 1.0, 1.0, 1.0]


def _front3d_float_array(values: object, width: int) -> np.ndarray:
    if not isinstance(values, list) or len(values) < width:
        return np.empty((0, width), dtype=np.float32)
    return np.asarray(values, dtype=np.float32).reshape(-1, width)


def _front3d_face_array(values: object) -> np.ndarray:
    if not isinstance(values, list) or len(values) < 3:
        return np.empty((0, 3), dtype=np.uint32)
    return np.asarray(values, dtype=np.uint32).reshape(-1, 3)


def _front3d_build_mesh_asset(
    *,
    mesh: dict[str, object],
    material: dict[str, object] | None,
    texture_archive: zipfile.ZipFile,
    output_path: Path,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vertices = _front3d_float_array(mesh.get("xyz"), 3)
    faces = _front3d_face_array(mesh.get("faces"))
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"Mesh {mesh.get('uid')} has no geometry.")

    normals = _front3d_float_array(mesh.get("normal"), 3)
    if normals.shape[0] != vertices.shape[0]:
        normals = _compute_vertex_normals(vertices, faces.astype(np.int64))

    uvs = _front3d_float_array(mesh.get("uv"), 2)
    if uvs.shape[0] != vertices.shape[0]:
        uvs = np.zeros((vertices.shape[0], 2), dtype=np.float32)

    texture_image = _front3d_texture_image(
        texture_archive,
        material.get("jid") if material else None,
    )
    _write_sage_glb(
        output_path=output_path,
        vertices=vertices.astype(np.float32),
        normals=normals.astype(np.float32),
        uvs=uvs.astype(np.float32),
        faces=faces.astype(np.uint32),
        texture_image=texture_image,
        base_color_factor=_front3d_base_color(material),
    )
    min_corner = vertices.min(axis=0)
    max_corner = vertices.max(axis=0)
    return {
        "asset_path": _repo_path(output_path),
        "native_size": (max_corner - min_corner).astype(float).tolist(),
        "triangle_count": int(faces.shape[0]),
        "textured": texture_image is not None,
    }


def _parse_obj_asset(obj_text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions: list[list[float]] = []
    texcoords: list[list[float]] = []
    normals: list[list[float]] = []
    unique_positions: list[list[float]] = []
    unique_uvs: list[list[float]] = []
    unique_normals: list[list[float]] = []
    index_lookup: dict[tuple[int, int, int], int] = {}
    faces: list[list[int]] = []

    for raw_line in obj_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("v "):
            _, x, y, z, *_ = line.split()
            positions.append([float(x), float(y), float(z)])
            continue
        if line.startswith("vt "):
            _, u, v, *_ = line.split()
            texcoords.append([float(u), 1.0 - float(v)])
            continue
        if line.startswith("vn "):
            _, x, y, z, *_ = line.split()
            normals.append([float(x), float(y), float(z)])
            continue
        if not line.startswith("f "):
            continue

        refs = line.split()[1:]
        if len(refs) < 3:
            continue

        def resolve_face_index(token: str) -> int:
            parts = token.split("/")
            position_index = int(parts[0]) - 1
            texcoord_index = int(parts[1]) - 1 if len(parts) >= 2 and parts[1] else -1
            normal_index = int(parts[2]) - 1 if len(parts) >= 3 and parts[2] else -1
            key = (position_index, texcoord_index, normal_index)
            if key not in index_lookup:
                index_lookup[key] = len(unique_positions)
                unique_positions.append(positions[position_index])
                unique_uvs.append(
                    texcoords[texcoord_index]
                    if texcoord_index >= 0 and texcoord_index < len(texcoords)
                    else [0.0, 0.0]
                )
                unique_normals.append(
                    normals[normal_index]
                    if normal_index >= 0 and normal_index < len(normals)
                    else [0.0, 0.0, 0.0]
                )
            return index_lookup[key]

        resolved = [resolve_face_index(token) for token in refs]
        for a, b, c in _triangulate_polygon(list(range(len(resolved)))):
            faces.append([resolved[a], resolved[b], resolved[c]])

    vertices_array = np.asarray(unique_positions, dtype=np.float32)
    normals_array = np.asarray(unique_normals, dtype=np.float32)
    uvs_array = np.asarray(unique_uvs, dtype=np.float32)
    faces_array = np.asarray(faces, dtype=np.uint32)
    if not np.any(normals_array):
        normals_array = _compute_vertex_normals(vertices_array, faces_array.astype(np.int64))
    return vertices_array, normals_array, uvs_array, faces_array


def _front3d_model_texture_image(
    model_archive: zipfile.ZipFile,
    model_dir: str,
) -> Image.Image | None:
    for candidate in (
        f"{model_dir}/texture.png",
        f"{model_dir}/texture.jpg",
        f"{model_dir}/texture.jpeg",
    ):
        try:
            return Image.open(BytesIO(model_archive.read(candidate))).convert("RGB")
        except KeyError:
            continue
    return None


def _build_front3d_model_asset(
    *,
    model_archive: zipfile.ZipFile,
    model_jid: str,
    output_path: Path,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir = f"3D-FUTURE-model/{model_jid}"
    obj_member = f"{model_dir}/raw_model.obj"
    obj_text = model_archive.read(obj_member).decode("utf-8", errors="ignore")
    vertices, normals, uvs, faces = _parse_obj_asset(obj_text)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"Model {model_jid} has no renderable geometry.")

    texture_image = _front3d_model_texture_image(model_archive, model_dir)
    _write_sage_glb(
        output_path=output_path,
        vertices=vertices,
        normals=normals,
        uvs=uvs,
        faces=faces,
        texture_image=texture_image,
        base_color_factor=[1.0, 1.0, 1.0, 1.0],
    )
    min_corner = vertices.min(axis=0)
    max_corner = vertices.max(axis=0)
    return {
        "asset_path": _repo_path(output_path),
        "native_size": (max_corner - min_corner).astype(float).tolist(),
        "triangle_count": int(faces.shape[0]),
        "textured": texture_image is not None,
    }


def _front3d_has_model_geometry(model_members: set[str], model_jid: str) -> bool:
    return f"3D-FUTURE-model/{model_jid}/raw_model.obj" in model_members


def build_3dfront_renderables(scene_limit: int | None = None) -> dict[str, object]:
    ensure_front3d_archives()

    dataset = FRONT3D_DATASET_KEY
    source_index_path = _preprocessed_index_path(dataset)
    if not source_index_path.exists():
        raise SystemExit(
            "Missing assets/preprocessed/3dfront/index.json. Run `dataset-downloader preprocess 3dfront` first."
        )

    output_root = RENDERABLE_ROOT / dataset
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    source_index = _read_json(source_index_path)
    shared_dir = output_root / "shared" / "models"
    shared_dir.mkdir(parents=True, exist_ok=True)

    scenes: list[dict[str, object]] = []
    shared_assets: dict[str, dict[str, object]] = {}
    scene_summaries = list(source_index.get("scenes", []))
    if scene_limit is not None:
        scene_summaries = scene_summaries[:scene_limit]

    _write_renderable_progress(
        dataset=dataset,
        scenes=scenes,
        source_scene_count=len(scene_summaries),
        status="in_progress",
        shared_asset_count=len(shared_assets),
    )

    layout_cache: dict[str, dict[str, object]] = {}
    with zipfile.ZipFile(FRONT3D_LAYOUT_ZIP) as layout_archive, zipfile.ZipFile(
        FRONT3D_MODEL_ZIP
    ) as model_archive, zipfile.ZipFile(FRONT3D_TEXTURE_ZIP) as texture_archive:
        model_members = set(model_archive.namelist())
        progress = _scene_progress(dataset, scene_summaries)
        for scene_summary in progress:
            summary = dict(scene_summary)
            progress.set_postfix_str(summary.get("scene_uid", summary.get("scene_id", "")), refresh=False)
            scene_manifest_path = REPO_ROOT / summary["scene_manifest"]
            scene_manifest = _read_json(scene_manifest_path)
            scene_id = scene_manifest["scene_id"]
            house_id = scene_manifest.get("subset") or (scene_manifest.get("normalized") or {}).get("house_id")
            layout_entry = (scene_manifest.get("normalized") or {}).get("house_layout_entry")
            if not isinstance(layout_entry, str):
                raise ValueError(f"3D-FRONT scene {scene_id} is missing house_layout_entry.")

            if layout_entry not in layout_cache:
                layout_cache[layout_entry] = json.loads(layout_archive.read(layout_entry))
            layout = layout_cache[layout_entry]
            mesh_by_uid = {
                item.get("uid"): item
                for item in layout.get("mesh", [])
                if isinstance(item, dict) and item.get("uid")
            }
            material_by_uid = {
                item.get("uid"): item
                for item in layout.get("material", [])
                if isinstance(item, dict) and item.get("uid")
            }

            room_manifest = ((scene_manifest.get("normalized") or {}).get("rooms") or [None])[0]
            if not isinstance(room_manifest, dict):
                raise ValueError(f"3D-FRONT scene {scene_id} is missing normalized room data.")

            room_shells: list[dict[str, object]] = []
            shell_refs = room_manifest.get("shell_refs") or []
            scene_output_dir = _front3d_renderable_scene_output(str(house_id or "house"), scene_id)
            for shell_ref in shell_refs:
                if not isinstance(shell_ref, dict):
                    continue
                mesh_uid = shell_ref.get("mesh_uid")
                if not isinstance(mesh_uid, str):
                    continue
                mesh = mesh_by_uid.get(mesh_uid)
                if not mesh:
                    continue
                material = material_by_uid.get(mesh.get("material"))
                shell_asset = _front3d_build_mesh_asset(
                    mesh=mesh,
                    material=material,
                    texture_archive=texture_archive,
                    output_path=scene_output_dir / "room_shells" / f"{safe_front3d_name(mesh_uid)}.glb",
                )
                room_shells.append(
                    {
                        "id": shell_ref.get("id") or mesh_uid,
                        "category": shell_ref.get("category") or map_front3d_shell_category(mesh.get("type")),
                        "asset_path": shell_asset["asset_path"],
                        "position": [0.0, 0.0, 0.0],
                        "rotation_y_deg": 0.0,
                        "scale": [1.0, 1.0, 1.0],
                        "room_id": room_manifest.get("id"),
                    }
                )

            renderable_objects: list[dict[str, object]] = []
            skipped_object_count = 0
            for obj in (scene_manifest.get("normalized") or {}).get("objects", []):
                metadata = obj.get("metadata") or {}
                model_jid = metadata.get("jid")
                if not metadata.get("valid") or not isinstance(model_jid, str):
                    continue
                if not _front3d_has_model_geometry(model_members, model_jid):
                    skipped_object_count += 1
                    continue

                if model_jid not in shared_assets:
                    shared_assets[model_jid] = _build_front3d_model_asset(
                        model_archive=model_archive,
                        model_jid=model_jid,
                        output_path=shared_dir / f"{safe_front3d_name(model_jid)}.glb",
                    )

                position = obj.get("position") or {}
                scale = obj.get("scale") or [1.0, 1.0, 1.0]
                quaternion = obj.get("quaternion")
                renderable_objects.append(
                    {
                        "id": obj["id"],
                        "asset_path": shared_assets[model_jid]["asset_path"],
                        "position": [
                            float(position.get("x", 0.0)),
                            float(position.get("y", 0.0)),
                            float(position.get("z", 0.0)),
                        ],
                        "rotation_y_deg": 0.0,
                        "quaternion": quaternion if isinstance(quaternion, list) and len(quaternion) >= 4 else None,
                        "scale": [
                            float(scale[0]) if len(scale) >= 1 else 1.0,
                            float(scale[1]) if len(scale) >= 2 else 1.0,
                            float(scale[2]) if len(scale) >= 3 else 1.0,
                        ],
                        "room_id": obj.get("room_id"),
                        "object_type": obj.get("type"),
                        "description": obj.get("description"),
                        "source_model_jid": model_jid,
                        "source_ref": metadata.get("source_ref"),
                    }
                )

            render_manifest = {
                "schema_version": SCHEMA_VERSION,
                "generated_at_utc": _now_utc(),
                "dataset": dataset,
                "scene_id": scene_id,
                "scene_uid": scene_manifest["scene_uid"],
                "subset": house_id,
                "house_id": house_id,
                "source_scene_manifest": summary["scene_manifest"],
                "room_shells": room_shells,
                "objects": renderable_objects,
                "skipped_object_count": skipped_object_count,
            }
            render_manifest_path = scene_output_dir / "scene.json"
            _write_json(render_manifest_path, render_manifest)
            scenes.append(
                {
                    "scene_id": scene_id,
                    "scene_uid": scene_manifest["scene_uid"],
                    "subset": house_id,
                    "render_manifest": _repo_path(render_manifest_path),
                    "object_count": len(renderable_objects),
                    "room_shell_count": len(room_shells),
                    "skipped_object_count": skipped_object_count,
                }
            )
            _write_renderable_progress(
                dataset=dataset,
                scenes=scenes,
                source_scene_count=len(scene_summaries),
                status="in_progress",
                shared_asset_count=len(shared_assets),
            )

    return _write_renderable_progress(
        dataset=dataset,
        scenes=scenes,
        source_scene_count=len(scene_summaries),
        status="ready",
        shared_asset_count=len(shared_assets),
    )

