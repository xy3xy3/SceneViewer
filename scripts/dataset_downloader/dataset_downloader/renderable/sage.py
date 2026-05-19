from __future__ import annotations

from .common import *

def _write_sage_glb(
    *,
    output_path: Path,
    vertices: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    faces: np.ndarray,
    texture_image: Image.Image | None,
    base_color_factor: list[float] | None = None,
) -> None:
    position_bytes = vertices.astype(np.float32).tobytes()
    normal_bytes = normals.astype(np.float32).tobytes()
    texcoord_bytes = uvs.astype(np.float32).tobytes()
    index_bytes = faces.astype(np.uint32).reshape(-1).tobytes()

    blob = bytearray()
    position_offset, position_length = _append_aligned_blob(blob, position_bytes)
    normal_offset, normal_length = _append_aligned_blob(blob, normal_bytes)
    texcoord_offset, texcoord_length = _append_aligned_blob(blob, texcoord_bytes)
    index_offset, index_length = _append_aligned_blob(blob, index_bytes)

    gltf = GLTF2()
    gltf.asset = {"version": "2.0"}
    gltf.scenes = [Scene(nodes=[0])]
    gltf.scene = 0
    gltf.nodes = [Node(mesh=0)]
    gltf.buffers = [Buffer(byteLength=len(blob))]
    gltf.bufferViews = [
        BufferView(
            buffer=0,
            byteOffset=position_offset,
            byteLength=position_length,
            target=ARRAY_BUFFER,
        ),
        BufferView(
            buffer=0,
            byteOffset=normal_offset,
            byteLength=normal_length,
            target=ARRAY_BUFFER,
        ),
        BufferView(
            buffer=0,
            byteOffset=texcoord_offset,
            byteLength=texcoord_length,
            target=ARRAY_BUFFER,
        ),
        BufferView(
            buffer=0,
            byteOffset=index_offset,
            byteLength=index_length,
            target=ELEMENT_ARRAY_BUFFER,
        ),
    ]
    gltf.accessors = [
        Accessor(
            bufferView=0,
            componentType=FLOAT,
            count=int(vertices.shape[0]),
            type=VEC3,
            min=vertices.min(axis=0).astype(float).tolist(),
            max=vertices.max(axis=0).astype(float).tolist(),
        ),
        Accessor(
            bufferView=1,
            componentType=FLOAT,
            count=int(normals.shape[0]),
            type=VEC3,
        ),
        Accessor(
            bufferView=2,
            componentType=FLOAT,
            count=int(uvs.shape[0]),
            type=VEC2,
            min=uvs.min(axis=0).astype(float).tolist(),
            max=uvs.max(axis=0).astype(float).tolist(),
        ),
        Accessor(
            bufferView=3,
            componentType=UNSIGNED_INT,
            count=int(faces.size),
            type=SCALAR,
        ),
    ]

    texture_index = None
    if texture_image is not None:
        gltf.images = [GLTFImage(uri=_image_to_data_uri(texture_image))]
        gltf.samplers = [
            Sampler(
                magFilter=9729,
                minFilter=9987,
                wrapS=33071,
                wrapT=33071,
            )
        ]
        gltf.textures = [Texture(source=0, sampler=0)]
        texture_index = 0
    else:
        gltf.images = []
        gltf.samplers = []
        gltf.textures = []

    pbr = PbrMetallicRoughness(
        baseColorFactor=base_color_factor or [1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    if texture_index is not None:
        pbr.baseColorTexture = {"index": texture_index}

    material = Material(
        doubleSided=True,
        alphaMode="OPAQUE",
        pbrMetallicRoughness=pbr,
    )
    gltf.materials = [material]
    primitive = Primitive(
        attributes=Attributes(
            POSITION=0,
            NORMAL=1,
            TEXCOORD_0=2,
        ),
        indices=3,
        material=0,
    )
    gltf.meshes = [Mesh(primitives=[primitive])]

    gltf.set_binary_blob(bytes(blob))
    output_path.write_bytes(b"".join(gltf.save_to_bytes()))


def _build_textured_sage_asset(
    *,
    ply_path: Path,
    texture_path: Path | None,
    output_path: Path,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ply = PlyData.read(ply_path.open("rb"))
    vertex_data = ply["vertex"].data
    x = np.asarray(vertex_data["x"], dtype=np.float32)
    y = np.asarray(vertex_data["y"], dtype=np.float32)
    z = np.asarray(vertex_data["z"], dtype=np.float32)
    source_vertices = np.column_stack((x, y, z))
    vertices = _sage_vertices_to_three(source_vertices)

    texcoord_element = ply["texcoord"].data if "texcoord" in ply else None
    uv_lookup = None
    if texcoord_element is not None:
        s = np.asarray(texcoord_element["s"], dtype=np.float32)
        t = np.asarray(texcoord_element["t"], dtype=np.float32)
        uv_lookup = np.column_stack((s, 1.0 - t)).astype(np.float32)

    unique_vertices: list[list[float]] = []
    unique_uvs: list[list[float]] = []
    unique_lookup: dict[tuple[int, int], int] = {}
    faces: list[list[int]] = []

    for face in ply["face"].data:
        vertex_indices = [int(value) for value in face["vertex_indices"]]
        texcoord_indices = (
            [int(value) for value in face["texcoord_indices"]]
            if "texcoord_indices" in face.dtype.names
            else [-1] * len(vertex_indices)
        )

        for a, b, c in _triangulate_polygon(list(range(len(vertex_indices)))):
            triangle = []
            for local_index in (a, b, c):
                vertex_index = vertex_indices[local_index]
                texcoord_index = texcoord_indices[local_index]
                key = (vertex_index, texcoord_index)
                if key not in unique_lookup:
                    unique_lookup[key] = len(unique_vertices)
                    unique_vertices.append(vertices[vertex_index].tolist())
                    if uv_lookup is not None and texcoord_index >= 0:
                        unique_uvs.append(uv_lookup[texcoord_index].tolist())
                    else:
                        unique_uvs.append([0.0, 0.0])
                triangle.append(unique_lookup[key])
            faces.append(triangle)

    vertices_array = np.asarray(unique_vertices, dtype=np.float32)
    faces_array = np.asarray(faces, dtype=np.uint32)
    uv_array = np.asarray(unique_uvs, dtype=np.float32)
    normals_array = _compute_vertex_normals(vertices_array, faces_array.astype(np.int64))

    texture_image = None
    if texture_path and texture_path.exists():
        texture_image = Image.open(texture_path).convert("RGB")

    min_corner = vertices_array.min(axis=0)
    max_corner = vertices_array.max(axis=0)
    size = (max_corner - min_corner).tolist()
    origin_offset = np.array(
        [
            (min_corner[0] + max_corner[0]) / 2.0,
            min_corner[1],
            (min_corner[2] + max_corner[2]) / 2.0,
        ],
        dtype=np.float32,
    )
    centered_vertices = vertices_array - origin_offset
    _write_sage_glb(
        output_path=output_path,
        vertices=centered_vertices,
        normals=normals_array,
        uvs=uv_array,
        faces=faces_array,
        texture_image=texture_image,
    )

    return {
        "asset_path": _repo_path(output_path),
        "native_size": [float(value) for value in size],
        "source_vertex_count": int(source_vertices.shape[0]),
        "triangle_count": int(len(faces)),
        "textured": texture_image is not None,
    }


def build_sage_renderables(scene_limit: int | None = None) -> dict[str, object]:
    dataset = "sage"
    source_index_path = _preprocessed_index_path(dataset)
    if not source_index_path.exists():
        raise SystemExit(
            "Missing assets/preprocessed/sage/index.json. Run `dataset-downloader preprocess sage` first."
        )

    output_root = RENDERABLE_ROOT / dataset
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    source_index = _read_json(source_index_path)
    shared_dir = output_root / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    shared_assets: dict[str, dict[str, object]] = {}
    scenes: list[dict[str, object]] = []

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
    progress = _scene_progress(dataset, scene_summaries)
    for scene_summary in progress:
        summary = dict(scene_summary)
        progress.set_postfix_str(summary.get("scene_uid", summary.get("scene_id", "")), refresh=False)
        scene_manifest_path = REPO_ROOT / summary["scene_manifest"]
        scene_manifest = _read_json(scene_manifest_path)
        scene_id = scene_manifest["scene_id"]
        scene_dir = REPO_ROOT / scene_manifest["source"]["extracted_dir"]

        renderable_objects: list[dict[str, object]] = []
        for obj in scene_manifest["normalized"]["objects"]:
            source_id = obj.get("source_id")
            mesh_info = obj.get("mesh") or {}
            ply_repo_path = mesh_info.get("ply")
            if not source_id or not ply_repo_path:
                continue

            if source_id not in shared_assets:
                ply_path = REPO_ROOT / ply_repo_path
                texture_repo_path = mesh_info.get("texture")
                texture_path = REPO_ROOT / texture_repo_path if texture_repo_path else None
                shared_assets[source_id] = _build_textured_sage_asset(
                    ply_path=ply_path,
                    texture_path=texture_path,
                    output_path=shared_dir / f"{source_id}.glb",
                )

            asset_meta = shared_assets[source_id]
            dimensions = obj.get("dimensions") or {}
            native_size = asset_meta["native_size"]
            renderable_objects.append(
                {
                    "id": obj["id"],
                    "asset_path": asset_meta["asset_path"],
                    "position": [
                        float(obj["position"]["x"]),
                        float(obj["position"].get("z", 0.0)),
                        float(obj["position"]["y"]),
                    ],
                    "rotation_y_deg": float(-(obj.get("rotation") or {}).get("z", 0.0)),
                    "scale": [
                        _safe_scale(dimensions.get("width"), native_size[0]),
                        _safe_scale(dimensions.get("height"), native_size[1]),
                        _safe_scale(dimensions.get("length"), native_size[2]),
                    ],
                    "native_size": native_size,
                    "description": obj.get("description"),
                    "type": obj.get("type"),
                    "source_id": source_id,
                }
            )

        renderable_rooms: list[dict[str, object]] = []
        for room in scene_manifest["normalized"]["rooms"]:
            wall_texture = None
            walls = room.get("walls") or []
            if walls:
                wall_texture = _renderable_room_material(
                    scene_dir,
                    walls[0].get("material"),
                )
            door_renderables = []
            for door in room.get("doors") or []:
                door_renderables.append(
                    {
                        "id": door["id"],
                        "wall_id": door["wall_id"],
                        "position_on_wall": door["position_on_wall"],
                        "width": door["width"],
                        "height": door["height"],
                        "texture_path": _renderable_room_material(scene_dir, door.get("door_material")),
                    }
                )

            renderable_rooms.append(
                {
                    "id": room["id"],
                    "room_type": room.get("room_type"),
                    "dimensions": room.get("dimensions"),
                    "ceiling_height": room.get("ceiling_height"),
                    "floor_texture_path": _renderable_room_material(scene_dir, room.get("floor_material")),
                    "wall_texture_path": wall_texture,
                    "walls": room.get("walls") or [],
                    "doors": door_renderables,
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
        }
        output_dir = _sage_renderable_scene_output(scene_id)
        render_manifest_path = output_dir / "scene.json"
        _write_json(render_manifest_path, render_manifest)
        scenes.append(
            {
                "scene_id": scene_id,
                "scene_uid": scene_manifest["scene_uid"],
                "render_manifest": _repo_path(render_manifest_path),
                "object_count": len(renderable_objects),
                "room_count": len(renderable_rooms),
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

