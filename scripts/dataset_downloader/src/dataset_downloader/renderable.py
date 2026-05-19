from __future__ import annotations

import base64
import json
import math
import shutil
import zipfile
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from PIL import Image
from plyfile import PlyData
from pygltflib import (
    ARRAY_BUFFER,
    ELEMENT_ARRAY_BUFFER,
    FLOAT,
    SCALAR,
    UNSIGNED_INT,
    Accessor,
    Attributes,
    Buffer,
    BufferView,
    GLTF2,
    Image as GLTFImage,
    Material,
    Mesh,
    Node,
    PbrMetallicRoughness,
    Primitive,
    Sampler,
    Scene,
    Texture,
    VEC2,
    VEC3,
)
from tqdm.auto import tqdm

from .config import DATASETS, PREPROCESSED_ROOT, RENDERABLE_ROOT, REPO_ROOT
from .front3d import (
    FRONT3D_DATASET_KEY,
    FRONT3D_LAYOUT_ZIP,
    FRONT3D_MODEL_ZIP,
    FRONT3D_TEXTURE_ZIP,
    ensure_front3d_archives,
    front3d_texture_member_candidates,
    map_front3d_shell_category,
    repo_relative_path,
    safe_front3d_name,
)
from .hsm import (
    HSM_DATASET_KEY,
    ensure_hsm_hssd_models,
    hsm_hssd_glb_path,
    repo_relative_path as hsm_repo_relative_path,
    hsm_three_quaternion_from_transform,
)


SCHEMA_VERSION = 1


def _now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _repo_path(path: Path) -> str:
    return repo_relative_path(path)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
    temp_path.replace(path)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _scene_progress(
    dataset: str,
    scene_summaries: list[dict[str, object]],
):
    return tqdm(
        scene_summaries,
        desc=f"renderable {dataset}",
        unit="scene",
        dynamic_ncols=True,
    )


def _preprocessed_index_path(dataset: str) -> Path:
    return PREPROCESSED_ROOT / dataset / "index.json"


def _renderable_index_path(dataset: str) -> Path:
    return RENDERABLE_ROOT / dataset / "index.json"


def _sage_renderable_scene_output(scene_id: str) -> Path:
    return RENDERABLE_ROOT / "sage" / scene_id


def _scenesmith_renderable_scene_output(subset: str, scene_id: str) -> Path:
    return RENDERABLE_ROOT / "scenesmith" / subset / scene_id


def _front3d_renderable_scene_output(house_id: str, scene_id: str) -> Path:
    return RENDERABLE_ROOT / FRONT3D_DATASET_KEY / safe_front3d_name(house_id) / safe_front3d_name(scene_id)


def _triangulate_polygon(indices: list[int]) -> list[tuple[int, int, int]]:
    return [(indices[0], indices[index], indices[index + 1]) for index in range(1, len(indices) - 1)]


def _axis_swap_source_to_three(vector: tuple[float, float, float] | list[float]) -> list[float]:
    return [float(vector[0]), float(vector[2]), -float(vector[1])]


def _quaternion_from_rotation_matrix(rotation: np.ndarray) -> list[float]:
    trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale

    length = math.sqrt(x * x + y * y + z * z + w * w)
    if length <= 1e-8:
        return [0.0, 0.0, 0.0, 1.0]
    return [x / length, y / length, z / length, w / length]


def _rotation_matrix_from_wxyz(quaternion: list[float]) -> np.ndarray:
    if len(quaternion) < 4:
        return np.identity(3, dtype=float)

    w, x, y, z = (float(value) for value in quaternion[:4])
    length = math.sqrt(w * w + x * x + y * y + z * z)
    if length <= 1e-8:
        return np.identity(3, dtype=float)

    w /= length
    x /= length
    y /= length
    z /= length
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=float,
    )


def _angle_axis_rotation_matrix(angle_deg: float, axis: list[float]) -> np.ndarray:
    axis_array = np.asarray(axis[:3], dtype=float)
    length = float(np.linalg.norm(axis_array))
    if length <= 1e-8 or abs(angle_deg) <= 1e-8:
        return np.identity(3, dtype=float)
    x, y, z = axis_array / length
    angle_rad = math.radians(angle_deg)
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    one_minus_cosine = 1.0 - cosine
    return np.array(
        [
            [
                cosine + x * x * one_minus_cosine,
                x * y * one_minus_cosine - z * sine,
                x * z * one_minus_cosine + y * sine,
            ],
            [
                y * x * one_minus_cosine + z * sine,
                cosine + y * y * one_minus_cosine,
                y * z * one_minus_cosine - x * sine,
            ],
            [
                z * x * one_minus_cosine - y * sine,
                z * y * one_minus_cosine + x * sine,
                cosine + z * z * one_minus_cosine,
            ],
        ],
        dtype=float,
    )


def _scenesmith_source_rotation_matrix(object_data: dict[str, object]) -> np.ndarray:
    transform = object_data.get("transform") or {}
    if not isinstance(transform, dict):
        return np.identity(3, dtype=float)

    rotation_wxyz = transform.get("rotation_wxyz")
    if isinstance(rotation_wxyz, list) and len(rotation_wxyz) >= 4:
        return _rotation_matrix_from_wxyz(rotation_wxyz)

    rotation = transform.get("rotation_angle_axis") or {}
    if not isinstance(rotation, dict):
        return np.identity(3, dtype=float)
    angle = float(rotation.get("angle_deg") or 0.0)
    axis = rotation.get("axis") or [0.0, 0.0, 1.0]
    if not isinstance(axis, list) or len(axis) < 3:
        return np.identity(3, dtype=float)
    return _angle_axis_rotation_matrix(angle, axis)


def _scenesmith_source_rotation_to_three(rotation_source: np.ndarray) -> np.ndarray:
    source_to_three = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    )
    return source_to_three @ rotation_source @ source_to_three.T


def _sage_vertices_to_three(vertices: np.ndarray) -> np.ndarray:
    # Source meshes are Z-up. Rotate them into three.js/glTF Y-up without flipping handedness.
    return np.column_stack((vertices[:, 0], vertices[:, 2], -vertices[:, 1])).astype(np.float32)


def _compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float32)
    tri_vertices = vertices[faces]
    face_normals = np.cross(
        tri_vertices[:, 1] - tri_vertices[:, 0],
        tri_vertices[:, 2] - tri_vertices[:, 0],
    )
    face_lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    valid_faces = face_lengths[:, 0] > 1e-8
    face_normals[valid_faces] /= face_lengths[valid_faces]

    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)

    normal_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    valid_vertices = normal_lengths[:, 0] > 1e-8
    normals[valid_vertices] /= normal_lengths[valid_vertices]
    normals[~valid_vertices] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return normals.astype(np.float32)


def _scene_uid_sort_key(scene_uid: str) -> tuple[str, str]:
    parts = scene_uid.split("/")
    if len(parts) >= 3:
        return parts[-2], parts[-1]
    return "", scene_uid


def _build_renderable_index(
    *,
    dataset: str,
    scenes: list[dict[str, object]],
    source_scene_count: int,
    status: str,
    shared_asset_count: int | None = None,
) -> dict[str, object]:
    index = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _now_utc(),
        "dataset": dataset,
        "status": status,
        "scene_count": len(scenes),
        "source_scene_count": source_scene_count,
        "scenes": sorted(scenes, key=lambda item: _scene_uid_sort_key(item["scene_uid"])),
    }
    if shared_asset_count is not None:
        index["shared_asset_count"] = shared_asset_count
    return index


def _write_renderable_progress(
    *,
    dataset: str,
    scenes: list[dict[str, object]],
    source_scene_count: int,
    status: str,
    shared_asset_count: int | None = None,
) -> dict[str, object]:
    index = _build_renderable_index(
        dataset=dataset,
        scenes=scenes,
        source_scene_count=source_scene_count,
        status=status,
        shared_asset_count=shared_asset_count,
    )
    _write_json(_renderable_index_path(dataset), index)
    write_renderable_catalog()
    return index


def _scenesmith_shell_id(asset_path: str) -> str:
    parts = list(Path(asset_path).parts[-3:])
    if not parts:
        return Path(asset_path).stem
    parts[-1] = Path(parts[-1]).stem
    return "__".join(parts)


def _align_to_4(size: int) -> int:
    return (size + 3) & ~3


def _append_aligned_blob(chunks: bytearray, payload: bytes) -> tuple[int, int]:
    offset = len(chunks)
    chunks.extend(payload)
    padded_size = _align_to_4(len(payload))
    if padded_size > len(payload):
        chunks.extend(b"\x00" * (padded_size - len(payload)))
    return offset, len(payload)


def _image_to_data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


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


def _safe_scale(target: float | None, native: float) -> float:
    if target is None or native <= 1e-6:
        return 1.0
    return float(target / native)


def _renderable_room_material(scene_dir: Path, material_name: str | None) -> str | None:
    if not material_name:
        return None
    material_path = scene_dir / "materials" / f"{material_name}.png"
    return _repo_path(material_path) if material_path.exists() else None


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
                    "scale": [1.0, 1.0, 1.0],
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


def write_renderable_catalog() -> dict[str, object]:
    datasets: list[dict[str, object]] = []
    for dataset in sorted(DATASETS):
        index_path = RENDERABLE_ROOT / dataset / "index.json"
        if not index_path.exists():
            continue
        index = _read_json(index_path)
        datasets.append(
            {
                "dataset": dataset,
                "scene_count": index["scene_count"],
                "index_path": _repo_path(index_path),
                "source_scene_count": index.get("source_scene_count"),
                "status": index.get("status", "ready"),
            }
        )

    catalog = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _now_utc(),
        "datasets": datasets,
    }
    _write_json(RENDERABLE_ROOT / "datasets.json", catalog)
    return catalog
