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

from ..config import DATASETS, PREPROCESSED_ROOT, RENDERABLE_ROOT, REPO_ROOT
from ..front3d import (
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
from ..hsm import (
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


def _safe_scale(target: float | None, native: float) -> float:
    if target is None or native <= 1e-6:
        return 1.0
    return float(target / native)


def _renderable_room_material(scene_dir: Path, material_name: str | None) -> str | None:
    if not material_name:
        return None
    material_path = scene_dir / "materials" / f"{material_name}.png"
    return _repo_path(material_path) if material_path.exists() else None



__all__ = [name for name in globals() if not name.startswith("__")]
