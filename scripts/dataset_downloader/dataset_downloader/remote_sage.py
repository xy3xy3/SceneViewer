from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub.utils import EntryNotFoundError

from .config import DATASETS, PREPROCESSED_ROOT, RENDERABLE_ROOT
from .download import RemoteArchive, _manifests_dir, _now_utc, download_selection, write_json
from .preprocess import preprocess_sage_dataset, write_dataset_catalog
from .renderable import build_sage_renderables, write_renderable_catalog


def _normalize_scene_id(value: str) -> str:
    scene_id = value.strip()
    if scene_id.endswith(".zip"):
        scene_id = scene_id[:-4]
    if not scene_id:
        raise SystemExit("Scene id must not be empty.")
    if "/" in scene_id or "\\" in scene_id:
        raise SystemExit(
            "Scene id must be a bare SAGE scene identifier, for example "
            "`20251228_133527_layout_6b049b06`."
        )
    return scene_id


def _sage_preview_rebuild_reason(*, requested: bool) -> str | None:
    if requested:
        return "requested"

    preprocessed_index = PREPROCESSED_ROOT / "sage" / "index.json"
    if not preprocessed_index.exists():
        return "missing_preprocessed_index"

    renderable_index = RENDERABLE_ROOT / "sage" / "index.json"
    if not renderable_index.exists():
        return "missing_renderable_index"

    return None


def import_remote_sage_scenes(
    *,
    scene_ids: list[str],
    destination_root: Path | None,
    force_download: bool,
    force_extract: bool,
    build_preview: bool,
) -> dict[str, object]:
    spec = DATASETS["sage"]
    normalized_scene_ids = [_normalize_scene_id(scene_id) for scene_id in scene_ids]
    unique_scene_ids = list(dict.fromkeys(normalized_scene_ids))
    destination = (destination_root or spec.destination_root).resolve()
    manifests_root = _manifests_dir(spec, destination)
    manifests_root.mkdir(parents=True, exist_ok=True)

    selection = [
        RemoteArchive(
            dataset=spec.key,
            repo_id=spec.repo_id,
            path=f"scenes/{scene_id}.zip",
            size_bytes=None,
            subset=None,
        )
        for scene_id in unique_scene_ids
    ]

    try:
        records = download_selection(
            spec,
            destination,
            selection,
            extract=True,
            force_download=force_download,
            force_extract=force_extract,
        )
    except EntryNotFoundError as error:
        missing_path = getattr(error, "server_message", None) or str(error)
        raise SystemExit(
            "Failed to download one of the requested SAGE scene archives from "
            f"`{spec.repo_id}`. Please confirm the scene id exists under `scenes/` "
            f"on Hugging Face. Details: {missing_path}"
        ) from error

    payload: dict[str, object] = {
        "generated_at_utc": _now_utc(),
        "dataset": "sage",
        "kind": "remote_import",
        "repo_id": spec.repo_id,
        "destination_root": str(destination),
        "scene_ids": unique_scene_ids,
        "scene_count": len(unique_scene_ids),
        "force_download": force_download,
        "force_extract": force_extract,
        "build_preview": build_preview,
        "records": records,
    }

    manifest_name = (
        f"remote_import_{unique_scene_ids[0]}.json"
        if len(unique_scene_ids) == 1
        else f"remote_import_batch_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    manifest_path = manifests_root / manifest_name
    payload["manifest_path"] = str(manifest_path)
    write_json(manifest_path, payload)

    preview_rebuild_reason = _sage_preview_rebuild_reason(requested=build_preview)
    payload["preview_built"] = preview_rebuild_reason is not None
    if preview_rebuild_reason is not None:
        payload["preview_build_reason"] = preview_rebuild_reason
        preprocess_index = preprocess_sage_dataset()
        dataset_catalog = write_dataset_catalog()
        renderable_index = build_sage_renderables()
        renderable_catalog = write_renderable_catalog()
        payload["preview_build"] = {
            "preprocessed_scene_count": preprocess_index["scene_count"],
            "preprocessed_skipped_count": preprocess_index["skipped_count"],
            "renderable_scene_count": renderable_index["scene_count"],
            "datasets_in_catalog": len(dataset_catalog["datasets"]),
            "renderable_datasets_in_catalog": len(renderable_catalog["datasets"]),
        }
        write_json(manifest_path, payload)

    return payload
