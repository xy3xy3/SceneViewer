from __future__ import annotations

import shutil
from pathlib import Path

from .config import DATASETS, PREPROCESSED_ROOT, RENDERABLE_ROOT
from .download import _extracted_dir, _manifests_dir, _now_utc, write_json
from .hssd import download_hssd_habitat_metadata
from .preprocess import preprocess_hssd_dataset, write_dataset_catalog
from .renderable import build_hssd_renderables, write_renderable_catalog


def _remove_existing_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.exists():
        shutil.rmtree(path)


def _hssd_preview_rebuild_reason(*, requested: bool) -> str | None:
    if requested:
        return "requested"

    preprocessed_index = PREPROCESSED_ROOT / "hssd" / "index.json"
    if not preprocessed_index.exists():
        return "missing_preprocessed_index"

    renderable_index = RENDERABLE_ROOT / "hssd" / "index.json"
    if not renderable_index.exists():
        return "missing_renderable_index"

    return None


def _copy_or_link_path(*, source: Path, target: Path, mode: str) -> None:
    if mode == "link":
        target.symlink_to(source, target_is_directory=source.is_dir())
        return
    if source.is_dir():
        shutil.copytree(source, target)
        return
    shutil.copy2(source, target)


def import_local_hssd_output(
    *,
    source: Path,
    destination_root: Path | None,
    mode: str,
    force: bool,
    build_preview: bool,
    sync_habitat_metadata: bool,
    metadata_force_download: bool,
    metadata_max_workers: int,
) -> dict[str, object]:
    spec = DATASETS["hssd"]
    resolved_source = source.resolve()
    if not resolved_source.exists():
        raise SystemExit(f"Source path does not exist: {source}")
    if not resolved_source.is_dir():
        raise SystemExit("Expected an HSSD source root directory containing `stages/`.")

    stages_root = resolved_source / "stages"
    if not stages_root.is_dir():
        raise SystemExit(
            f"Expected {resolved_source} to contain a `stages/` directory with stage GLBs."
        )

    resolved_destination = (destination_root or spec.destination_root).resolve()
    extracted_root = _extracted_dir(spec, resolved_destination)
    manifests_root = _manifests_dir(spec, resolved_destination)
    extracted_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    imported_assets: list[dict[str, object]] = []
    skipped_assets: list[dict[str, object]] = []
    staged_entries = ["stages", "objects", "support-surfaces"]
    present_entries = [
        entry_name
        for entry_name in staged_entries
        if (resolved_source / entry_name).exists()
    ]

    for entry_name in present_entries:
        source_path = resolved_source / entry_name
        target_path = extracted_root / entry_name
        if target_path.exists() or target_path.is_symlink():
            if not force:
                skipped_assets.append(
                    {
                        "entry": entry_name,
                        "source_path": str(source_path),
                        "target_path": str(target_path),
                        "reason": "target_exists",
                    }
                )
                continue
            _remove_existing_path(target_path)

        try:
            _copy_or_link_path(source=source_path, target=target_path, mode=mode)
        except OSError as error:
            _remove_existing_path(target_path)
            skipped_assets.append(
                {
                    "entry": entry_name,
                    "source_path": str(source_path),
                    "target_path": str(target_path),
                    "reason": "import_failed",
                    "error": str(error),
                }
            )
            continue

        imported_assets.append(
            {
                "entry": entry_name,
                "source_path": str(source_path),
                "target_path": str(target_path),
                "mode": mode,
            }
        )

    stage_count = len(list(stages_root.glob("*.glb")))
    stage_ids = sorted(path.stem for path in stages_root.glob("*.glb"))
    payload: dict[str, object] = {
        "generated_at_utc": _now_utc(),
        "dataset": "hssd",
        "kind": "local_import",
        "source_root": str(resolved_source),
        "destination_root": str(resolved_destination),
        "mode": mode,
        "force": force,
        "skip_existing": not force,
        "build_preview": build_preview,
        "stage_count": stage_count,
        "imported_asset_count": len(imported_assets),
        "skipped_asset_count": len(skipped_assets),
        "imported_assets": imported_assets,
        "skipped_assets": skipped_assets,
    }

    if sync_habitat_metadata:
        payload["habitat_metadata_sync"] = download_hssd_habitat_metadata(
            destination_root=resolved_destination,
            scene_ids=stage_ids,
            force_download=metadata_force_download,
            max_workers=metadata_max_workers,
        )

    manifest_path = manifests_root / "local_import.json"
    payload["manifest_path"] = str(manifest_path)
    write_json(manifest_path, payload)

    preview_rebuild_reason = _hssd_preview_rebuild_reason(requested=build_preview)
    payload["preview_built"] = preview_rebuild_reason is not None
    if preview_rebuild_reason is not None:
        payload["preview_build_reason"] = preview_rebuild_reason
        preprocess_index = preprocess_hssd_dataset()
        dataset_catalog = write_dataset_catalog()
        renderable_index = build_hssd_renderables()
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
