"""Import HSM data from a local HSM project checkout.

This avoids re-downloading scene JSONs, support region annotations,
and preprocessed metadata from Hugging Face when they already exist
on disk (e.g. from running ``setup.sh`` in the HSM repo).
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

from .config import ASSETS_ROOT, DATASETS
from .download import _extracted_dir, _manifests_dir, _now_utc, write_json
from .hsm import HSM_DATASET_KEY, hsm_generated_scenes_root, hsm_support_region_root
from .preprocess import preprocess_hsm_dataset, write_dataset_catalog
from .renderable import build_hsm_renderables, write_renderable_catalog
from . import env_config


def _remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    _remove_existing(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "link":
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
    else:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _hsm_preview_rebuild_reason(*, requested: bool) -> str | None:
    if requested:
        return "requested"
    preprocessed_index = ASSETS_ROOT / "preprocessed" / HSM_DATASET_KEY / "index.json"
    if not preprocessed_index.exists():
        return "missing_preprocessed_index"
    renderable_index = ASSETS_ROOT / "renderable" / HSM_DATASET_KEY / "index.json"
    if not renderable_index.exists():
        return "missing_renderable_index"
    return None


def import_local_hsm_output(
    *,
    source: Path,
    destination_root: Path | None,
    hssd_source: Path | None,
    mode: str,
    force: bool,
    build_preview: bool,
) -> dict[str, object]:
    """Import HSM data from a local project directory.

    Expected layout under *source*::

        generated_scenes/*.json       # scene JSON files
        support_region_dataset/       # support region annotations (annot/, annot_surface/)
        data/preprocessed/            # metadata (object_categories.json, hssd_wnsynsetkey_index.json)

    If *hssd_source* is given (or ``SCENEVIEWER_HSSD_ROOT`` is set),
    the HSSD model GLBs are linked from that path instead of downloading.
    """
    spec = DATASETS[HSM_DATASET_KEY]
    resolved_source = source.resolve()
    if not resolved_source.exists():
        raise SystemExit(f"Source path does not exist: {source}")

    resolved_destination = (destination_root or spec.destination_root).resolve()

    # --- Discover source data ---
    generated_scenes_dir = resolved_source / "generated_scenes"
    if not generated_scenes_dir.is_dir():
        generated_scenes_dir = None

    # support_region_dataset/ at source/ or source/support_region_dataset/
    support_region_dir = resolved_source / "support_region_dataset"
    if not support_region_dir.is_dir():
        # Check if annot/annot_surface are directly under source
        if (resolved_source / "annot").is_dir() or (resolved_source / "annot_surface").is_dir():
            support_region_dir = resolved_source
        else:
            support_region_dir = None

    # data/preprocessed/ metadata
    preprocessed_data_dir = resolved_source / "data" / "preprocessed"
    if not preprocessed_data_dir.is_dir():
        preprocessed_data_dir = None

    if generated_scenes_dir is None and support_region_dir is None and preprocessed_data_dir is None:
        raise SystemExit(
            f"Cannot find any HSM data under {source}.\n"
            "Expected at least one of:\n"
            "  - generated_scenes/ (scene JSON files)\n"
            "  - support_region_dataset/ (annot/ and annot_surface/)\n"
            "  - data/preprocessed/ (metadata JSON files)\n\n"
            "If the HSM project is not set up yet, either:\n"
            "  1. Run the HSM setup.sh to download all data.\n"
            "  2. Use `dataset-downloader download hsm --sample-size N` to download scenes from HF."
        )

    imported: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    raw_root = resolved_destination / "source" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    # --- Import generated scenes ---
    if generated_scenes_dir is not None:
        scene_dest = hsm_generated_scenes_root(resolved_destination)
        if scene_dest.exists() and not force:
            skipped.append({
                "kind": "generated_scenes",
                "target_dir": str(scene_dest),
                "reason": "target_exists",
            })
        else:
            _link_or_copy(generated_scenes_dir, scene_dest, mode)
            imported.append({
                "kind": "generated_scenes",
                "source": str(generated_scenes_dir),
                "target": str(scene_dest),
                "mode": mode,
                "scene_count": len(list(scene_dest.rglob("*.json"))) if scene_dest.exists() else 0,
            })
    else:
        skipped.append({
            "kind": "generated_scenes",
            "reason": "not_found",
            "searched": str(resolved_source),
        })

    # --- Import support region dataset ---
    if support_region_dir is not None:
        support_dest = hsm_support_region_root(resolved_destination)
        if support_dest.exists() and not force:
            skipped.append({
                "kind": "support_region_dataset",
                "target_dir": str(support_dest),
                "reason": "target_exists",
            })
        else:
            _link_or_copy(support_region_dir, support_dest, mode)
            imported.append({
                "kind": "support_region_dataset",
                "source": str(support_region_dir),
                "target": str(support_dest),
                "mode": mode,
            })
    else:
        skipped.append({
            "kind": "support_region_dataset",
            "reason": "not_found",
            "searched": str(resolved_source),
        })

    # --- Import metadata ---
    metadata_dest = ASSETS_ROOT / HSM_DATASET_KEY / "metadata"
    if preprocessed_data_dir is not None:
        metadata_dest.mkdir(parents=True, exist_ok=True)
        for name in ("object_categories.json", "hssd_wnsynsetkey_index.json"):
            src_file = preprocessed_data_dir / name
            dst_file = metadata_dest / name
            if src_file.exists():
                if dst_file.exists() and not force:
                    skipped.append({
                        "kind": "metadata",
                        "file": name,
                        "reason": "target_exists",
                    })
                else:
                    _link_or_copy(src_file, dst_file, "copy")
                    imported.append({
                        "kind": "metadata",
                        "source": str(src_file),
                        "target": str(dst_file),
                    })
    else:
        skipped.append({
            "kind": "metadata",
            "reason": "data/preprocessed/ not found",
        })

    # --- Optionally link HSSD models ---
    hssd_root = hssd_source or env_config.get_path("SCENEVIEWER_HSSD_ROOT")
    hssd_linked = False
    if hssd_root is not None:
        hssd_resolved = hssd_root.resolve()
        hssd_objects = hssd_resolved / "objects"
        if hssd_objects.is_dir():
            hssd_dest = resolved_destination / "hssd-models"
            if hssd_dest.exists() and not force:
                skipped.append({
                    "kind": "hssd-models",
                    "target_dir": str(hssd_dest),
                    "reason": "target_exists",
                })
            else:
                _link_or_copy(hssd_resolved, hssd_dest, mode)
                hssd_linked = True
                imported.append({
                    "kind": "hssd-models",
                    "source": str(hssd_resolved),
                    "target": str(hssd_dest),
                    "mode": mode,
                })
        else:
            skipped.append({
                "kind": "hssd-models",
                "reason": "objects/ not found in HSSD source",
                "hssd_source": str(hssd_resolved),
            })
    else:
        skipped.append({
            "kind": "hssd-models",
            "reason": "no HSSD source configured (set SCENEVIEWER_HSSD_ROOT or pass --hssd-source)",
        })

    # --- Write manifest ---
    manifests_root = _manifests_dir(spec, resolved_destination)
    manifests_root.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "generated_at_utc": _now_utc(),
        "dataset": HSM_DATASET_KEY,
        "kind": "local_import",
        "source_root": str(resolved_source),
        "destination_root": str(resolved_destination),
        "mode": mode,
        "force": force,
        "build_preview": build_preview,
        "hssd_linked": hssd_linked,
        "imported_count": len(imported),
        "skipped_count": len(skipped),
        "imported": imported,
        "skipped": skipped,
    }

    manifest_path = manifests_root / "local_import.json"
    payload["manifest_path"] = str(manifest_path)
    write_json(manifest_path, payload)

    # --- Optionally build preview ---
    preview_reason = _hsm_preview_rebuild_reason(requested=build_preview)
    payload["preview_built"] = preview_reason is not None
    if preview_reason is not None:
        payload["preview_build_reason"] = preview_reason
        preprocess_index = preprocess_hsm_dataset()
        dataset_catalog = write_dataset_catalog()
        renderable_index = build_hsm_renderables()
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
