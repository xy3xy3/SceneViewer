from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import DATASETS, PREPROCESSED_ROOT, RENDERABLE_ROOT, REPO_ROOT
from .hsm import HSM_DATASET_KEY
from .preprocess import (
    preprocess_hsm_dataset,
    preprocess_sage_dataset,
    preprocess_sceneweaver_dataset,
    preprocess_scenesmith_dataset,
    write_dataset_catalog,
)
from .renderable import (
    build_hsm_renderables,
    build_sage_renderables,
    build_sceneweaver_renderables,
    build_scenesmith_renderables,
    write_renderable_catalog,
)

SCENE_MANAGED_DATASETS = {"hsm", "sage", "scenesmith", "sceneweaver"}


@dataclass(frozen=True)
class SceneRecord:
    dataset: str
    scene_id: str
    scene_uid: str
    subset: str | None
    description: str | None
    title: str | None
    scene_manifest_path: Path
    scene_manifest: dict[str, object]


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _repo_path_to_absolute(path: str | None) -> Path | None:
    if not isinstance(path, str) or not path:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _remove_existing(path: Path) -> bool:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return True
    if path.exists():
        shutil.rmtree(path)
        return True
    return False


def _preprocessed_index_path(dataset: str) -> Path:
    return PREPROCESSED_ROOT / dataset / "index.json"


def _renderable_index_path(dataset: str) -> Path:
    return RENDERABLE_ROOT / dataset / "index.json"


def _load_scene_records(dataset: str) -> list[SceneRecord]:
    index_path = _preprocessed_index_path(dataset)
    if not index_path.exists():
        raise SystemExit(
            f"Missing {index_path.relative_to(REPO_ROOT)}. "
            f"Run `dataset-downloader preprocess {dataset}` first."
        )

    index = _read_json(index_path)
    records: list[SceneRecord] = []
    for summary in index.get("scenes", []):
        if not isinstance(summary, dict):
            continue
        manifest_ref = summary.get("scene_manifest")
        manifest_path = _repo_path_to_absolute(manifest_ref if isinstance(manifest_ref, str) else None)
        if manifest_path is None or not manifest_path.exists():
            continue
        manifest = _read_json(manifest_path)
        records.append(
            SceneRecord(
                dataset=dataset,
                scene_id=str(summary.get("scene_id") or manifest.get("scene_id") or manifest_path.parent.name),
                scene_uid=str(summary.get("scene_uid") or manifest.get("scene_uid") or manifest_path.parent.name),
                subset=summary.get("subset") if isinstance(summary.get("subset"), str) else None,
                description=summary.get("description") if isinstance(summary.get("description"), str) else None,
                title=summary.get("title") if isinstance(summary.get("title"), str) else None,
                scene_manifest_path=manifest_path,
                scene_manifest=manifest,
            )
        )
    return records


def list_local_scenes(
    *,
    dataset: str,
    subset: str | None = None,
    query: str | None = None,
) -> dict[str, object]:
    records = _load_scene_records(dataset)
    normalized_query = query.strip().lower() if isinstance(query, str) and query.strip() else None

    scenes: list[dict[str, object]] = []
    for record in records:
        if subset and record.subset != subset:
            continue
        if normalized_query:
            haystacks = [
                record.scene_id,
                record.scene_uid,
                record.subset or "",
                record.title or "",
                record.description or "",
            ]
            if all(normalized_query not in value.lower() for value in haystacks):
                continue
        scenes.append(
            {
                "scene_id": record.scene_id,
                "scene_uid": record.scene_uid,
                "subset": record.subset,
                "title": record.title,
                "description": record.description,
                "scene_manifest": str(record.scene_manifest_path.relative_to(REPO_ROOT)),
            }
        )

    return {
        "dataset": dataset,
        "subset": subset,
        "query": normalized_query,
        "scene_count": len(scenes),
        "scenes": scenes,
    }


def _scene_key(record: SceneRecord) -> str:
    if record.subset:
        return f"{record.subset}/{record.scene_id}"
    return record.scene_id


def _match_scene_record(
    records: list[SceneRecord],
    scene_ref: str,
    subset: str | None,
) -> SceneRecord:
    normalized_ref = scene_ref.strip()
    matches = [
        record
        for record in records
        if (subset is None or record.subset == subset)
        and normalized_ref in {record.scene_uid, record.scene_id, _scene_key(record)}
    ]
    if not matches:
        raise SystemExit(
            f"No scene matched `{scene_ref}` in dataset `{records[0].dataset if records else 'unknown'}`."
        )
    if len(matches) > 1:
        suggestions = ", ".join(sorted(record.scene_uid for record in matches))
        raise SystemExit(
            f"`{scene_ref}` is ambiguous. Use one of the exact scene_uids instead: {suggestions}"
        )
    return matches[0]


def _record_source_paths(record: SceneRecord) -> list[Path]:
    manifest = record.scene_manifest
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    asset_paths: list[Path] = []

    if record.dataset == HSM_DATASET_KEY:
        scene_json = _repo_path_to_absolute(source.get("scene_json") if isinstance(source, dict) else None)
        if scene_json is not None:
            asset_paths.append(scene_json)
        return asset_paths

    extracted_dir = _repo_path_to_absolute(source.get("extracted_dir") if isinstance(source, dict) else None)
    if extracted_dir is not None:
        asset_paths.append(extracted_dir)
    return asset_paths


def _run_dataset_rebuild(dataset: str) -> tuple[dict[str, object], dict[str, object]]:
    if dataset == HSM_DATASET_KEY:
        preprocess_index = preprocess_hsm_dataset()
        renderable_index = build_hsm_renderables()
    elif dataset == "sage":
        preprocess_index = preprocess_sage_dataset()
        renderable_index = build_sage_renderables()
    elif dataset == "sceneweaver":
        preprocess_index = preprocess_sceneweaver_dataset()
        renderable_index = build_sceneweaver_renderables()
    elif dataset == "scenesmith":
        preprocess_index = preprocess_scenesmith_dataset()
        renderable_index = build_scenesmith_renderables()
    else:
        raise SystemExit(
            f"`remove-scenes` currently supports: {', '.join(sorted(SCENE_MANAGED_DATASETS))}."
        )

    write_dataset_catalog()
    write_renderable_catalog()
    return preprocess_index, renderable_index


def remove_local_scenes(
    *,
    dataset: str,
    scene_refs: list[str],
    subset: str | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    if dataset not in SCENE_MANAGED_DATASETS:
        raise SystemExit(
            f"`remove-scenes` currently supports: {', '.join(sorted(SCENE_MANAGED_DATASETS))}."
        )
    if not scene_refs:
        raise SystemExit("Provide at least one scene id or scene_uid to remove.")

    records = _load_scene_records(dataset)
    selected: list[SceneRecord] = []
    seen_uids: set[str] = set()
    for scene_ref in scene_refs:
        record = _match_scene_record(records, scene_ref, subset)
        if record.scene_uid in seen_uids:
            continue
        seen_uids.add(record.scene_uid)
        selected.append(record)

    removal_plan: list[dict[str, object]] = []
    for record in selected:
        source_paths = _record_source_paths(record)
        removal_plan.append(
            {
                "scene_id": record.scene_id,
                "scene_uid": record.scene_uid,
                "subset": record.subset,
                "source_paths": [
                    str(path.relative_to(REPO_ROOT)) if path.is_absolute() and path.is_relative_to(REPO_ROOT) else str(path)
                    for path in source_paths
                ],
                "preprocessed_dir": str(record.scene_manifest_path.parent.relative_to(REPO_ROOT)),
            }
        )

    if dry_run:
        return {
            "dataset": dataset,
            "dry_run": True,
            "subset": subset,
            "scene_count": len(selected),
            "scenes": removal_plan,
            "rebuild_performed": False,
        }

    removed_paths: list[str] = []
    missing_paths: list[str] = []
    for record in selected:
        for path in _record_source_paths(record):
            if _remove_existing(path):
                removed_paths.append(
                    str(path.relative_to(REPO_ROOT)) if path.is_absolute() and path.is_relative_to(REPO_ROOT) else str(path)
                )
            else:
                missing_paths.append(
                    str(path.relative_to(REPO_ROOT)) if path.is_absolute() and path.is_relative_to(REPO_ROOT) else str(path)
                )

    preprocess_index, renderable_index = _run_dataset_rebuild(dataset)
    return {
        "dataset": dataset,
        "dry_run": False,
        "subset": subset,
        "scene_count": len(selected),
        "scenes": removal_plan,
        "removed_paths": removed_paths,
        "missing_paths": missing_paths,
        "rebuild_performed": True,
        "preprocessed_scene_count": preprocess_index.get("scene_count"),
        "renderable_scene_count": renderable_index.get("scene_count"),
    }
