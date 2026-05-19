from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import tarfile
import zipfile

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.hf_api import RepoFile

from .config import DATASETS, DatasetSpec, SCENESMITH_ALL_SUBSETS
from .hsm import (
    HSM_DATASET_KEY,
    HSM_HSSD_DECOMPOSED_REPO_ID,
    HSM_HSSD_MODELS_REPO_ID,
    HSM_HSSD_ROOT,
    hsm_generated_scenes_root,
    hsm_scene_model_ids,
    hsm_support_region_root,
    load_hsm_scene,
)


@dataclass(frozen=True)
class RemoteArchive:
    dataset: str
    repo_id: str
    path: str
    size_bytes: int | None
    subset: str | None

    @property
    def scene_id(self) -> str:
        return Path(self.path).stem

    @property
    def archive_name(self) -> str:
        return Path(self.path).name


def _now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _human_gib(num_bytes: int | None) -> float | None:
    if num_bytes is None:
        return None
    return round(num_bytes / (1024**3), 3)


def _archives_dir(spec: DatasetSpec, destination: Path) -> Path:
    return destination / "source" / "archives"


def _extracted_dir(spec: DatasetSpec, destination: Path) -> Path:
    return destination / "source" / "extracted"


def _manifests_dir(spec: DatasetSpec, destination: Path) -> Path:
    return destination / "manifests"


def _matches_spec(spec: DatasetSpec, path: str) -> bool:
    if not path.endswith(spec.archive_suffix):
        return False
    if spec.archive_prefix is None:
        return path.count("/") == 1
    return path.startswith(spec.archive_prefix)


def _subset_from_path(spec: DatasetSpec, path: str) -> str | None:
    if spec.key != "scenesmith":
        return None
    return path.split("/", 1)[0]


def list_remote_archives(
    api: HfApi,
    spec: DatasetSpec,
    subsets: set[str] | None,
) -> list[RemoteArchive]:
    archives: list[RemoteArchive] = []

    for node in api.list_repo_tree(
        repo_id=spec.repo_id,
        repo_type="dataset",
        recursive=True,
        expand=False,
    ):
        if not isinstance(node, RepoFile):
            continue

        if not _matches_spec(spec, node.path):
            continue

        subset = _subset_from_path(spec, node.path)
        if subsets and subset not in subsets:
            continue

        archives.append(
            RemoteArchive(
                dataset=spec.key,
                repo_id=spec.repo_id,
                path=node.path,
                size_bytes=node.size,
                subset=subset,
            )
        )

    archives.sort(key=lambda item: item.path)
    return archives


def summarize_archives(archives: Iterable[RemoteArchive]) -> dict[str, object]:
    archives = list(archives)
    total_size = sum(item.size_bytes or 0 for item in archives)
    per_subset: dict[str, dict[str, object]] = {}

    grouped: dict[str, list[RemoteArchive]] = defaultdict(list)
    for item in archives:
        grouped[item.subset or "all"].append(item)

    for subset, items in sorted(grouped.items()):
        subset_total = sum(entry.size_bytes or 0 for entry in items)
        per_subset[subset] = {
            "count": len(items),
            "total_gib": _human_gib(subset_total),
        }

    return {
        "count": len(archives),
        "total_gib": _human_gib(total_size),
        "per_subset": per_subset,
    }


def balanced_sample(
    archives: list[RemoteArchive],
    sample_size: int,
    seed: int,
) -> list[RemoteArchive]:
    if sample_size >= len(archives):
        return list(archives)

    rng = random.Random(seed)
    grouped: dict[str, list[RemoteArchive]] = defaultdict(list)
    for item in archives:
        grouped[item.subset or "all"].append(item)

    if len(grouped) == 1:
        return rng.sample(archives, sample_size)

    ordered_keys = list(grouped)
    rng.shuffle(ordered_keys)
    for key in ordered_keys:
        rng.shuffle(grouped[key])

    requested_per_bucket = max(1, math.floor(sample_size / len(grouped)))
    selected: list[RemoteArchive] = []
    leftovers: list[RemoteArchive] = []

    for key in ordered_keys:
        bucket = grouped[key]
        take = min(requested_per_bucket, len(bucket))
        selected.extend(bucket[:take])
        leftovers.extend(bucket[take:])

    if len(selected) < sample_size:
        selected.extend(rng.sample(leftovers, sample_size - len(selected)))
    elif len(selected) > sample_size:
        selected = rng.sample(selected, sample_size)

    selected.sort(key=lambda item: item.path)
    return selected


def filter_archives_by_size(
    archives: list[RemoteArchive],
    *,
    max_size_gib: float | None,
) -> list[RemoteArchive]:
    if max_size_gib is None:
        return archives

    max_size_bytes = max_size_gib * (1024**3)
    return [
        item
        for item in archives
        if item.size_bytes is not None and item.size_bytes <= max_size_bytes
    ]


def _selected_subsets(spec: DatasetSpec, requested: list[str] | None) -> set[str] | None:
    if spec.key != "scenesmith":
        return None

    if requested:
        if "all" in requested:
            return set(SCENESMITH_ALL_SUBSETS)
        return set(requested)

    return set(spec.default_subsets)


def build_index_manifest(
    spec: DatasetSpec,
    subsets: set[str] | None,
    archives: list[RemoteArchive],
    destination: Path,
) -> dict[str, object]:
    return {
        "generated_at_utc": _now_utc(),
        "dataset": spec.key,
        "repo_id": spec.repo_id,
        "destination_root": str(destination),
        "subsets": sorted(subsets) if subsets else [],
        "summary": summarize_archives(archives),
        "archives": [asdict(item) for item in archives],
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + os.linesep)


def _archive_local_path(destination: Path, remote: RemoteArchive) -> Path:
    return _archives_dir(DATASETS[remote.dataset], destination) / remote.path


def _download_hsm_support_assets(
    *,
    spec: DatasetSpec,
    destination: Path,
    model_ids: set[str],
    available_support_paths: set[str],
    force_download: bool,
) -> list[dict[str, object]]:
    support_records: list[dict[str, object]] = []
    raw_root = destination / "source" / "raw"
    for model_id in sorted(model_ids):
        for remote_path in (
            f"support_region_dataset/annot/{model_id}.glb",
            f"support_region_dataset/annot_surface/{model_id}.glb",
        ):
            if remote_path not in available_support_paths:
                continue
            downloaded_path = Path(
                hf_hub_download(
                    repo_id=spec.repo_id,
                    repo_type="dataset",
                    filename=remote_path,
                    local_dir=raw_root,
                    force_download=force_download,
                )
            )
            support_records.append(
                {
                    "dataset": spec.key,
                    "kind": "support_region_asset",
                    "model_id": model_id,
                    "remote_path": remote_path,
                    "local_path": str(downloaded_path),
                }
            )
    return support_records


def download_hsm_selection(
    spec: DatasetSpec,
    destination: Path,
    selection: list[RemoteArchive],
    *,
    force_download: bool,
) -> list[dict[str, object]]:
    destination.mkdir(parents=True, exist_ok=True)
    raw_root = destination / "source" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    hsm_generated_scenes_root(destination).mkdir(parents=True, exist_ok=True)
    hsm_support_region_root(destination).mkdir(parents=True, exist_ok=True)
    _manifests_dir(spec, destination).mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    model_ids: set[str] = set()

    for remote in selection:
        downloaded_path = Path(
            hf_hub_download(
                repo_id=spec.repo_id,
                repo_type="dataset",
                filename=remote.path,
                local_dir=raw_root,
                force_download=force_download,
            )
        )
        scene_payload = load_hsm_scene(downloaded_path)
        model_ids.update(hsm_scene_model_ids(scene_payload))
        results.append(
            {
                "dataset": remote.dataset,
                "kind": "scene_json",
                "scene_id": remote.scene_id,
                "remote_path": remote.path,
                "archive_size_bytes": remote.size_bytes,
                "local_path": str(downloaded_path),
                "status": "downloaded",
            }
        )

    available_support_paths = {
        node.path
        for node in HfApi().list_repo_tree(
            repo_id=spec.repo_id,
            repo_type="dataset",
            recursive=True,
            expand=False,
        )
        if isinstance(node, RepoFile) and node.path.startswith("support_region_dataset/")
    }
    results.extend(
        _download_hsm_support_assets(
            spec=spec,
            destination=destination,
            model_ids=model_ids,
            available_support_paths=available_support_paths,
            force_download=force_download,
        )
    )
    return results


def _extract_path(destination: Path, remote: RemoteArchive) -> Path:
    root = _extracted_dir(DATASETS[remote.dataset], destination)
    if remote.dataset == "sage":
        return root / remote.scene_id
    assert remote.subset is not None
    return root / remote.subset / remote.scene_id


def _safe_extract_tar(archive_path: Path, target_dir: Path) -> None:
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            member_path = target_dir / member.name
            if not member_path.resolve().is_relative_to(target_dir.resolve()):
                raise ValueError(f"Refusing to extract path outside target dir: {member.name}")
        archive.extractall(target_dir)


def _extract_archive(archive_path: Path, target_dir: Path, suffix: str) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    if suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(target_dir)
        return
    if suffix == ".tar":
        _safe_extract_tar(archive_path, target_dir)
        return
    raise ValueError(f"Unsupported archive type: {suffix}")


def download_selection(
    spec: DatasetSpec,
    destination: Path,
    selection: list[RemoteArchive],
    *,
    extract: bool,
    force_download: bool,
    force_extract: bool,
) -> list[dict[str, object]]:
    destination.mkdir(parents=True, exist_ok=True)
    _archives_dir(spec, destination).mkdir(parents=True, exist_ok=True)
    _extracted_dir(spec, destination).mkdir(parents=True, exist_ok=True)
    _manifests_dir(spec, destination).mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []

    for remote in selection:
        local_archive = _archive_local_path(destination, remote)
        local_archive.parent.mkdir(parents=True, exist_ok=True)

        downloaded_path = Path(
            hf_hub_download(
                repo_id=spec.repo_id,
                repo_type="dataset",
                filename=remote.path,
                local_dir=_archives_dir(spec, destination),
                force_download=force_download,
            )
        )

        extract_dir = _extract_path(destination, remote)
        extracted = False
        if extract:
            if extract_dir.exists() and any(extract_dir.iterdir()):
                if force_extract:
                    shutil.rmtree(extract_dir)
                else:
                    results.append(
                        {
                            "dataset": remote.dataset,
                            "subset": remote.subset,
                            "scene_id": remote.scene_id,
                            "remote_path": remote.path,
                            "archive_size_bytes": remote.size_bytes,
                            "local_archive_path": str(downloaded_path),
                            "extract_dir": str(extract_dir),
                            "extracted": False,
                            "status": "skipped_existing_extract",
                        }
                    )
                    continue

            _extract_archive(downloaded_path, extract_dir, spec.archive_suffix)
            extracted = True

        results.append(
            {
                "dataset": remote.dataset,
                "subset": remote.subset,
                "scene_id": remote.scene_id,
                "remote_path": remote.path,
                "archive_size_bytes": remote.size_bytes,
                "local_archive_path": str(downloaded_path),
                "extract_dir": str(extract_dir) if extract else None,
                "extracted": extracted,
                "status": "downloaded",
            }
        )

    return results


def build_download_manifest(
    spec: DatasetSpec,
    destination: Path,
    subsets: set[str] | None,
    selection: list[RemoteArchive],
    sample_size: int,
    seed: int,
    dry_run: bool,
    records: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "generated_at_utc": _now_utc(),
        "dataset": spec.key,
        "repo_id": spec.repo_id,
        "destination_root": str(destination),
        "subsets": sorted(subsets) if subsets else [],
        "sample_size_requested": sample_size,
        "sample_size_selected": len(selection),
        "seed": seed,
        "dry_run": dry_run,
        "summary": summarize_archives(selection),
        "selection": [asdict(item) for item in selection],
        "records": records,
    }


def _collect_local_hsm_model_ids(source_root: Path) -> set[str]:
    model_ids: set[str] = set()
    if not source_root.exists():
        return model_ids
    for scene_path in sorted(source_root.rglob("*.json")):
        try:
            scene_payload = load_hsm_scene(scene_path)
        except json.JSONDecodeError:
            continue
        model_ids.update(hsm_scene_model_ids(scene_payload))
    return model_ids


def _hsm_glb_patterns_for_model_ids(model_ids: set[str]) -> list[str]:
    patterns: list[str] = []
    for model_id in sorted(model_ids):
        if "_part_" in model_id:
            base_id = model_id.split("_part_", 1)[0]
            patterns.append(f"objects/decomposed/{base_id}/{model_id}.glb")
        else:
            patterns.append(f"objects/{model_id[0]}/{model_id}.glb")
    return patterns


def build_hsm_hssd_download_manifest(
    *,
    destination_root: Path,
    mode: str,
    model_ids: set[str],
    include_decomposed: bool,
    full_objects: bool,
    dry_run: bool,
    max_workers: int,
    records: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "generated_at_utc": _now_utc(),
        "dataset": HSM_DATASET_KEY,
        "kind": "hssd_models",
        "destination_root": str(destination_root),
        "mode": mode,
        "model_id_count": len(model_ids),
        "include_decomposed": include_decomposed,
        "full_objects": full_objects,
        "dry_run": dry_run,
        "max_workers": max_workers,
        "model_ids": sorted(model_ids),
        "records": records,
    }


def download_hsm_hssd_assets(
    *,
    destination_root: Path,
    source_scene_root: Path,
    include_decomposed: bool,
    full_objects: bool,
    dry_run: bool,
    force_download: bool,
    max_workers: int,
) -> dict[str, object]:
    destination_root.mkdir(parents=True, exist_ok=True)

    model_ids = _collect_local_hsm_model_ids(source_scene_root)
    if not full_objects and not model_ids:
        raise SystemExit(
            "No local HSM scene JSONs were found to infer HSSD model ids. "
            "Run `dataset-downloader download hsm ...` first, or use `hsm-hssd --full-objects`."
        )

    records: list[dict[str, object]] = []
    object_patterns = ["objects/**/*"] if full_objects else _hsm_glb_patterns_for_model_ids(model_ids)

    object_result = snapshot_download(
        repo_id=HSM_HSSD_MODELS_REPO_ID,
        repo_type="dataset",
        local_dir=destination_root,
        allow_patterns=object_patterns,
        force_download=force_download,
        max_workers=max_workers,
        dry_run=dry_run,
    )
    records.append(
        {
            "repo_id": HSM_HSSD_MODELS_REPO_ID,
            "target_dir": str(destination_root),
            "pattern_count": len(object_patterns),
            "patterns": object_patterns if len(object_patterns) <= 200 else object_patterns[:200] + ["..."],
                "result": (
                [item.filename for item in object_result]
                if isinstance(object_result, list)
                else str(object_result)
            ),
        }
    )

    if include_decomposed:
        decomposed_patterns = ["objects/decomposed/**/*_part_*.glb"]
        decomposed_result = snapshot_download(
            repo_id=HSM_HSSD_DECOMPOSED_REPO_ID,
            repo_type="dataset",
            local_dir=destination_root,
            allow_patterns=decomposed_patterns,
            ignore_patterns=["objects/decomposed/**/*_part.*.glb"],
            force_download=force_download,
            max_workers=max_workers,
            dry_run=dry_run,
        )
        records.append(
            {
                "repo_id": HSM_HSSD_DECOMPOSED_REPO_ID,
                "target_dir": str(destination_root),
                "pattern_count": len(decomposed_patterns),
                "patterns": decomposed_patterns,
                "result": (
                    [item.filename for item in decomposed_result]
                    if isinstance(decomposed_result, list)
                    else str(decomposed_result)
                ),
            }
        )

    mode = "full_objects" if full_objects else "referenced_objects"
    return build_hsm_hssd_download_manifest(
        destination_root=destination_root,
        mode=mode,
        model_ids=model_ids,
        include_decomposed=include_decomposed,
        full_objects=full_objects,
        dry_run=dry_run,
        max_workers=max_workers,
        records=records,
    )


def _resolve_paths(args: argparse.Namespace, spec: DatasetSpec) -> tuple[Path, Path]:
    destination = (args.destination or spec.destination_root).resolve()
    manifests_dir = _manifests_dir(spec, destination)
    return destination, manifests_dir


def _resolve_index_output(
    args: argparse.Namespace,
    spec: DatasetSpec,
    manifests_dir: Path,
) -> Path:
    if args.output:
        return args.output.resolve()
    return manifests_dir / "remote_index.json"


def _resolve_download_output(
    args: argparse.Namespace,
    spec: DatasetSpec,
    manifests_dir: Path,
) -> Path:
    if args.manifest:
        return args.manifest.resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return manifests_dir / f"download_sample_{stamp}.json"

