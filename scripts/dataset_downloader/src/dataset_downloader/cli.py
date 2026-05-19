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
from .preprocess import (
    preprocess_hsm_dataset,
    preprocess_3dfront_dataset,
    preprocess_sage_dataset,
    preprocess_scenesmith_dataset,
    write_dataset_catalog,
)
from .renderable import (
    build_hsm_renderables,
    build_3dfront_renderables,
    build_sage_renderables,
    build_scenesmith_renderables,
    write_renderable_catalog,
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


def dataset_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("dataset", choices=sorted(DATASETS))
    common.add_argument(
        "--subset",
        action="append",
        dest="subsets",
        help=(
            "Subset(s) to include for SceneSmith. Repeat the flag to select more than one. "
            "Use --subset all to include every subset."
        ),
    )
    common.add_argument(
        "--destination",
        type=Path,
        help="Override the default dataset output directory.",
    )

    index_parser = subparsers.add_parser(
        "index",
        parents=[common],
        help="Build a remote archive index without downloading files.",
    )
    index_parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path for the generated index manifest JSON.",
    )

    download_parser = subparsers.add_parser(
        "download",
        parents=[common],
        help="Randomly sample remote archives and download them locally.",
    )
    download_parser.add_argument(
        "--sample-size",
        type=int,
        required=True,
        help="How many scene archives to select.",
    )
    download_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible sampling.",
    )
    download_parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional output path for the download manifest JSON.",
    )
    download_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the sample and write the manifest, but skip downloads.",
    )
    download_parser.add_argument(
        "--max-size-gib",
        type=float,
        help="Only sample archives whose compressed size is at or below this limit.",
    )
    download_parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download archive files only and skip extraction.",
    )
    download_parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even when the archive already exists locally.",
    )
    download_parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Delete an existing extract directory and unpack the archive again.",
    )

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Normalize downloaded scenes into a shared manifest format for web preview.",
    )
    preprocess_parser.add_argument(
        "dataset",
        choices=[*sorted(DATASETS), "all"],
        help="Which dataset to preprocess.",
    )
    preprocess_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of scenes to preprocess for fast local iteration.",
    )

    renderable_parser = subparsers.add_parser(
        "renderable",
        help="Build web-renderable scene assets and manifests from preprocessed data.",
    )
    renderable_parser.add_argument(
        "dataset",
        choices=[*sorted(DATASETS), "all"],
        help="Which dataset to turn into renderable assets.",
    )
    renderable_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of scenes to convert into renderable assets.",
    )

    hsm_hssd_parser = subparsers.add_parser(
        "hsm-hssd",
        help="Download HSSD GLB assets needed for HSM rendering.",
    )
    hsm_hssd_parser.add_argument(
        "--destination",
        type=Path,
        default=HSM_HSSD_ROOT,
        help="Target directory for HSSD assets. Defaults to assets/hsm/hssd-models.",
    )
    hsm_hssd_parser.add_argument(
        "--source-scenes",
        type=Path,
        default=hsm_generated_scenes_root(DATASETS[HSM_DATASET_KEY].destination_root),
        help="Directory containing local HSM scene JSONs used to infer referenced model ids.",
    )
    hsm_hssd_parser.add_argument(
        "--full-objects",
        action="store_true",
        help="Download the full HSSD objects tree instead of only mesh ids referenced by local HSM scenes.",
    )
    hsm_hssd_parser.add_argument(
        "--include-decomposed",
        action="store_true",
        help="Also download decomposed part meshes from hssd/hssd-hab.",
    )
    hsm_hssd_parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even when files already exist locally.",
    )
    hsm_hssd_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the matching files without actually downloading them.",
    )
    hsm_hssd_parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum concurrent download workers passed to huggingface_hub.snapshot_download.",
    )
    hsm_hssd_parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional output path for the generated HSSD download manifest JSON.",
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


def _sanitize_subset_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return sanitized or "local"


def _default_scenesmith_import_subset(source: Path) -> str:
    parent_name = source.parent.name
    grandparent_name = source.parent.parent.name

    if source.name.startswith("scene_"):
        if re.fullmatch(r"\d{2}-\d{2}-\d{2}", parent_name):
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", grandparent_name):
                return _sanitize_subset_name(f"local-{grandparent_name}-{parent_name}")
            return _sanitize_subset_name(f"local-{parent_name}")
        if parent_name:
            return _sanitize_subset_name(f"local-{parent_name}")

    if re.fullmatch(r"\d{2}-\d{2}-\d{2}", source.name):
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", parent_name):
            return _sanitize_subset_name(f"local-{parent_name}-{source.name}")

    return _sanitize_subset_name(f"local-{source.name}")


def _is_scenesmith_scene_dir(path: Path) -> bool:
    return path.is_dir() and (path / "package.xml").exists() and (path / "combined_house").is_dir()


def _has_scenesmith_scene_ancestor(path: Path, stop_at: Path) -> bool:
    for parent in path.parents:
        if parent == stop_at:
            return False
        if _is_scenesmith_scene_dir(parent):
            return True
    return False


def _discover_local_scenesmith_scene_dirs(source: Path) -> list[Path]:
    resolved = source.resolve()
    if not resolved.exists():
        raise SystemExit(f"Source path does not exist: {source}")

    if _is_scenesmith_scene_dir(resolved):
        return [resolved]

    if resolved.is_dir():
        scene_dirs = [
            path.resolve()
            for path in sorted(resolved.iterdir())
            if _is_scenesmith_scene_dir(path)
        ]
        if scene_dirs:
            return scene_dirs

    raise SystemExit(
        "Expected a SceneSmith scene directory or an experiment output directory containing "
        "`scene_*` children with `package.xml` and `combined_house/`."
    )


def _discover_local_scenesmith_import_groups(
    source: Path,
) -> tuple[list[tuple[str, list[Path]]], list[dict[str, object]]]:
    resolved = source.resolve()
    if not resolved.exists():
        raise SystemExit(f"Source path does not exist: {source}")

    if _is_scenesmith_scene_dir(resolved):
        return [(_default_scenesmith_import_subset(resolved), [resolved])], []

    direct_scene_dirs = (
        [
            path.resolve()
            for path in sorted(resolved.iterdir())
            if _is_scenesmith_scene_dir(path)
        ]
        if resolved.is_dir()
        else []
    )
    if direct_scene_dirs:
        return [(_default_scenesmith_import_subset(resolved), direct_scene_dirs)], []

    if not resolved.is_dir():
        raise SystemExit(
            "Expected a SceneSmith scene directory, an experiment output directory, or an outputs root."
        )

    grouped: dict[str, list[Path]] = {}
    skipped: list[dict[str, object]] = []
    candidate_dirs = [
        path.resolve()
        for path in sorted(resolved.rglob("scene_*"))
        if path.is_dir()
    ]
    for candidate in candidate_dirs:
        if _has_scenesmith_scene_ancestor(candidate, resolved):
            continue
        if _is_scenesmith_scene_dir(candidate):
            subset_name = _default_scenesmith_import_subset(candidate)
            grouped.setdefault(subset_name, []).append(candidate)
            continue
        skipped.append(
            {
                "scene_id": candidate.name,
                "source_dir": str(candidate),
                "reason": "invalid_scene_dir",
            }
        )

    if grouped:
        return [
            (subset_name, scene_dirs)
            for subset_name, scene_dirs in sorted(grouped.items())
        ], skipped

    raise SystemExit(
        "Did not find any valid SceneSmith `scene_*` directories under "
        f"{source}. A valid scene needs `package.xml` and `combined_house/`."
    )


def _remove_existing_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.exists():
        shutil.rmtree(path)


def import_local_scenesmith_output(
    *,
    source: Path,
    subset: str | None,
    destination_root: Path | None,
    mode: str,
    force: bool,
    build_preview: bool,
) -> dict[str, object]:
    spec = DATASETS["scenesmith"]
    resolved_source = source.resolve()
    discovered_groups, discovery_skipped = _discover_local_scenesmith_import_groups(
        resolved_source
    )
    if subset is None:
        import_groups = [
            (
                subset_name,
                [(scene_dir, scene_dir.name) for scene_dir in scene_dirs],
            )
            for subset_name, scene_dirs in discovered_groups
        ]
    else:
        scene_name_counts: dict[str, int] = defaultdict(int)
        for _, grouped_scene_dirs in discovered_groups:
            for scene_dir in grouped_scene_dirs:
                scene_name_counts[scene_dir.name] += 1

        scene_dirs = []
        for source_subset_name, grouped_scene_dirs in discovered_groups:
            for scene_dir in grouped_scene_dirs:
                target_scene_id = scene_dir.name
                if len(discovered_groups) > 1 or scene_name_counts[scene_dir.name] > 1:
                    target_scene_id = _sanitize_subset_name(
                        f"{source_subset_name}__{scene_dir.name}"
                    )
                scene_dirs.append((scene_dir, target_scene_id))
        import_groups = [(_sanitize_subset_name(subset), scene_dirs)]
    resolved_destination = (destination_root or spec.destination_root).resolve()
    extracted_root = _extracted_dir(spec, resolved_destination)
    manifests_root = _manifests_dir(spec, resolved_destination)
    manifests_root.mkdir(parents=True, exist_ok=True)

    imported: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = list(discovery_skipped)

    for subset_name, scene_entries in import_groups:
        subset_root = extracted_root / subset_name
        subset_root.mkdir(parents=True, exist_ok=True)
        for scene_dir, target_scene_id in scene_entries:
            target_dir = subset_root / target_scene_id
            if target_dir.exists() or target_dir.is_symlink():
                if not force:
                    skipped.append(
                        {
                            "scene_id": target_scene_id,
                            "source_scene_id": scene_dir.name,
                            "subset": subset_name,
                            "source_dir": str(scene_dir),
                            "target_dir": str(target_dir),
                            "reason": "target_exists",
                        }
                    )
                    continue
                _remove_existing_path(target_dir)

            try:
                if mode == "link":
                    target_dir.symlink_to(scene_dir, target_is_directory=True)
                else:
                    shutil.copytree(scene_dir, target_dir)
            except OSError as error:
                skipped.append(
                    {
                        "scene_id": target_scene_id,
                        "source_scene_id": scene_dir.name,
                        "subset": subset_name,
                        "source_dir": str(scene_dir),
                        "target_dir": str(target_dir),
                        "reason": "import_failed",
                        "error": str(error),
                    }
                )
                continue

            imported.append(
                {
                    "scene_id": target_scene_id,
                    "source_scene_id": scene_dir.name,
                    "subset": subset_name,
                    "source_dir": str(scene_dir),
                    "target_dir": str(target_dir),
                    "mode": mode,
                }
            )

    subset_names = [subset_name for subset_name, _ in import_groups]

    payload: dict[str, object] = {
        "generated_at_utc": _now_utc(),
        "dataset": "scenesmith",
        "kind": "local_import",
        "source_root": str(resolved_source),
        "destination_root": str(resolved_destination),
        "subsets": subset_names,
        "subset": subset_names[0] if len(subset_names) == 1 else None,
        "mode": mode,
        "force": force,
        "build_preview": build_preview,
        "imported_scene_count": len(imported),
        "skipped_scene_count": len(skipped),
        "imported_scenes": imported,
        "skipped_scenes": skipped,
    }

    manifest_name = (
        f"local_import_{subset_names[0]}.json"
        if len(subset_names) == 1
        else f"local_import_batch_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    manifest_path = manifests_root / manifest_name
    payload["manifest_path"] = str(manifest_path)
    write_json(manifest_path, payload)

    if build_preview:
        preprocess_index = preprocess_scenesmith_dataset()
        dataset_catalog = write_dataset_catalog()
        renderable_index = build_scenesmith_renderables()
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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dataset-downloader",
        description=(
            "Sample and download SceneViewer preview data from Hugging Face datasets."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    dataset_parser(subparsers)
    import_scenesmith_parser = subparsers.add_parser(
        "import-scenesmith-local",
        help="Import a local SceneSmith output directory into assets/scenesmith/source/extracted/.",
    )
    import_scenesmith_parser.add_argument(
        "source",
        type=Path,
        help=(
            "A SceneSmith scene directory, or an experiment output directory containing "
            "`scene_*` subdirectories. Passing an outputs root imports all valid nested results."
        ),
    )
    import_scenesmith_parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help=(
            "Subset name to create under assets/scenesmith/source/extracted/. "
            "Defaults to a subset inferred from the source path."
        ),
    )
    import_scenesmith_parser.add_argument(
        "--destination",
        type=Path,
        default=DATASETS["scenesmith"].destination_root,
        help="Target SceneSmith dataset root. Defaults to assets/scenesmith.",
    )
    import_scenesmith_parser.add_argument(
        "--mode",
        choices=["link", "copy"],
        default="link",
        help="Whether to symlink the local scene directories or copy them into the repo.",
    )
    import_scenesmith_parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace existing imported scene directories with the same target path.",
    )
    import_scenesmith_parser.add_argument(
        "--build-preview",
        action="store_true",
        help="Also refresh SceneSmith preprocessed and renderable assets after import.",
    )

    args = parser.parse_args()
    if args.command == "import-scenesmith-local":
        payload = import_local_scenesmith_output(
            source=args.source,
            subset=args.subset,
            destination_root=args.destination,
            mode=args.mode,
            force=args.force,
            build_preview=args.build_preview,
        )
        print(json.dumps(payload, indent=2))
        return

    if args.command == "hsm-hssd":
        destination_root = args.destination.resolve()
        source_scene_root = args.source_scenes.resolve()
        payload = download_hsm_hssd_assets(
            destination_root=destination_root,
            source_scene_root=source_scene_root,
            include_decomposed=args.include_decomposed,
            full_objects=args.full_objects,
            dry_run=args.dry_run,
            force_download=args.force_download,
            max_workers=args.max_workers,
        )
        output = args.manifest.resolve() if args.manifest else destination_root / "manifests" / "hssd_download.json"
        write_json(output, payload)
        print(
            json.dumps(
                {
                    "dataset": HSM_DATASET_KEY,
                    "kind": "hssd_models",
                    "mode": payload["mode"],
                    "model_id_count": payload["model_id_count"],
                    "include_decomposed": payload["include_decomposed"],
                    "full_objects": payload["full_objects"],
                    "dry_run": payload["dry_run"],
                    "output": str(output),
                    "destination": str(destination_root),
                    "license_note": (
                        "You must accept the gated Hugging Face licenses for "
                        "hssd/hssd-models and hssd/hssd-hab, and authenticate with `hf auth login`."
                    ),
                },
                indent=2,
            )
        )
        return

    if args.command == "preprocess":
        targets = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
        for target in targets:
            if target == "3dfront":
                preprocess_3dfront_dataset(scene_limit=args.limit)
            elif target == HSM_DATASET_KEY:
                preprocess_hsm_dataset(scene_limit=args.limit)
            elif target == "sage":
                preprocess_sage_dataset(scene_limit=args.limit)
            elif target == "scenesmith":
                preprocess_scenesmith_dataset(scene_limit=args.limit)
        catalog = write_dataset_catalog()
        print(json.dumps(catalog, indent=2))
        return

    if args.command == "renderable":
        targets = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
        for target in targets:
            if target == "3dfront":
                build_3dfront_renderables(scene_limit=args.limit)
            elif target == HSM_DATASET_KEY:
                build_hsm_renderables(scene_limit=args.limit)
            elif target == "sage":
                build_sage_renderables(scene_limit=args.limit)
            elif target == "scenesmith":
                build_scenesmith_renderables(scene_limit=args.limit)
        catalog = write_renderable_catalog()
        print(json.dumps(catalog, indent=2))
        return

    spec = DATASETS[args.dataset]
    if not spec.supports_remote_download:
        raise SystemExit(
            f"{spec.key} is a manual-download dataset. Place the required archives in assets/ "
            "and use `dataset-downloader preprocess` / `dataset-downloader renderable` instead."
        )
    subsets = _selected_subsets(spec, args.subsets)
    destination, manifests_dir = _resolve_paths(args, spec)

    api = HfApi()
    archives = list_remote_archives(api, spec, subsets)

    if args.command == "index":
        payload = build_index_manifest(spec, subsets, archives, destination)
        output = _resolve_index_output(args, spec, manifests_dir)
        write_json(output, payload)
        print(
            json.dumps(
                {
                    "dataset": spec.key,
                    "repo_id": spec.repo_id,
                    "count": payload["summary"]["count"],
                    "output": str(output),
                    "subsets": payload["subsets"],
                },
                indent=2,
            )
        )
        return

    if args.sample_size <= 0:
        raise SystemExit("--sample-size must be greater than zero.")

    filtered_archives = filter_archives_by_size(
        archives,
        max_size_gib=args.max_size_gib,
    )
    if not filtered_archives:
        raise SystemExit("No archives matched the requested filters.")

    if args.sample_size > len(filtered_archives):
        raise SystemExit(
            f"--sample-size {args.sample_size} exceeds the {len(filtered_archives)} "
            "archives that matched the requested filters."
        )

    selection = balanced_sample(filtered_archives, args.sample_size, args.seed)
    records: list[dict[str, object]] = []
    if not args.dry_run:
        if spec.key == HSM_DATASET_KEY:
            records = download_hsm_selection(
                spec,
                destination,
                selection,
                force_download=args.force_download,
            )
        else:
            records = download_selection(
                spec,
                destination,
                selection,
                extract=not args.no_extract,
                force_download=args.force_download,
                force_extract=args.force_extract,
            )

    payload = build_download_manifest(
        spec,
        destination,
        subsets,
        selection,
        args.sample_size,
        args.seed,
        args.dry_run,
        records,
    )
    output = _resolve_download_output(args, spec, manifests_dir)
    write_json(output, payload)

    summary = {
        "dataset": spec.key,
        "repo_id": spec.repo_id,
        "sample_size_selected": payload["sample_size_selected"],
        "seed": args.seed,
        "output": str(output),
        "destination": str(destination),
        "subsets": payload["subsets"],
    }
    if args.max_size_gib is not None:
        summary["max_size_gib"] = args.max_size_gib
    if spec.license_note:
        summary["license_note"] = spec.license_note
    print(json.dumps(summary, indent=2))
