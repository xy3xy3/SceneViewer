from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import tarfile
import zipfile

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile

from .config import DATASETS, DatasetSpec, SCENESMITH_ALL_SUBSETS


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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dataset-downloader",
        description=(
            "Sample and download SceneViewer preview data from Hugging Face datasets."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    dataset_parser(subparsers)

    args = parser.parse_args()
    spec = DATASETS[args.dataset]
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
