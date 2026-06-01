from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

from .config import DATASETS, DatasetSpec, SCENESMITH_ALL_SUBSETS
from .download import (
    balanced_sample,
    build_download_manifest,
    build_hsm_hssd_download_manifest,
    build_index_manifest,
    download_hsm_hssd_assets,
    download_hsm_selection,
    download_selection,
    filter_archives_by_size,
    list_remote_archives,
    write_json,
    _resolve_download_output,
    _resolve_index_output,
    _resolve_paths,
    _selected_subsets,
)
from .hsm import HSM_DATASET_KEY, HSM_HSSD_ROOT, hsm_generated_scenes_root
from .local_sceneweaver import import_local_sceneweaver_output
from .local_scenesmith import import_local_scenesmith_output
from .preprocess import (
    preprocess_3dfront_dataset,
    preprocess_hsm_dataset,
    preprocess_sage_dataset,
    preprocess_sceneweaver_dataset,
    preprocess_scenesmith_dataset,
    write_dataset_catalog,
)
from .renderable import (
    build_3dfront_renderables,
    build_hsm_renderables,
    build_sage_renderables,
    build_sceneweaver_renderables,
    build_scenesmith_renderables,
    write_renderable_catalog,
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
            "A SceneSmith scene directory, a local SceneSmith tar/zip scene archive, "
            "or an experiment output directory containing `scene_*` subdirectories. "
            "Passing an outputs root imports all valid nested results."
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
        help=(
            "Whether to symlink local scene directories or copy them into the repo. "
            "Archive inputs are always extracted."
        ),
    )
    import_scenesmith_parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace existing imported scene directories with the same target path.",
    )
    import_scenesmith_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip scenes whose target directories already exist. "
            "When omitted, existing targets are replaced from the source outputs."
        ),
    )
    import_scenesmith_parser.add_argument(
        "--build-preview",
        action="store_true",
        help="Also refresh SceneSmith preprocessed and renderable assets after import.",
    )
    import_sceneweaver_parser = subparsers.add_parser(
        "import-sceneweaver-local",
        help="Import a local SceneWeaver output directory into assets/sceneweaver/source/extracted/.",
    )
    import_sceneweaver_parser.add_argument(
        "source",
        type=Path,
        help=(
            "A SceneWeaver run directory, or a root containing multiple run outputs. "
            "Passing a parent outputs directory imports all valid nested results."
        ),
    )
    import_sceneweaver_parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help=(
            "Subset name to create under assets/sceneweaver/source/extracted/. "
            "Defaults to a subset inferred from the source path."
        ),
    )
    import_sceneweaver_parser.add_argument(
        "--destination",
        type=Path,
        default=DATASETS["sceneweaver"].destination_root,
        help="Target SceneWeaver dataset root. Defaults to assets/sceneweaver.",
    )
    import_sceneweaver_parser.add_argument(
        "--mode",
        choices=["link", "copy"],
        default="link",
        help="Whether to symlink the local run directories or copy them into the repo.",
    )
    import_sceneweaver_parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace existing imported run directories with the same target path.",
    )
    import_sceneweaver_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip scenes whose target directories already exist. "
            "When omitted, existing targets are replaced from the source outputs."
        ),
    )
    import_sceneweaver_parser.add_argument(
        "--build-preview",
        action="store_true",
        help="Also refresh SceneWeaver preprocessed and renderable assets after import.",
    )

    args = parser.parse_args()
    if args.command == "import-scenesmith-local":
        replace_existing = args.force and not args.skip_existing
        payload = import_local_scenesmith_output(
            source=args.source,
            subset=args.subset,
            destination_root=args.destination,
            mode=args.mode,
            force=replace_existing,
            build_preview=args.build_preview,
        )
        print(json.dumps(payload, indent=2))
        return
    if args.command == "import-sceneweaver-local":
        replace_existing = args.force and not args.skip_existing
        payload = import_local_sceneweaver_output(
            source=args.source,
            subset=args.subset,
            destination_root=args.destination,
            mode=args.mode,
            force=replace_existing,
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
            elif target == "sceneweaver":
                preprocess_sceneweaver_dataset(scene_limit=args.limit)
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
            elif target == "sceneweaver":
                build_sceneweaver_renderables(scene_limit=args.limit)
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
