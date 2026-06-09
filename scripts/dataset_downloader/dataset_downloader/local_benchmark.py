from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import ASSETS_ROOT
from .download import _now_utc, write_json
from .hsm import HSM_DATASET_KEY
from .local_scenesmith import import_local_scenesmith_output
from .preprocess import (
    preprocess_hsm_dataset,
    preprocess_sage_dataset,
    preprocess_scenesmith_dataset,
    write_dataset_catalog,
)
from .renderable import (
    build_hsm_renderables,
    build_sage_renderables,
    build_scenesmith_renderables,
    write_renderable_catalog,
)


_BENCHMARK_LABEL_BY_DATASET = {
    HSM_DATASET_KEY: "HSM",
    "sage": "SAGE-10k",
    "scenesmith": "SceneSmith",
}

_SUPPORTED_DATASETS = tuple(_BENCHMARK_LABEL_BY_DATASET)


@dataclass(frozen=True)
class _BenchmarkSceneResult:
    dataset: str
    label: str
    result_dir: Path
    result_name: str
    scene_json_path: Path
    payload: dict[str, object]


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
        return
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _import_annotation_sidecar(
    *,
    result_dir: Path,
    destination_root: Path,
    mode: str,
    force: bool,
) -> dict[str, object] | None:
    source_path = result_dir / "asset_annotations"
    if not source_path.exists():
        return None

    if destination_root.exists() or destination_root.is_symlink():
        if not force:
            return {
                "status": "skipped",
                "reason": "target_exists",
                "target_path": str(destination_root),
            }
    _link_or_copy(source_path, destination_root, mode)
    return {
        "status": "imported",
        "source_path": str(source_path),
        "target_path": str(destination_root),
        "mode": mode,
    }


def _coerce_benchmark_assets_root(source: Path) -> Path:
    resolved = source.resolve()
    candidate_roots = [resolved]
    if (resolved / "assets").is_dir():
        candidate_roots.insert(0, (resolved / "assets").resolve())

    for candidate in candidate_roots:
        if any((candidate / label).is_dir() for label in _BENCHMARK_LABEL_BY_DATASET.values()):
            return candidate

    raise SystemExit(
        "Could not find benchmark scene outputs under "
        f"{source}. Expected a directory containing one or more of: "
        f"{', '.join(sorted(_BENCHMARK_LABEL_BY_DATASET.values()))}."
    )


def _read_scene_payload(scene_json_path: Path) -> dict[str, object]:
    payload = json.loads(scene_json_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {scene_json_path}")
    return payload


def _discover_benchmark_results(assets_root: Path, dataset: str) -> list[_BenchmarkSceneResult]:
    label = _BENCHMARK_LABEL_BY_DATASET[dataset]
    result_root = assets_root / label
    if not result_root.is_dir():
        return []

    results: list[_BenchmarkSceneResult] = []
    for scene_json_path in sorted(result_root.glob("*/scene.json")):
        result_dir = scene_json_path.parent
        payload = _read_scene_payload(scene_json_path)
        results.append(
            _BenchmarkSceneResult(
                dataset=dataset,
                label=label,
                result_dir=result_dir,
                result_name=result_dir.name,
                scene_json_path=scene_json_path,
                payload=payload,
            )
        )
    return results


def _sanitize_subset_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return sanitized or "local"


def _existing_source_path(raw_path: str) -> Path | None:
    candidates: list[Path] = []
    raw_path = raw_path.strip()
    if raw_path:
        candidates.append(Path(raw_path))
        if "?" in raw_path:
            candidates.append(Path(raw_path.split("?", 1)[0]))

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def _benchmark_source_path(result: _BenchmarkSceneResult) -> Path:
    raw_path = result.payload.get("source_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"Missing source_path in {result.scene_json_path}")

    source_path = _existing_source_path(raw_path)
    if source_path is not None:
        return source_path

    raise ValueError(
        f"Could not resolve source_path `{raw_path}` from {result.scene_json_path}"
    )


def _scenesmith_subset_name(result: _BenchmarkSceneResult) -> str | None:
    source_subset = result.payload.get("source_subset")
    if isinstance(source_subset, str) and source_subset.strip():
        return _sanitize_subset_name(source_subset)

    if "_scene_" in result.result_name:
        prefix, _, _ = result.result_name.partition("_scene_")
        if prefix:
            return _sanitize_subset_name(prefix)

    return None


def _scenesmith_scene_id_hint(result: _BenchmarkSceneResult) -> str | None:
    source_scene_id = result.payload.get("source_scene_id")
    if isinstance(source_scene_id, int):
        return f"scene_{source_scene_id:03d}"
    if result.result_name.startswith("scene_"):
        return result.result_name
    if "_scene_" in result.result_name:
        return "scene_" + result.result_name.rsplit("_scene_", 1)[-1]
    return None


def _resolve_scenesmith_scene_source(result: _BenchmarkSceneResult) -> tuple[Path, str | None]:
    source_path = _benchmark_source_path(result)
    subset = _scenesmith_subset_name(result)

    if source_path.is_file():
        return source_path, subset

    if (source_path / "package.xml").exists() and (source_path / "combined_house").is_dir():
        return source_path, subset

    scene_id_hint = _scenesmith_scene_id_hint(result)
    if scene_id_hint is None:
        raise ValueError(
            f"Could not infer SceneSmith scene id from benchmark result {result.result_name}"
        )

    direct_candidates = [
        source_path / scene_id_hint,
        source_path / f"{scene_id_hint}.tar",
        source_path / f"{scene_id_hint}.zip",
        source_path / f"{scene_id_hint}.tar?download=true",
        source_path / f"{scene_id_hint}.zip?download=true",
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate, subset

    for candidate in sorted(source_path.glob(f"{scene_id_hint}*")):
        if candidate.is_dir() or candidate.is_file():
            return candidate, subset

    raise ValueError(
        f"Could not find a SceneSmith scene source for {result.result_name} under {source_path}"
    )


def _import_hsm_results(
    *,
    results: list[_BenchmarkSceneResult],
    assets_root: Path,
    destination_root: Path,
    mode: str,
    force: bool,
) -> dict[str, object]:
    dataset_root = destination_root / HSM_DATASET_KEY
    generated_root = dataset_root / "source" / "raw" / "generated_scenes"
    imported: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    annotation_sidecars: list[dict[str, object]] = []

    for result in results:
        source_path = _benchmark_source_path(result)
        if not source_path.is_file():
            skipped.append(
                {
                    "result": result.result_name,
                    "reason": "source_not_file",
                    "source_path": str(source_path),
                }
            )
            continue

        target_path = generated_root / source_path.name
        annotation_sidecar = _import_annotation_sidecar(
            result_dir=result.result_dir,
            destination_root=dataset_root
            / "source"
            / "benchmark_annotations"
            / target_path.stem
            / "asset_annotations",
            mode=mode,
            force=force,
        )
        if annotation_sidecar is not None:
            annotation_sidecars.append(
                {
                    "result": result.result_name,
                    "scene_id": target_path.stem,
                    **annotation_sidecar,
                }
            )
        if target_path.exists() or target_path.is_symlink():
            if not force:
                skipped.append(
                    {
                        "result": result.result_name,
                        "reason": "target_exists",
                        "source_path": str(source_path),
                        "target_path": str(target_path),
                    }
                )
                continue
        _link_or_copy(source_path, target_path, mode)
        imported.append(
            {
                "result": result.result_name,
                "source_path": str(source_path),
                "target_path": str(target_path),
                "mode": mode,
            }
        )

    benchmark_dataset_root = assets_root / HSM_DATASET_KEY
    metadata_root = benchmark_dataset_root / "metadata"
    metadata_imported: list[str] = []
    metadata_skipped: list[dict[str, object]] = []
    for metadata_name in ("annotations.csv", "object_categories.json", "hssd_wnsynsetkey_index.json"):
        source_path = metadata_root / metadata_name
        if not source_path.exists():
            continue
        target_path = dataset_root / "metadata" / metadata_name
        if target_path.exists() or target_path.is_symlink():
            if not force:
                metadata_skipped.append(
                    {
                        "file": metadata_name,
                        "reason": "target_exists",
                        "target_path": str(target_path),
                    }
                )
                continue
        _link_or_copy(source_path, target_path, "copy")
        metadata_imported.append(str(target_path))

    support_region_source = benchmark_dataset_root / "source" / "raw" / "support_region_dataset"
    support_region_target = dataset_root / "source" / "raw" / "support_region_dataset"
    support_region_status: dict[str, object] | None = None
    if support_region_source.exists():
        if support_region_target.exists() or support_region_target.is_symlink():
            if force:
                _link_or_copy(support_region_source, support_region_target, mode)
                support_region_status = {
                    "status": "imported",
                    "source_path": str(support_region_source),
                    "target_path": str(support_region_target),
                    "mode": mode,
                }
            else:
                support_region_status = {
                    "status": "skipped",
                    "reason": "target_exists",
                    "target_path": str(support_region_target),
                }
        else:
            _link_or_copy(support_region_source, support_region_target, mode)
            support_region_status = {
                "status": "imported",
                "source_path": str(support_region_source),
                "target_path": str(support_region_target),
                "mode": mode,
            }

    hssd_source = benchmark_dataset_root / "hssd-models"
    hssd_status: dict[str, object] | None = None
    if hssd_source.exists():
        hssd_target = dataset_root / "hssd-models"
        if hssd_target.exists() or hssd_target.is_symlink():
            if force:
                _link_or_copy(hssd_source, hssd_target, mode)
                hssd_status = {
                    "status": "imported",
                    "source_path": str(hssd_source),
                    "target_path": str(hssd_target),
                    "mode": mode,
                }
            else:
                hssd_status = {
                    "status": "skipped",
                    "reason": "target_exists",
                    "target_path": str(hssd_target),
                }
        else:
            _link_or_copy(hssd_source, hssd_target, mode)
            hssd_status = {
                "status": "imported",
                "source_path": str(hssd_source),
                "target_path": str(hssd_target),
                "mode": mode,
            }

    return {
        "dataset": HSM_DATASET_KEY,
        "selected_scene_count": len(results),
        "imported_scene_count": len(imported),
        "skipped_scene_count": len(skipped),
        "imported_scenes": imported,
        "skipped_scenes": skipped,
        "metadata_imported": metadata_imported,
        "metadata_skipped": metadata_skipped,
        "annotation_sidecars": annotation_sidecars,
        "support_region_dataset": support_region_status,
        "hssd_models": hssd_status,
    }


def _import_sage_results(
    *,
    results: list[_BenchmarkSceneResult],
    destination_root: Path,
    mode: str,
    force: bool,
) -> dict[str, object]:
    dataset_root = destination_root / "sage"
    extracted_root = dataset_root / "source" / "extracted"
    imported: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    annotation_sidecars: list[dict[str, object]] = []

    for result in results:
        source_path = _benchmark_source_path(result)
        if not source_path.is_dir():
            skipped.append(
                {
                    "result": result.result_name,
                    "reason": "source_not_directory",
                    "source_path": str(source_path),
                }
            )
            continue

        target_path = extracted_root / source_path.name
        annotation_sidecar = _import_annotation_sidecar(
            result_dir=result.result_dir,
            destination_root=dataset_root
            / "source"
            / "benchmark_annotations"
            / target_path.name
            / "asset_annotations",
            mode=mode,
            force=force,
        )
        if annotation_sidecar is not None:
            annotation_sidecars.append(
                {
                    "result": result.result_name,
                    "scene_id": target_path.name,
                    **annotation_sidecar,
                }
            )
        if target_path.exists() or target_path.is_symlink():
            if not force:
                skipped.append(
                    {
                        "result": result.result_name,
                        "reason": "target_exists",
                        "source_path": str(source_path),
                        "target_path": str(target_path),
                    }
                )
                continue
        _link_or_copy(source_path, target_path, mode)
        imported.append(
            {
                "result": result.result_name,
                "source_path": str(source_path),
                "target_path": str(target_path),
                "mode": mode,
            }
        )

    return {
        "dataset": "sage",
        "selected_scene_count": len(results),
        "imported_scene_count": len(imported),
        "skipped_scene_count": len(skipped),
        "imported_scenes": imported,
        "skipped_scenes": skipped,
        "annotation_sidecars": annotation_sidecars,
    }


def _import_scenesmith_results(
    *,
    results: list[_BenchmarkSceneResult],
    destination_root: Path,
    mode: str,
    force: bool,
) -> dict[str, object]:
    imported: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    seen_sources: set[tuple[str, str | None]] = set()
    annotation_sidecars: list[dict[str, object]] = []

    for result in results:
        try:
            source_path, subset = _resolve_scenesmith_scene_source(result)
        except ValueError as error:
            skipped.append(
                {
                    "result": result.result_name,
                    "reason": "source_resolution_failed",
                    "error": str(error),
                }
            )
            continue

        dedupe_key = (str(source_path), subset)
        if dedupe_key in seen_sources:
            skipped.append(
                {
                    "result": result.result_name,
                    "reason": "duplicate_source",
                    "source_path": str(source_path),
                    "subset": subset,
                }
            )
            continue
        seen_sources.add(dedupe_key)

        payload = import_local_scenesmith_output(
            source=source_path,
            subset=subset,
            destination_root=destination_root / "scenesmith",
            mode=mode,
            force=force,
            build_preview=False,
        )
        imported.append(
            {
                "result": result.result_name,
                "source_path": str(source_path),
                "subset": subset,
                "imported_scene_count": payload.get("imported_scene_count"),
                "skipped_scene_count": payload.get("skipped_scene_count"),
                "manifest_path": payload.get("manifest_path"),
            }
        )
        scene_sidecar_entries: list[dict[str, object]] = []
        for scene_key in ("imported_scenes", "skipped_scenes"):
            scene_entries = payload.get(scene_key)
            if not isinstance(scene_entries, list):
                continue
            for scene_entry in scene_entries:
                if not isinstance(scene_entry, dict):
                    continue
                target_scene_id = scene_entry.get("scene_id")
                target_subset = scene_entry.get("subset")
                if not isinstance(target_scene_id, str) or not isinstance(target_subset, str):
                    continue
                dedupe_key = (target_subset, target_scene_id)
                if dedupe_key in {
                    (entry.get("subset"), entry.get("scene_id")) for entry in scene_sidecar_entries
                }:
                    continue
                scene_sidecar_entries.append(
                    {
                        "scene_id": target_scene_id,
                        "subset": target_subset,
                    }
                )

        for scene_entry in scene_sidecar_entries:
            target_scene_id = str(scene_entry["scene_id"])
            target_subset = str(scene_entry["subset"])
            annotation_sidecar = _import_annotation_sidecar(
                result_dir=result.result_dir,
                destination_root=destination_root
                / "scenesmith"
                / "source"
                / "benchmark_annotations"
                / target_subset
                / target_scene_id
                / "asset_annotations",
                mode=mode,
                force=force,
            )
            if annotation_sidecar is not None:
                annotation_sidecars.append(
                    {
                        "result": result.result_name,
                        "scene_id": target_scene_id,
                        "subset": target_subset,
                        **annotation_sidecar,
                    }
                )

    return {
        "dataset": "scenesmith",
        "selected_scene_count": len(results),
        "imported_group_count": len(imported),
        "skipped_group_count": len(skipped),
        "imports": imported,
        "skipped": skipped,
        "annotation_sidecars": annotation_sidecars,
    }


def _build_preview_indexes(datasets: list[str]) -> dict[str, object]:
    preprocessed_counts: dict[str, int] = {}
    renderable_counts: dict[str, int] = {}

    for dataset in datasets:
        if dataset == HSM_DATASET_KEY:
            preprocessed_counts[dataset] = preprocess_hsm_dataset()["scene_count"]
        elif dataset == "sage":
            preprocessed_counts[dataset] = preprocess_sage_dataset()["scene_count"]
        elif dataset == "scenesmith":
            preprocessed_counts[dataset] = preprocess_scenesmith_dataset()["scene_count"]

    dataset_catalog = write_dataset_catalog()

    for dataset in datasets:
        if dataset == HSM_DATASET_KEY:
            renderable_counts[dataset] = build_hsm_renderables()["scene_count"]
        elif dataset == "sage":
            renderable_counts[dataset] = build_sage_renderables()["scene_count"]
        elif dataset == "scenesmith":
            renderable_counts[dataset] = build_scenesmith_renderables()["scene_count"]

    renderable_catalog = write_renderable_catalog()

    return {
        "preprocessed_scene_counts": preprocessed_counts,
        "renderable_scene_counts": renderable_counts,
        "datasets_in_catalog": len(dataset_catalog["datasets"]),
        "renderable_datasets_in_catalog": len(renderable_catalog["datasets"]),
    }


def import_local_benchmark_output(
    *,
    source: Path,
    datasets: list[str] | None,
    destination_root: Path | None,
    mode: str,
    force: bool,
    build_preview: bool,
) -> dict[str, object]:
    benchmark_assets_root = _coerce_benchmark_assets_root(source)
    selected_datasets = list(dict.fromkeys(datasets or _SUPPORTED_DATASETS))
    destination = (destination_root or ASSETS_ROOT).resolve()

    if build_preview and destination != ASSETS_ROOT.resolve():
        raise SystemExit(
            "--build-preview currently requires the default repo assets root "
            f"({ASSETS_ROOT}) because preprocess/renderable indexes are repo-global."
        )

    results_by_dataset = {
        dataset: _discover_benchmark_results(benchmark_assets_root, dataset)
        for dataset in selected_datasets
    }

    imported_datasets: dict[str, dict[str, object]] = {}
    for dataset, results in results_by_dataset.items():
        if dataset == HSM_DATASET_KEY:
            imported_datasets[dataset] = _import_hsm_results(
                results=results,
                assets_root=benchmark_assets_root,
                destination_root=destination,
                mode=mode,
                force=force,
            )
        elif dataset == "sage":
            imported_datasets[dataset] = _import_sage_results(
                results=results,
                destination_root=destination,
                mode=mode,
                force=force,
            )
        elif dataset == "scenesmith":
            imported_datasets[dataset] = _import_scenesmith_results(
                results=results,
                destination_root=destination,
                mode=mode,
                force=force,
            )

    payload: dict[str, object] = {
        "generated_at_utc": _now_utc(),
        "kind": "benchmark_local_import",
        "source_root": str(source.resolve()),
        "benchmark_assets_root": str(benchmark_assets_root),
        "destination_root": str(destination),
        "datasets": selected_datasets,
        "mode": mode,
        "force": force,
        "build_preview": build_preview,
        "imports": imported_datasets,
    }

    manifests_root = destination / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        manifests_root
        / f"benchmark_import_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    payload["manifest_path"] = str(manifest_path)
    write_json(manifest_path, payload)

    if build_preview:
        payload["preview_build"] = _build_preview_indexes(selected_datasets)
        write_json(manifest_path, payload)

    return payload
