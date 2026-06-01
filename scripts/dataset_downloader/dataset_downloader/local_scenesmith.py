from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from .config import DATASETS, PREPROCESSED_ROOT, RENDERABLE_ROOT
from .download import _extract_archive, _extracted_dir, _manifests_dir, _now_utc, write_json
from .preprocess import preprocess_scenesmith_dataset, write_dataset_catalog
from .renderable import build_scenesmith_renderables, write_renderable_catalog


@dataclass(frozen=True)
class _LocalSceneSmithImportSource:
    path: Path
    scene_id: str
    kind: str


def _sanitize_subset_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return sanitized or "local"


def _scenesmith_archive_suffix(path: Path) -> str | None:
    lower_name = path.name.lower()
    for suffix in (".tar", ".zip"):
        if lower_name.endswith(suffix) or f"{suffix}?" in lower_name:
            return suffix
    return None


def _is_scenesmith_archive_file(path: Path) -> bool:
    return path.is_file() and _scenesmith_archive_suffix(path) is not None


def _archive_scene_id(path: Path) -> str:
    scene_id = _sanitize_subset_name(path.stem)
    return scene_id or "scene"


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


def _is_scenesmith_archive_candidate(path: Path) -> bool:
    return _is_scenesmith_archive_file(path) and path.name.startswith("scene_")


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
) -> tuple[list[tuple[str, list[_LocalSceneSmithImportSource]]], list[dict[str, object]]]:
    resolved = source.resolve()
    if not resolved.exists():
        raise SystemExit(f"Source path does not exist: {source}")

    if _is_scenesmith_scene_dir(resolved):
        return [
            (
                _default_scenesmith_import_subset(resolved),
                [_LocalSceneSmithImportSource(resolved, resolved.name, "directory")],
            )
        ], []

    if _is_scenesmith_archive_file(resolved):
        return [
            (
                _default_scenesmith_import_subset(resolved),
                [_LocalSceneSmithImportSource(resolved, _archive_scene_id(resolved), "archive")],
            )
        ], []

    direct_scene_sources = []
    if resolved.is_dir():
        for path in sorted(resolved.iterdir()):
            candidate = path.resolve()
            if _is_scenesmith_scene_dir(candidate):
                direct_scene_sources.append(
                    _LocalSceneSmithImportSource(candidate, candidate.name, "directory")
                )
            elif _is_scenesmith_archive_candidate(candidate):
                direct_scene_sources.append(
                    _LocalSceneSmithImportSource(candidate, _archive_scene_id(candidate), "archive")
                )
    if direct_scene_sources:
        return [(_default_scenesmith_import_subset(resolved), direct_scene_sources)], []

    if not resolved.is_dir():
        raise SystemExit(
            "Expected a SceneSmith scene directory, a local SceneSmith scene archive, "
            "an experiment output directory, or an outputs root."
        )

    grouped: dict[str, list[_LocalSceneSmithImportSource]] = {}
    skipped: list[dict[str, object]] = []
    candidate_paths = [
        path.resolve()
        for path in sorted(resolved.rglob("scene_*"))
    ]
    for candidate in candidate_paths:
        if _has_scenesmith_scene_ancestor(candidate, resolved):
            continue
        if _is_scenesmith_scene_dir(candidate):
            subset_name = _default_scenesmith_import_subset(candidate)
            grouped.setdefault(subset_name, []).append(
                _LocalSceneSmithImportSource(candidate, candidate.name, "directory")
            )
            continue
        if _is_scenesmith_archive_candidate(candidate):
            subset_name = _default_scenesmith_import_subset(candidate)
            grouped.setdefault(subset_name, []).append(
                _LocalSceneSmithImportSource(candidate, _archive_scene_id(candidate), "archive")
            )
            continue
        if candidate.is_dir():
            skipped.append(
                {
                    "scene_id": candidate.name,
                    "source_dir": str(candidate),
                    "reason": "invalid_scene_dir",
                }
            )

    if grouped:
        return [
            (subset_name, scene_sources)
            for subset_name, scene_sources in sorted(grouped.items())
        ], skipped

    raise SystemExit(
        "Did not find any valid SceneSmith inputs under "
        f"{source}. A valid scene needs `package.xml` and `combined_house/`, "
        "either as a directory or inside a local tar/zip archive."
    )


def _remove_existing_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.exists():
        shutil.rmtree(path)


def _scenesmith_preview_rebuild_reason(*, requested: bool) -> str | None:
    if requested:
        return "requested"

    preprocessed_index = PREPROCESSED_ROOT / "scenesmith" / "index.json"
    if not preprocessed_index.exists():
        return "missing_preprocessed_index"

    renderable_index = RENDERABLE_ROOT / "scenesmith" / "index.json"
    if not renderable_index.exists():
        return "missing_renderable_index"

    return None


def _flatten_single_extracted_scene_dir(target_dir: Path) -> bool:
    entries = list(target_dir.iterdir())
    if len(entries) != 1:
        return False
    nested_dir = entries[0]
    if not nested_dir.is_dir() or not _is_scenesmith_scene_dir(nested_dir):
        return False
    for child in nested_dir.iterdir():
        child.rename(target_dir / child.name)
    nested_dir.rmdir()
    return True


def _extract_scenesmith_archive_to_dir(archive_path: Path, target_dir: Path) -> None:
    archive_suffix = _scenesmith_archive_suffix(archive_path)
    if archive_suffix is None:
        raise ValueError(f"Unsupported archive type for SceneSmith import: {archive_path.name}")

    _extract_archive(archive_path, target_dir, archive_suffix)
    if _is_scenesmith_scene_dir(target_dir):
        return
    if _flatten_single_extracted_scene_dir(target_dir) and _is_scenesmith_scene_dir(target_dir):
        return
    raise ValueError(
        "Archive did not extract into a valid SceneSmith scene directory. "
        "Expected `package.xml` and `combined_house/` at the root."
    )


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
                [(scene_source, scene_source.scene_id) for scene_source in scene_sources],
            )
            for subset_name, scene_sources in discovered_groups
        ]
    else:
        scene_name_counts: dict[str, int] = defaultdict(int)
        for _, grouped_scene_sources in discovered_groups:
            for scene_source in grouped_scene_sources:
                scene_name_counts[scene_source.scene_id] += 1

        scene_sources = []
        for source_subset_name, grouped_scene_sources in discovered_groups:
            for scene_source in grouped_scene_sources:
                target_scene_id = scene_source.scene_id
                if len(discovered_groups) > 1 or scene_name_counts[scene_source.scene_id] > 1:
                    target_scene_id = _sanitize_subset_name(
                        f"{source_subset_name}__{scene_source.scene_id}"
                    )
                scene_sources.append((scene_source, target_scene_id))
        import_groups = [(_sanitize_subset_name(subset), scene_sources)]
    resolved_destination = (destination_root or spec.destination_root).resolve()
    extracted_root = _extracted_dir(spec, resolved_destination)
    manifests_root = _manifests_dir(spec, resolved_destination)
    manifests_root.mkdir(parents=True, exist_ok=True)

    imported: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = list(discovery_skipped)

    for subset_name, scene_entries in import_groups:
        subset_root = extracted_root / subset_name
        subset_root.mkdir(parents=True, exist_ok=True)
        for scene_source, target_scene_id in scene_entries:
            target_dir = subset_root / target_scene_id
            if target_dir.exists() or target_dir.is_symlink():
                if not force:
                    skipped_entry = {
                        "scene_id": target_scene_id,
                        "source_scene_id": scene_source.scene_id,
                        "source_kind": scene_source.kind,
                        "subset": subset_name,
                        "source_path": str(scene_source.path),
                        "target_dir": str(target_dir),
                        "reason": "target_exists",
                    }
                    if scene_source.kind == "directory":
                        skipped_entry["source_dir"] = str(scene_source.path)
                    else:
                        skipped_entry["source_archive"] = str(scene_source.path)
                    skipped.append(
                        skipped_entry
                    )
                    continue
                _remove_existing_path(target_dir)

            try:
                if scene_source.kind == "directory":
                    if mode == "link":
                        target_dir.symlink_to(scene_source.path, target_is_directory=True)
                    else:
                        shutil.copytree(scene_source.path, target_dir)
                else:
                    _extract_scenesmith_archive_to_dir(scene_source.path, target_dir)
            except (OSError, ValueError) as error:
                _remove_existing_path(target_dir)
                skipped_entry = {
                    "scene_id": target_scene_id,
                    "source_scene_id": scene_source.scene_id,
                    "source_kind": scene_source.kind,
                    "subset": subset_name,
                    "source_path": str(scene_source.path),
                    "target_dir": str(target_dir),
                    "reason": "import_failed",
                    "error": str(error),
                }
                if scene_source.kind == "directory":
                    skipped_entry["source_dir"] = str(scene_source.path)
                else:
                    skipped_entry["source_archive"] = str(scene_source.path)
                skipped.append(
                    skipped_entry
                )
                continue

            imported_entry = {
                "scene_id": target_scene_id,
                "source_scene_id": scene_source.scene_id,
                "source_kind": scene_source.kind,
                "subset": subset_name,
                "source_path": str(scene_source.path),
                "target_dir": str(target_dir),
                "mode": mode if scene_source.kind == "directory" else "extract",
            }
            if scene_source.kind == "directory":
                imported_entry["source_dir"] = str(scene_source.path)
            else:
                imported_entry["source_archive"] = str(scene_source.path)
            imported.append(
                imported_entry
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
        "skip_existing": not force,
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

    preview_rebuild_reason = _scenesmith_preview_rebuild_reason(requested=build_preview)
    payload["preview_built"] = preview_rebuild_reason is not None
    if preview_rebuild_reason is not None:
        payload["preview_build_reason"] = preview_rebuild_reason
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
