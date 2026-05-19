from __future__ import annotations

import re
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from .config import DATASETS
from .download import _extracted_dir, _manifests_dir, _now_utc, write_json
from .preprocess import preprocess_scenesmith_dataset, write_dataset_catalog
from .renderable import build_scenesmith_renderables, write_renderable_catalog

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

