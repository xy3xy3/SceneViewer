from __future__ import annotations

from .common import *

def write_renderable_catalog() -> dict[str, object]:
    datasets: list[dict[str, object]] = []
    for dataset in sorted(DATASETS):
        index_path = RENDERABLE_ROOT / dataset / "index.json"
        if not index_path.exists():
            continue
        index = _read_json(index_path)
        datasets.append(
            {
                "dataset": dataset,
                "scene_count": index["scene_count"],
                "index_path": _repo_path(index_path),
                "source_scene_count": index.get("source_scene_count"),
                "status": index.get("status", "ready"),
            }
        )

    catalog = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _now_utc(),
        "datasets": datasets,
    }
    _write_json(RENDERABLE_ROOT / "datasets.json", catalog)
    return catalog
