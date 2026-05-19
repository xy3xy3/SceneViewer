from __future__ import annotations

from .common import *

def write_dataset_catalog() -> dict[str, object]:
    datasets: list[dict[str, object]] = []
    for dataset in sorted(DATASETS):
        index_path = PREPROCESSED_ROOT / dataset / "index.json"
        if not index_path.exists():
            continue
        index = json.loads(index_path.read_text())
        datasets.append(
            {
                "dataset": index["dataset"],
                "scene_count": index["scene_count"],
                "skipped_count": index["skipped_count"],
                "index_path": _repo_path(index_path),
            }
        )

    catalog = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _now_utc(),
        "datasets": datasets,
    }
    _write_json(PREPROCESSED_ROOT / "datasets.json", catalog)
    return catalog
