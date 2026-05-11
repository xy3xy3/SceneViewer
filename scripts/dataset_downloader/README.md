# Dataset Downloader

Small `uv`-managed helper project for sampling large Hugging Face scene datasets into
stable local folders that SceneViewer can preprocess later.

Right now the tool focuses on download orchestration only:

- build a remote archive index
- choose a reproducible random sample
- download only the selected scene archives
- extract them into a consistent local layout
- write JSON manifests for later preprocessing / web preview

## Datasets

- `sage` -> `nvidia/SAGE-10k`
- `scenesmith` -> `nepfaff/scenesmith-example-scenes`

SceneSmith defaults to `Room` + `House` only. That keeps the first pass focused on
the full-method scenes and avoids accidentally pulling the `NotGenerated` subset,
whose upstream assets are CC BY-NC 4.0.

## Output Layout

The tool writes into the repo-level asset folders you requested:

- `/data/L202500274/SceneViewer/assets/sage`
- `/data/L202500274/SceneViewer/assets/scenesmith`

Inside each dataset root it uses the same structure:

```text
assets/<dataset>/
├── manifests/
│   ├── remote_index.json
│   └── download_sample_<timestamp>.json
└── source/
    ├── archives/
    └── extracted/
```

That gives us a clean separation between:

- upstream raw archives
- extracted raw scenes
- later normalized/web-ready outputs

## Setup

```bash
cd /data/L202500274/SceneViewer/scripts/dataset_downloader
uv sync
```

## Usage

Build a remote index only:

```bash
uv run dataset-downloader index sage
uv run dataset-downloader index scenesmith
```

Preview a sample selection without downloading:

```bash
uv run dataset-downloader download sage --sample-size 5 --dry-run
uv run dataset-downloader download scenesmith --sample-size 4 --dry-run
```

Download and extract a small sample:

```bash
uv run dataset-downloader download sage --sample-size 5 --seed 7
uv run dataset-downloader download scenesmith --sample-size 4 --seed 7
```

Keep SceneSmith samples small enough for a first local preview pass:

```bash
uv run dataset-downloader download scenesmith \
  --subset Room \
  --sample-size 2 \
  --max-size-gib 1.0 \
  --seed 7
```

Include extra SceneSmith subsets explicitly:

```bash
uv run dataset-downloader download scenesmith \
  --subset Room \
  --subset House \
  --subset NoCritic \
  --sample-size 6 \
  --dry-run
```

Include every SceneSmith subset:

```bash
uv run dataset-downloader download scenesmith --subset all --sample-size 8 --dry-run
```

## Preprocess

Write normalized scene manifests to:

- `/data/L202500274/SceneViewer/assets/preprocessed/sage`
- `/data/L202500274/SceneViewer/assets/preprocessed/scenesmith`

Run per dataset:

```bash
uv run dataset-downloader preprocess sage
uv run dataset-downloader preprocess scenesmith
```

Or regenerate everything plus a shared dataset catalog:

```bash
uv run dataset-downloader preprocess all
```

The preprocessor is tolerant of partially downloaded scenes. If a scene is missing
critical files, it is recorded under `skipped_scenes` in the dataset `index.json`
instead of aborting the whole run.

## Notes For The Next Step

The download manifest is intentionally structured so the preprocessing stage can read
one JSON file and know:

- which dataset each scene came from
- which subset it belongs to
- which remote archive produced it
- where the extracted scene lives locally

That should make the later "normalize to a shared web preview schema" step much
simpler.
