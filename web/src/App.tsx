import { useEffect, useMemo, useState } from "react";
import {
  Boxes,
  Database,
  DoorOpen,
  Image as ImageIcon,
  Layers3,
  ScrollText,
} from "lucide-react";
import { ScenePreviewCanvas } from "./components/ScenePreviewCanvas";
import { fetchRepoJson, toRepoAssetUrl } from "./lib/repoAssets";
import type {
  DatasetCatalog,
  DatasetCatalogEntry,
  DatasetIndex,
  NormalizedObject,
  NormalizedRoom,
  SceneManifest,
  SceneSummary,
} from "./types";
import "./index.css";

function formatDatasetLabel(dataset: string): string {
  return dataset === "sage" ? "SAGE" : "SceneSmith";
}

function formatSceneLabel(scene: SceneSummary): string {
  if (scene.subset) {
    return `${scene.subset} / ${scene.scene_id}`;
  }
  return scene.title || scene.scene_id;
}

function collectPreviewImages(scene: SceneManifest | null): string[] {
  return (scene?.display.preview_images ?? [])
    .map((path) => toRepoAssetUrl(path))
    .filter((value): value is string => Boolean(value));
}

function summarizeObjectTypes(objects: NormalizedObject[]) {
  const counts = new Map<string, number>();
  for (const object of objects) {
    const key = object.type || object.object_type || "unknown";
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1]);
}

function sourceAssetEntries(scene: SceneManifest | null) {
  if (!scene) {
    return [];
  }

  return Object.entries(scene.assets)
    .map(([key, value]) => {
      if (Array.isArray(value)) {
        return [key, `${value.length} files`] as const;
      }
      return [key, value] as const;
    })
    .filter(([, value]) => Boolean(value));
}

function roomSubtitle(room: NormalizedRoom, dataset: SceneManifest["dataset"]): string {
  if (dataset === "sage") {
    const width = room.dimensions?.width;
    const length = room.dimensions?.length;
    const objectCount = room.object_ids?.length ?? 0;
    return `${room.room_type || "room"} · ${width ?? "?"}m x ${length ?? "?"}m · ${objectCount} objects`;
  }

  const objectCount = room.object_count ?? room.objects?.length ?? 0;
  return `${objectCount} objects · ${room.floor_plan_assets?.wall_gltfs?.length ?? 0} wall meshes`;
}

export default function App() {
  const [catalog, setCatalog] = useState<DatasetCatalog | null>(null);
  const [datasetIndices, setDatasetIndices] = useState<Record<string, DatasetIndex>>({});
  const [sceneCache, setSceneCache] = useState<Record<string, SceneManifest>>({});
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [selectedSceneUid, setSelectedSceneUid] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadCatalog() {
      try {
        setLoading(true);
        const nextCatalog = await fetchRepoJson<DatasetCatalog>("assets/preprocessed/datasets.json");
        if (cancelled) {
          return;
        }
        setCatalog(nextCatalog);

        const indices = await Promise.all(
          nextCatalog.datasets.map(async (entry) => [
            entry.dataset,
            await fetchRepoJson<DatasetIndex>(entry.index_path),
          ]),
        );

        if (cancelled) {
          return;
        }

        const nextIndices = Object.fromEntries(indices);
        setDatasetIndices(nextIndices);

        const defaultDataset = nextCatalog.datasets[0]?.dataset ?? "";
        const defaultScene = nextIndices[defaultDataset]?.scenes[0]?.scene_uid ?? "";
        setSelectedDataset(defaultDataset);
        setSelectedSceneUid(defaultScene);
        setError(null);
      } catch (loadError) {
        if (!cancelled) {
          const message =
            loadError instanceof Error ? loadError.message : "Failed to load dataset catalog";
          setError(message);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void loadCatalog();

    return () => {
      cancelled = true;
    };
  }, []);

  const selectedDatasetEntry = catalog?.datasets.find(
    (entry) => entry.dataset === selectedDataset,
  ) as DatasetCatalogEntry | undefined;

  const selectedDatasetIndex = datasetIndices[selectedDataset];
  const selectedSceneSummary = selectedDatasetIndex?.scenes.find(
    (scene) => scene.scene_uid === selectedSceneUid,
  );

  useEffect(() => {
    if (!selectedSceneSummary) {
      return;
    }

    const summary = selectedSceneSummary;
    if (sceneCache[summary.scene_uid]) {
      return;
    }

    let cancelled = false;
    async function loadScene() {
      try {
        const manifest = await fetchRepoJson<SceneManifest>(summary.scene_manifest);
        if (cancelled) {
          return;
        }
        setSceneCache((current) => ({
          ...current,
          [summary.scene_uid]: manifest,
        }));
      } catch (loadError) {
        if (!cancelled) {
          const message =
            loadError instanceof Error ? loadError.message : "Failed to load scene manifest";
          setError(message);
        }
      }
    }

    void loadScene();

    return () => {
      cancelled = true;
    };
  }, [sceneCache, selectedSceneSummary]);

  const selectedScene = selectedSceneSummary
    ? sceneCache[selectedSceneSummary.scene_uid] ?? null
    : null;

  const previewImages = useMemo(() => collectPreviewImages(selectedScene), [selectedScene]);
  const objectTypeSummary = useMemo(
    () => summarizeObjectTypes(selectedScene?.normalized.objects ?? []),
    [selectedScene],
  );
  const assetEntries = useMemo(() => sourceAssetEntries(selectedScene), [selectedScene]);

  function handleDatasetChange(dataset: string) {
    setSelectedDataset(dataset);
    const nextScene = datasetIndices[dataset]?.scenes[0]?.scene_uid ?? "";
    setSelectedSceneUid(nextScene);
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <Layers3 size={18} />
          </div>
          <div>
            <h1>SceneViewer</h1>
            <p>Preview preprocessed SAGE and SceneSmith scenes directly from repo assets.</p>
          </div>
        </div>

        <div className="toolbar">
          <label className="select-shell">
            <span>Dataset</span>
            <select
              value={selectedDataset}
              onChange={(event) => handleDatasetChange(event.target.value)}
              disabled={loading || !catalog?.datasets.length}
            >
              {catalog?.datasets.map((entry) => (
                <option key={entry.dataset} value={entry.dataset}>
                  {formatDatasetLabel(entry.dataset)} ({entry.scene_count})
                </option>
              ))}
            </select>
          </label>

          <label className="select-shell select-shell-scene">
            <span>Scene</span>
            <select
              value={selectedSceneUid}
              onChange={(event) => setSelectedSceneUid(event.target.value)}
              disabled={loading || !(selectedDatasetIndex?.scenes.length ?? 0)}
            >
              {selectedDatasetIndex?.scenes.map((scene) => (
                <option key={scene.scene_uid} value={scene.scene_uid}>
                  {formatSceneLabel(scene)}
                </option>
              ))}
            </select>
          </label>
        </div>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <main className="workspace">
        <section className="preview-panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">{selectedScene?.dataset ? formatDatasetLabel(selectedScene.dataset) : "Dataset"}</p>
              <h2>{selectedScene?.display.title || selectedSceneSummary?.title || "Choose a scene"}</h2>
              <p className="panel-subtitle">
                {selectedScene?.display.subtitle ||
                  selectedScene?.description ||
                  "Select a preprocessed scene to inspect its structure and assets."}
              </p>
            </div>

            <div className="stat-strip">
              <div className="stat-pill">
                <Database size={14} />
                <span>{selectedDatasetEntry?.scene_count ?? 0} scenes</span>
              </div>
              <div className="stat-pill">
                <Boxes size={14} />
                <span>{selectedScene?.stats.object_count ?? 0} objects</span>
              </div>
              <div className="stat-pill">
                <DoorOpen size={14} />
                <span>{selectedScene?.stats.room_count ?? 0} rooms</span>
              </div>
            </div>
          </div>

          <ScenePreviewCanvas scene={selectedScene} />
        </section>

        <aside className="info-panel">
          <section className="info-card">
            <div className="section-title">
              <ScrollText size={16} />
              <h3>Prompt / Description</h3>
            </div>
            <p className="long-copy">
              {selectedScene?.description || "This scene does not include a richer prompt yet."}
            </p>
          </section>

          <section className="info-card">
            <div className="section-title">
              <Database size={16} />
              <h3>Overview</h3>
            </div>
            <div className="metric-grid">
              {Object.entries(selectedScene?.stats ?? {}).map(([key, value]) => (
                <div key={key} className="metric-card">
                  <span>{key.replaceAll("_", " ")}</span>
                  <strong>{value}</strong>
                </div>
              ))}
            </div>
          </section>

          <section className="info-card">
            <div className="section-title">
              <Layers3 size={16} />
              <h3>Rooms</h3>
            </div>
            <div className="list-stack">
              {(selectedScene?.normalized.rooms ?? []).map((room) => (
                <article key={room.id} className="list-row">
                  <div>
                    <strong>{room.id}</strong>
                    <p>{selectedScene ? roomSubtitle(room, selectedScene.dataset) : ""}</p>
                  </div>
                </article>
              ))}
            </div>
          </section>

          <section className="info-card">
            <div className="section-title">
              <Boxes size={16} />
              <h3>Object Types</h3>
            </div>
            <div className="list-stack compact">
              {objectTypeSummary.slice(0, 12).map(([type, count]) => (
                <div key={type} className="type-row">
                  <span>{type}</span>
                  <strong>{count}</strong>
                </div>
              ))}
            </div>
          </section>

          <section className="info-card">
            <div className="section-title">
              <ImageIcon size={16} />
              <h3>Preview Images</h3>
            </div>
            <div className="image-grid">
              {previewImages.slice(0, 6).map((url) => (
                <a key={url} href={url} target="_blank" rel="noreferrer" className="image-thumb">
                  <img src={url} alt="" loading="lazy" />
                </a>
              ))}
            </div>
          </section>

          <section className="info-card">
            <div className="section-title">
              <Database size={16} />
              <h3>Source Assets</h3>
            </div>
            <div className="list-stack compact">
              {assetEntries.map(([key, value]) => (
                <div key={key} className="type-row asset-row">
                  <span>{key}</span>
                  <strong>{String(value)}</strong>
                </div>
              ))}
            </div>
          </section>
        </aside>
      </main>
    </div>
  );
}
