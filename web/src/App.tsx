import { useEffect, useMemo, useState } from "react";
import {
  Boxes,
  Check,
  Copy,
  Database,
  Image as ImageIcon,
  Layers3,
  Tag,
  ScrollText,
} from "lucide-react";
import { ScenePreviewCanvas } from "./components/ScenePreviewCanvas";
import { fetchRepoJson, toRepoAssetUrl } from "./lib/repoAssets";
import type {
  DatasetCatalog,
  DatasetIndex,
  NormalizedObject,
  NormalizedRoom,
  RenderableDatasetCatalog,
  RenderableDatasetIndex,
  RenderableSceneManifest,
  RenderableSceneSummary,
  SceneManifest,
  SceneSummary,
} from "./types";
import "./index.css";

type WallDisplayMode = "solid" | "transparent" | "hidden" | "wireframe";

function formatDatasetLabel(dataset: string): string {
  return dataset === "sage" ? "SAGE" : "SceneSmith";
}

function formatSceneLabel(
  scene: RenderableSceneSummary,
  metadata?: SceneSummary | null,
): string {
  if (scene.subset) {
    return `${scene.subset} / ${scene.scene_id}`;
  }
  return metadata?.title || scene.scene_id;
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

async function copyText(text: string): Promise<boolean> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return true;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "absolute";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  document.body.removeChild(textarea);
  return copied;
}

export default function App() {
  const [catalog, setCatalog] = useState<DatasetCatalog | null>(null);
  const [datasetIndices, setDatasetIndices] = useState<Record<string, DatasetIndex>>({});
  const [renderDatasetIndices, setRenderDatasetIndices] = useState<
    Record<string, RenderableDatasetIndex>
  >({});
  const [sceneCache, setSceneCache] = useState<Record<string, SceneManifest>>({});
  const [renderSceneCache, setRenderSceneCache] = useState<Record<string, RenderableSceneManifest>>(
    {},
  );
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [selectedSceneUid, setSelectedSceneUid] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [wallOpacity, setWallOpacity] = useState(0.35);
  const [wallDisplayMode, setWallDisplayMode] = useState<WallDisplayMode>("transparent");
  const [showObjectLabels, setShowObjectLabels] = useState(false);
  const [copyState, setCopyState] = useState<"idle" | "copied" | "failed">("idle");

  useEffect(() => {
    let cancelled = false;

    async function loadCatalog() {
      try {
        setLoading(true);
        const [preprocessedCatalog, nextRenderCatalog] = await Promise.all([
          fetchRepoJson<DatasetCatalog>("assets/preprocessed/datasets.json"),
          fetchRepoJson<RenderableDatasetCatalog>("assets/renderable/datasets.json"),
        ]);
        if (cancelled) {
          return;
        }

        const renderEntries = new Map(
          nextRenderCatalog.datasets.map((entry) => [entry.dataset, entry] as const),
        );
        const nextCatalog: DatasetCatalog = {
          ...preprocessedCatalog,
          datasets: preprocessedCatalog.datasets
            .filter((entry) => renderEntries.has(entry.dataset))
            .map((entry) => ({
              ...entry,
              scene_count: renderEntries.get(entry.dataset)?.scene_count ?? entry.scene_count,
              skipped_count: entry.skipped_count ?? 0,
            })),
        };

        const preprocessedIndices = await Promise.all(
          nextCatalog.datasets.map(async (entry) => [
            entry.dataset,
            await fetchRepoJson<DatasetIndex>(entry.index_path),
          ]),
        );
        const renderIndices = await Promise.all(
          nextCatalog.datasets.map(async (entry) => {
            const renderEntry = renderEntries.get(entry.dataset);
            if (!renderEntry) {
              throw new Error(`Missing renderable catalog entry for ${entry.dataset}`);
            }
            return [
              entry.dataset,
              await fetchRepoJson<RenderableDatasetIndex>(renderEntry.index_path),
            ] as const;
          }),
        );

        if (cancelled) {
          return;
        }

        const nextIndices = Object.fromEntries(preprocessedIndices);
        const nextRenderIndices = Object.fromEntries(renderIndices);
        setCatalog(nextCatalog);
        setDatasetIndices(nextIndices);
        setRenderDatasetIndices(nextRenderIndices);

        const defaultDataset = nextCatalog.datasets[0]?.dataset ?? "";
        const defaultScene = nextRenderIndices[defaultDataset]?.scenes[0]?.scene_uid ?? "";
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

  const selectedDatasetIndex = renderDatasetIndices[selectedDataset];
  const selectedPreprocessedDatasetIndex = datasetIndices[selectedDataset];
  const selectedSceneRenderSummary = selectedDatasetIndex?.scenes.find(
    (scene) => scene.scene_uid === selectedSceneUid,
  );
  const selectedSceneSummary = selectedPreprocessedDatasetIndex?.scenes.find(
    (scene) => scene.scene_uid === selectedSceneUid,
  );

  const preprocessedSceneSummaryMap = useMemo(
    () =>
      new Map(
        (selectedPreprocessedDatasetIndex?.scenes ?? []).map((scene) => [scene.scene_uid, scene] as const),
      ),
    [selectedPreprocessedDatasetIndex],
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

  useEffect(() => {
    if (!selectedSceneRenderSummary) {
      return;
    }

    const summary = selectedSceneRenderSummary;
    if (renderSceneCache[summary.scene_uid]) {
      return;
    }

    let cancelled = false;
    async function loadRenderScene() {
      try {
        const manifest = await fetchRepoJson<RenderableSceneManifest>(summary.render_manifest);
        if (cancelled) {
          return;
        }
        setRenderSceneCache((current) => ({
          ...current,
          [summary.scene_uid]: manifest,
        }));
      } catch (loadError) {
        if (!cancelled) {
          const message =
            loadError instanceof Error ? loadError.message : "Failed to load render manifest";
          setError(message);
        }
      }
    }

    void loadRenderScene();

    return () => {
      cancelled = true;
    };
  }, [renderSceneCache, selectedSceneRenderSummary]);

  const selectedScene = selectedSceneSummary
    ? sceneCache[selectedSceneSummary.scene_uid] ?? null
    : null;
  const selectedRenderScene = selectedSceneRenderSummary
    ? renderSceneCache[selectedSceneRenderSummary.scene_uid] ?? null
    : null;

  const previewImages = useMemo(() => collectPreviewImages(selectedScene), [selectedScene]);
  const objectTypeSummary = useMemo(
    () => summarizeObjectTypes(selectedScene?.normalized.objects ?? []),
    [selectedScene],
  );
  const assetEntries = useMemo(() => sourceAssetEntries(selectedScene), [selectedScene]);
  const selectedSceneLabel = selectedSceneRenderSummary
    ? formatSceneLabel(
        selectedSceneRenderSummary,
        preprocessedSceneSummaryMap.get(selectedSceneRenderSummary.scene_uid),
      )
    : selectedSceneSummary?.title || selectedSceneUid || "Choose a scene";

  useEffect(() => {
    if (copyState === "idle") {
      return;
    }

    const timer = window.setTimeout(() => setCopyState("idle"), 1800);
    return () => window.clearTimeout(timer);
  }, [copyState]);

  function handleDatasetChange(dataset: string) {
    setSelectedDataset(dataset);
    const nextScene = renderDatasetIndices[dataset]?.scenes[0]?.scene_uid ?? "";
    setSelectedSceneUid(nextScene);
  }

  function handleWallOpacityChange(value: string) {
    setWallOpacity(Number(value) / 100);
  }

  async function handleCopySceneName() {
    if (!selectedSceneLabel || selectedSceneLabel === "Choose a scene") {
      setCopyState("failed");
      return;
    }

    try {
      const copied = await copyText(selectedSceneLabel);
      setCopyState(copied ? "copied" : "failed");
    } catch {
      setCopyState("failed");
    }
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
            <p>Preview SAGE and SceneSmith scenes with repo-local renderable assets.</p>
          </div>
        </div>

        <div className="toolbar">
          <label className="select-shell">
            <span>Dataset</span>
            <select
              value={selectedDataset}
              onChange={(event) => handleDatasetChange(event.target.value)}
              disabled={loading || !(catalog?.datasets.length ?? 0)}
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
                  {formatSceneLabel(scene, preprocessedSceneSummaryMap.get(scene.scene_uid))}
                </option>
              ))}
            </select>
          </label>

          <label className="control-shell control-shell-range">
            <span>Wall Opacity</span>
            <div className="range-control">
              <input
                type="range"
                min={0}
                max={100}
                step={5}
                value={Math.round(wallOpacity * 100)}
                onInput={(event) => handleWallOpacityChange(event.currentTarget.value)}
                onChange={(event) => handleWallOpacityChange(event.currentTarget.value)}
              />
              <strong>{Math.round(wallOpacity * 100)}%</strong>
            </div>
          </label>

          <label className="select-shell">
            <span>Wall View</span>
            <select
              value={wallDisplayMode}
              onChange={(event) => setWallDisplayMode(event.target.value as WallDisplayMode)}
            >
              <option value="solid">Solid</option>
              <option value="transparent">Transparent</option>
              <option value="hidden">Hidden</option>
              <option value="wireframe">Wireframe</option>
            </select>
          </label>

          <label className="control-shell control-shell-toggle">
            <span>Object Labels</span>
            <div className="toggle-control">
              <Tag size={14} />
              <input
                type="checkbox"
                checked={showObjectLabels}
                onChange={(event) => setShowObjectLabels(event.target.checked)}
              />
              <strong>{showObjectLabels ? "On" : "Off"}</strong>
            </div>
          </label>
        </div>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <main className="workspace">
        <section className="preview-panel">
          <ScenePreviewCanvas
            scene={selectedScene}
            renderScene={selectedRenderScene}
            wallOpacity={wallOpacity}
            wallDisplayMode={wallDisplayMode}
            showObjectLabels={showObjectLabels}
          />
        </section>

        <aside className="info-panel">
          <div className="info-panel-header">
            <div className="scene-identity">
              <span>Scene Name</span>
              <strong>{selectedSceneLabel}</strong>
            </div>
            <button type="button" className="scene-copy-button" onClick={handleCopySceneName}>
              {copyState === "copied" ? <Check size={15} /> : <Copy size={15} />}
              <span>
                {copyState === "copied"
                  ? "Copied"
                  : copyState === "failed"
                    ? "Retry"
                    : "Copy"}
              </span>
            </button>
          </div>

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
