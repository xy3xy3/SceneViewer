import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Boxes,
  Check,
  ChevronLeft,
  ChevronRight,
  Copy,
  Database,
  Image as ImageIcon,
  Layers3,
  Tag,
  ScrollText,
  RotateCcw,
} from "lucide-react";
import {
  ScenePreviewCanvas,
  type ScenePointerDebugSnapshot,
  type ScenePreviewDebugObjectSnapshot,
  ScenePreviewProgressIndicator,
  type ScenePreviewProgressSnapshot,
} from "./components/ScenePreviewCanvas";
import type { Vector3Tuple } from "./components/scenePreview/shared";
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
type DebugAxis = "x" | "y" | "z";
type CoordinateDraft = Record<DebugAxis, string>;

type RenderableDebugObject = {
  id: string;
  label: string;
  originalPosition: Vector3Tuple;
  sourceId?: string | null;
};

const DEBUG_AXES: Array<{ axis: DebugAxis; index: 0 | 1 | 2 }> = [
  { axis: "x", index: 0 },
  { axis: "y", index: 1 },
  { axis: "z", index: 2 },
];

function formatDatasetLabel(dataset: string): string {
  if (dataset === "hsm") {
    return "HSM";
  }
  if (dataset === "sage") {
    return "SAGE";
  }
  if (dataset === "scenesmith") {
    return "SceneSmith";
  }
  if (dataset === "sceneweaver") {
    return "SceneWeaver";
  }
  if (dataset === "hssd") {
    return "HSSD";
  }
  return "3D-FRONT";
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
  if (dataset === "sage" || dataset === "hsm") {
    const width = room.dimensions?.width;
    const length = room.dimensions?.length;
    const objectCount = room.object_ids?.length ?? 0;
    return `${room.room_type || "room"} · ${width ?? "?"}m x ${length ?? "?"}m · ${objectCount} objects`;
  }

  if (dataset === "3dfront") {
    const width = room.dimensions?.width;
    const length = room.dimensions?.length;
    const objectCount = room.object_count ?? room.object_ids?.length ?? 0;
    const shellCount = room.shell_refs?.length ?? 0;
    return `${room.room_type || "room"} · ${width?.toFixed?.(2) ?? "?"}m x ${length?.toFixed?.(2) ?? "?"}m · ${objectCount} objects · ${shellCount} shells`;
  }

  if (dataset === "sceneweaver") {
    const width = room.dimensions?.width;
    const length = room.dimensions?.length;
    const objectCount = room.object_count ?? room.object_ids?.length ?? 0;
    return `${room.room_type || "room"} · ${width?.toFixed?.(2) ?? "?"}m x ${length?.toFixed?.(2) ?? "?"}m · ${objectCount} objects`;
  }

  if (dataset === "hssd") {
    const objectCount = room.object_count ?? room.object_ids?.length ?? 0;
    return `${room.room_type || "stage"} · ${objectCount} objects · stage GLB preview`;
  }

  const objectCount = room.object_count ?? room.objects?.length ?? 0;
  return `${objectCount} objects · ${room.floor_plan_assets?.wall_gltfs?.length ?? 0} wall meshes`;
}

function formatCoordinate(value: number): string {
  if (!Number.isFinite(value)) {
    return "-";
  }
  return value.toFixed(3);
}

function formatCoordinateInput(value: number): string {
  if (!Number.isFinite(value)) {
    return "0";
  }
  return value.toFixed(3);
}

function formatVector(value: Vector3Tuple): string {
  return `${formatCoordinate(value[0])}, ${formatCoordinate(value[1])}, ${formatCoordinate(value[2])}`;
}

function positionsEqual(left: Vector3Tuple, right: Vector3Tuple, epsilon = 0.0001): boolean {
  return left.every((value, index) => Math.abs(value - right[index]) <= epsilon);
}

function buildRenderableDebugObjects(
  scene: SceneManifest | null,
  renderScene: RenderableSceneManifest | null,
): RenderableDebugObject[] {
  if (!renderScene) {
    return [];
  }

  const sourceObjects = new Map((scene?.normalized.objects ?? []).map((object) => [object.id, object] as const));

  switch (renderScene.dataset) {
    case "sage":
      return renderScene.objects.map((object) => ({
        id: object.id,
        label: object.type || object.description || object.source_id || object.id,
        originalPosition: object.position,
        sourceId: object.source_id,
      }));
    case "hsm":
      return renderScene.objects.map((object) => ({
        id: object.id,
        label:
          object.name ||
          object.semantic_label ||
          object.object_type ||
          object.category ||
          object.description ||
          object.source_id ||
          object.id,
        originalPosition: object.position,
        sourceId: object.source_id,
      }));
    case "scenesmith":
      return renderScene.objects.map((object) => {
        const sourceObject = sourceObjects.get(object.id);
        return {
          id: object.id,
          label:
            sourceObject?.name ||
            sourceObject?.object_type ||
            object.object_type ||
            sourceObject?.description ||
            object.description ||
            object.id,
          originalPosition: object.position,
        };
      });
    case "3dfront":
      return renderScene.objects.map((object) => {
        const sourceObject = sourceObjects.get(object.id);
        return {
          id: object.id,
          label:
            sourceObject?.name ||
            sourceObject?.type ||
            sourceObject?.object_type ||
            object.object_type ||
            object.description ||
            object.source_ref ||
            object.id,
          originalPosition: object.position,
          sourceId: object.source_model_jid ?? object.source_ref,
        };
      });
    case "sceneweaver":
    case "hssd":
      return renderScene.objects.map((object) => ({
        id: object.id,
        label: object.object_type || object.description || object.source_id || object.id,
        originalPosition: object.position,
        sourceId: object.source_id,
      }));
  }
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
  const [selectedObjectId, setSelectedObjectId] = useState<string | null>(null);
  const [objectPositionOverrides, setObjectPositionOverrides] = useState<
    Record<string, Vector3Tuple>
  >({});
  const [pointerDebug, setPointerDebug] = useState<ScenePointerDebugSnapshot | null>(null);
  const [positionDraft, setPositionDraft] = useState<CoordinateDraft | null>(null);
  const [copyState, setCopyState] = useState<"idle" | "copied" | "failed">("idle");
  const [previewProgress, setPreviewProgress] = useState<ScenePreviewProgressSnapshot | null>(
    null,
  );
  const [hiddenReadyProgressSceneUid, setHiddenReadyProgressSceneUid] = useState<string | null>(
    null,
  );

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
  const selectedSceneIndex = selectedDatasetIndex?.scenes.findIndex(
    (scene) => scene.scene_uid === selectedSceneUid,
  ) ?? -1;
  const hasPreviousScene = selectedSceneIndex > 0;
  const hasNextScene =
    selectedSceneIndex >= 0 &&
    selectedSceneIndex < (selectedDatasetIndex?.scenes.length ?? 0) - 1;
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
  const visiblePreviewProgress =
    previewProgress?.sceneUid === selectedRenderScene?.scene_uid ? previewProgress : null;
  const topbarPreviewProgress =
    visiblePreviewProgress?.previewReady &&
    hiddenReadyProgressSceneUid === visiblePreviewProgress.sceneUid
      ? null
      : visiblePreviewProgress;

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
  const renderableDebugObjects = useMemo(
    () => buildRenderableDebugObjects(selectedScene, selectedRenderScene),
    [selectedRenderScene, selectedScene],
  );
  const renderableDebugObjectMap = useMemo(
    () => new Map(renderableDebugObjects.map((object) => [object.id, object] as const)),
    [renderableDebugObjects],
  );
  const selectedObjectDebugInfo = useMemo<ScenePreviewDebugObjectSnapshot | null>(() => {
    if (!selectedObjectId) {
      return null;
    }

    const selectedObject = renderableDebugObjectMap.get(selectedObjectId);
    if (!selectedObject) {
      return null;
    }

    const overridePosition = objectPositionOverrides[selectedObject.id];
    return {
      id: selectedObject.id,
      label: selectedObject.label,
      originalPosition: selectedObject.originalPosition,
      currentPosition: overridePosition ?? selectedObject.originalPosition,
      hasOverride: Boolean(overridePosition),
    };
  }, [objectPositionOverrides, renderableDebugObjectMap, selectedObjectId]);
  const selectedObjectDelta = selectedObjectDebugInfo
    ? ([
        selectedObjectDebugInfo.currentPosition[0] - selectedObjectDebugInfo.originalPosition[0],
        selectedObjectDebugInfo.currentPosition[1] - selectedObjectDebugInfo.originalPosition[1],
        selectedObjectDebugInfo.currentPosition[2] - selectedObjectDebugInfo.originalPosition[2],
      ] as Vector3Tuple)
    : null;
  const hasPositionOverrides = Object.keys(objectPositionOverrides).length > 0;

  useEffect(() => {
    if (copyState === "idle") {
      return;
    }

    const timer = window.setTimeout(() => setCopyState("idle"), 1800);
    return () => window.clearTimeout(timer);
  }, [copyState]);

  useEffect(() => {
    setSelectedObjectId(null);
    setObjectPositionOverrides({});
    setPointerDebug(null);
    setPositionDraft(null);
  }, [selectedSceneUid]);

  useEffect(() => {
    if (!selectedObjectId || renderableDebugObjectMap.has(selectedObjectId)) {
      return;
    }
    setSelectedObjectId(null);
  }, [renderableDebugObjectMap, selectedObjectId]);

  useEffect(() => {
    if (!selectedObjectDebugInfo) {
      setPositionDraft(null);
      return;
    }

    setPositionDraft({
      x: formatCoordinateInput(selectedObjectDebugInfo.currentPosition[0]),
      y: formatCoordinateInput(selectedObjectDebugInfo.currentPosition[1]),
      z: formatCoordinateInput(selectedObjectDebugInfo.currentPosition[2]),
    });
  }, [
    selectedObjectDebugInfo?.id,
    selectedObjectDebugInfo?.currentPosition[0],
    selectedObjectDebugInfo?.currentPosition[1],
    selectedObjectDebugInfo?.currentPosition[2],
  ]);

  useEffect(() => {
    if (!visiblePreviewProgress?.previewReady) {
      return;
    }

    const timer = window.setTimeout(() => {
      setHiddenReadyProgressSceneUid(visiblePreviewProgress.sceneUid);
    }, 1000);

    return () => window.clearTimeout(timer);
  }, [visiblePreviewProgress?.previewReady, visiblePreviewProgress?.sceneUid]);

  function handleDatasetChange(dataset: string) {
    setSelectedDataset(dataset);
    const nextScene = renderDatasetIndices[dataset]?.scenes[0]?.scene_uid ?? "";
    setSelectedSceneUid(nextScene);
  }

  function handleWallOpacityChange(value: string) {
    setWallOpacity(Number(value) / 100);
  }

  const handlePreviewProgressChange = useCallback((snapshot: ScenePreviewProgressSnapshot) => {
    setPreviewProgress(snapshot);
  }, []);

  const handlePointerDebugChange = useCallback((snapshot: ScenePointerDebugSnapshot | null) => {
    setPointerDebug(snapshot);
  }, []);

  const updateSelectedObjectPosition = useCallback(
    (objectId: string, nextPosition: Vector3Tuple) => {
      const sourceObject = renderableDebugObjectMap.get(objectId);
      if (!sourceObject) {
        return;
      }

      setObjectPositionOverrides((current) => {
        if (positionsEqual(sourceObject.originalPosition, nextPosition)) {
          if (!(objectId in current)) {
            return current;
          }

          const next = { ...current };
          delete next[objectId];
          return next;
        }

        return {
          ...current,
          [objectId]: nextPosition,
        };
      });
    },
    [renderableDebugObjectMap],
  );

  const handlePositionDraftChange = useCallback(
    (axis: DebugAxis, value: string) => {
      if (!selectedObjectDebugInfo) {
        return;
      }

      setPositionDraft((current) => ({
        x: current?.x ?? formatCoordinateInput(selectedObjectDebugInfo.currentPosition[0]),
        y: current?.y ?? formatCoordinateInput(selectedObjectDebugInfo.currentPosition[1]),
        z: current?.z ?? formatCoordinateInput(selectedObjectDebugInfo.currentPosition[2]),
        [axis]: value,
      }));

      const parsed = Number(value);
      if (!Number.isFinite(parsed)) {
        return;
      }

      const axisIndex = DEBUG_AXES.find((entry) => entry.axis === axis)?.index;
      if (axisIndex === undefined) {
        return;
      }

      const nextPosition = [...selectedObjectDebugInfo.currentPosition] as Vector3Tuple;
      nextPosition[axisIndex] = parsed;
      updateSelectedObjectPosition(selectedObjectDebugInfo.id, nextPosition);
    },
    [selectedObjectDebugInfo, updateSelectedObjectPosition],
  );

  const resetPositionDraft = useCallback(() => {
    if (!selectedObjectDebugInfo) {
      setPositionDraft(null);
      return;
    }

    setPositionDraft({
      x: formatCoordinateInput(selectedObjectDebugInfo.currentPosition[0]),
      y: formatCoordinateInput(selectedObjectDebugInfo.currentPosition[1]),
      z: formatCoordinateInput(selectedObjectDebugInfo.currentPosition[2]),
    });
  }, [selectedObjectDebugInfo]);

  const handlePositionDraftBlur = useCallback(
    (axis: DebugAxis) => {
      if (!positionDraft) {
        return;
      }

      const value = Number(positionDraft[axis]);
      if (Number.isFinite(value)) {
        return;
      }

      resetPositionDraft();
    },
    [positionDraft, resetPositionDraft],
  );

  const handleResetSelectedObjectPosition = useCallback(() => {
    if (!selectedObjectDebugInfo) {
      return;
    }
    updateSelectedObjectPosition(selectedObjectDebugInfo.id, selectedObjectDebugInfo.originalPosition);
  }, [selectedObjectDebugInfo, updateSelectedObjectPosition]);

  const handleResetAllObjectPositions = useCallback(() => {
    setObjectPositionOverrides({});
  }, []);

  function handleSceneStep(direction: -1 | 1) {
    if (!selectedDatasetIndex?.scenes.length || selectedSceneIndex < 0) {
      return;
    }

    const nextScene = selectedDatasetIndex.scenes[selectedSceneIndex + direction];
    if (!nextScene) {
      return;
    }

    setSelectedSceneUid(nextScene.scene_uid);
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
            <p>Preview HSM, SAGE, SceneSmith, SceneWeaver, HSSD, and 3D-FRONT scenes with repo-local renderable assets.</p>
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
            <div className="scene-select-row">
              <button
                type="button"
                className="scene-switch-button"
                onClick={() => handleSceneStep(-1)}
                disabled={loading || !hasPreviousScene}
                aria-label="Previous scene"
              >
                <ChevronLeft size={16} />
              </button>
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
              <button
                type="button"
                className="scene-switch-button"
                onClick={() => handleSceneStep(1)}
                disabled={loading || !hasNextScene}
                aria-label="Next scene"
              >
                <ChevronRight size={16} />
              </button>
            </div>
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

        {selectedRenderScene ? (
          <ScenePreviewProgressIndicator
            progress={topbarPreviewProgress}
            className="topbar-progress"
          />
        ) : null}
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
            selectedObjectId={selectedObjectId}
            selectedObjectDebugInfo={selectedObjectDebugInfo}
            objectPositionOverrides={objectPositionOverrides}
            onSelectedObjectChange={setSelectedObjectId}
            onPointerDebugChange={handlePointerDebugChange}
            onProgressChange={handlePreviewProgressChange}
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
              <Tag size={16} />
              <h3>Debug Coordinates</h3>
            </div>

            <div className="debug-coordinate-grid">
              <div className="debug-coordinate-card">
                <span>Canvas</span>
                <strong>
                  {pointerDebug ? `${pointerDebug.canvas[0]}, ${pointerDebug.canvas[1]}` : "--"}
                </strong>
                <p>Mouse position inside the preview canvas</p>
              </div>
              <div className="debug-coordinate-card">
                <span>World</span>
                <strong>{pointerDebug?.world ? formatVector(pointerDebug.world) : "--"}</strong>
                <p>Ground-plane coordinates under the cursor</p>
              </div>
            </div>

            {selectedObjectDebugInfo ? (
              <div className="debug-editor">
                <div className="debug-object-summary">
                  <div>
                    <span>Selected</span>
                    <strong>{selectedObjectDebugInfo.label}</strong>
                    <p>{selectedObjectDebugInfo.id}</p>
                  </div>
                  <div className="debug-pill">
                    {selectedObjectDebugInfo.hasOverride ? "Simulated override" : "Original render position"}
                  </div>
                </div>

                <div className="debug-coordinate-list">
                  <div className="debug-coordinate-row">
                    <span>Original</span>
                    <code>{formatVector(selectedObjectDebugInfo.originalPosition)}</code>
                  </div>
                  <div className="debug-coordinate-row">
                    <span>Current</span>
                    <code>{formatVector(selectedObjectDebugInfo.currentPosition)}</code>
                  </div>
                  <div className="debug-coordinate-row">
                    <span>Delta</span>
                    <code>{selectedObjectDelta ? formatVector(selectedObjectDelta) : "--"}</code>
                  </div>
                </div>

                <div className="debug-axis-editor">
                  {DEBUG_AXES.map(({ axis }) => (
                    <label key={axis} className="debug-axis-field">
                      <span>{axis.toUpperCase()}</span>
                      <input
                        type="number"
                        inputMode="decimal"
                        step="0.05"
                        value={positionDraft?.[axis] ?? ""}
                        onChange={(event) => handlePositionDraftChange(axis, event.target.value)}
                        onBlur={() => handlePositionDraftBlur(axis)}
                      />
                    </label>
                  ))}
                </div>

                <div className="debug-actions">
                  <button
                    type="button"
                    className="debug-action-button"
                    onClick={handleResetSelectedObjectPosition}
                    disabled={!selectedObjectDebugInfo.hasOverride}
                  >
                    <RotateCcw size={14} />
                    <span>Reset Object</span>
                  </button>
                  <button
                    type="button"
                    className="debug-action-button debug-action-button-secondary"
                    onClick={handleResetAllObjectPositions}
                    disabled={!hasPositionOverrides}
                  >
                    <RotateCcw size={14} />
                    <span>Reset All</span>
                  </button>
                </div>

                <p className="debug-hint">
                  这些改动只在当前网页会话里生效，用于人类调试，不会写回场景文件。
                </p>
              </div>
            ) : (
              <p className="long-copy">
                在左侧 3D 预览里点击任意物体后，这里会显示它的当前坐标，并允许你临时模拟修改
                <code>x / y / z</code>。
              </p>
            )}
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
