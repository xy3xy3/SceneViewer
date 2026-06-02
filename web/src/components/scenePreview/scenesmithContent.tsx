import { useEffect, useMemo, useState } from "react";
import type { RenderableSceneSmithSceneManifest, SceneManifest } from "../../types";
import {
  BatchedAssetModels,
  Bounds,
  type Vector3Tuple,
  type RenderProgressSnapshot,
  type SceneBounds,
  type WallDisplayMode,
  ObjectHitTargets,
  ObjectLabels,
  SceneBoundsController,
  SelectionOverlays,
  buildObjectLabels,
  createEmptyBatchProgress,
  updateMeasuredBoundsMap,
} from "./shared";
import { buildSceneSmithObjectAssets, buildSceneSmithShellAssets, sceneSmithRoomGeometryPaths } from "./scenesmith/assets";
import { computeSceneSmithBounds } from "./scenesmith/bounds";
import { buildSceneSmithInspectableObjects } from "./scenesmith/inspectables";
import { useSceneSmithShellTransforms } from "./scenesmith/useSceneSmithShellTransforms";

export function SceneSmithPreviewContent({
  scene,
  renderScene,
  wallOpacity,
  wallDisplayMode,
  showObjectLabels,
  selectedObjectId,
  onSelectedObjectChange,
  objectPositionOverrides,
  onRenderProgressChange,
}: {
  scene: SceneManifest | null;
  renderScene: RenderableSceneSmithSceneManifest;
  wallOpacity: number;
  wallDisplayMode: WallDisplayMode;
  showObjectLabels: boolean;
  selectedObjectId: string | null;
  onSelectedObjectChange: (id: string | null) => void;
  objectPositionOverrides?: Record<string, Vector3Tuple>;
  onRenderProgressChange: (snapshot: RenderProgressSnapshot) => void;
}) {
  const [hoveredObjectId, setHoveredObjectId] = useState<string | null>(null);
  const [fitVersion, setFitVersion] = useState(0);
  const roomGeometryPaths = useMemo(() => sceneSmithRoomGeometryPaths(scene), [scene]);
  const [shellsReady, setShellsReady] = useState(renderScene.room_shells.length === 0);
  const [measuredShellBounds, setMeasuredShellBounds] = useState<Record<string, SceneBounds>>({});
  const [measuredObjectBounds, setMeasuredObjectBounds] = useState<Record<string, SceneBounds>>({});
  const { shellTransformsLoaded, shellTransformMap } = useSceneSmithShellTransforms(roomGeometryPaths);
  const shellAssets = useMemo(
    () =>
      buildSceneSmithShellAssets({
        renderScene,
        shellTransformMap,
        shellTransformsLoaded,
        wallDisplayMode,
        wallOpacity,
      }),
    [renderScene, shellTransformMap, shellTransformsLoaded, wallDisplayMode, wallOpacity],
  );
  const objectAssets = useMemo(
    () => buildSceneSmithObjectAssets(renderScene, objectPositionOverrides),
    [objectPositionOverrides, renderScene],
  );
  const [shellBatchProgress, setShellBatchProgress] = useState(() =>
    createEmptyBatchProgress(renderScene.room_shells.length),
  );
  const [objectBatchProgress, setObjectBatchProgress] = useState(() =>
    createEmptyBatchProgress(renderScene.objects.length),
  );
  const sceneBounds = useMemo(
    () => computeSceneSmithBounds(scene, measuredShellBounds),
    [measuredShellBounds, scene],
  );
  const inspectableObjects = useMemo(
    () =>
      buildSceneSmithInspectableObjects({
        scene,
        renderScene,
        measuredObjectBounds,
        positionOverrides: objectPositionOverrides,
      }),
    [measuredObjectBounds, objectPositionOverrides, renderScene, scene],
  );
  const objectLabels = useMemo(
    () => buildObjectLabels(inspectableObjects, showObjectLabels, selectedObjectId, hoveredObjectId),
    [hoveredObjectId, inspectableObjects, selectedObjectId, showObjectLabels],
  );

  useEffect(() => {
    const shellCompleted = Math.min(shellBatchProgress.readyCount, shellAssets.length);
    const objectCompleted = Math.min(objectBatchProgress.readyCount, objectAssets.length);
    const total = shellAssets.length + objectAssets.length;
    const completed = shellCompleted + (shellsReady ? objectCompleted : 0);
    const ready =
      (shellAssets.length === 0 || shellBatchProgress.complete) &&
      (objectAssets.length === 0 || objectBatchProgress.complete);
    const progress = total === 0 ? 100 : Math.round((completed / total) * 100);

    let stage = "Scene ready";
    let detail = "All room shells and objects are mounted";
    if (!shellTransformsLoaded) {
      stage = "Preparing room shells";
      detail = "Loading SceneSmith room shell transforms";
    } else if (!shellsReady) {
      stage = "Preparing room shells";
      detail =
        shellAssets.length === 0
          ? "No room shell assets to stage"
          : `Mounted ${shellCompleted}/${shellAssets.length} room shell assets`;
    } else if (!ready) {
      stage = "Preparing objects";
      detail =
        objectAssets.length === 0
          ? "No object assets to stage"
          : `Mounted ${objectCompleted}/${objectAssets.length} object assets`;
    }

    onRenderProgressChange({
      ready,
      stage,
      detail,
      completed,
      total,
      progress,
    });
  }, [
    objectAssets.length,
    objectBatchProgress.complete,
    objectBatchProgress.readyCount,
    onRenderProgressChange,
    shellAssets.length,
    shellBatchProgress.complete,
    shellBatchProgress.readyCount,
    shellTransformsLoaded,
    shellsReady,
  ]);

  function handleObjectSelect(id: string, additive: boolean) {
    if (additive && selectedObjectId === id) {
      onSelectedObjectChange(null);
      return;
    }
    onSelectedObjectChange(id);
  }

  return (
    <Bounds key={renderScene.scene_uid} clip observe={false} margin={1.18}>
      <SceneBoundsController sceneKey={renderScene.scene_uid} fitVersion={fitVersion} />
      <group
        onPointerMissed={() => {
          setHoveredObjectId(null);
          onSelectedObjectChange(null);
          document.body.style.cursor = "default";
        }}
      >
        <mesh position={sceneBounds.center} visible={false}>
          <boxGeometry args={sceneBounds.size} />
          <meshBasicMaterial transparent opacity={0} depthWrite={false} />
        </mesh>
        {shellTransformsLoaded ? (
          <BatchedAssetModels
            key={`${renderScene.scene_uid}::scenesmith-shells`}
            items={shellAssets}
            batchSize={24}
            materialProfile="scenesmith"
            onItemBounds={(id, bounds) =>
              setMeasuredShellBounds((current) => updateMeasuredBoundsMap(current, id, bounds))
            }
            onProgress={setShellBatchProgress}
            onComplete={() => {
              setShellsReady(true);
              setFitVersion((current) => current + 1);
            }}
          />
        ) : null}
        {shellsReady ? (
          <>
            <BatchedAssetModels
              key={`${renderScene.scene_uid}::scenesmith-objects`}
              items={objectAssets}
              batchSize={20}
              onItemBounds={(id, bounds) =>
                setMeasuredObjectBounds((current) => updateMeasuredBoundsMap(current, id, bounds))
              }
              materialProfile="scenesmith"
              onProgress={setObjectBatchProgress}
            />
            <ObjectHitTargets
              items={inspectableObjects}
              activeId={selectedObjectId}
              onHoverChange={setHoveredObjectId}
              onSelect={handleObjectSelect}
            />
            <SelectionOverlays
              items={inspectableObjects}
              activeId={selectedObjectId}
              hoveredId={hoveredObjectId}
            />
            <ObjectLabels items={objectLabels} />
          </>
        ) : null}
      </group>
    </Bounds>
  );
}
