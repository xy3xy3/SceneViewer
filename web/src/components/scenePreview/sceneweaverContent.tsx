import { useEffect, useMemo, useState } from "react";
import type { RenderableWholeSceneGlbSceneManifest } from "../../types";
import {
  BatchedAssetModels,
  Bounds,
  type AssetPlacement,
  type InspectableObject,
  type RenderProgressSnapshot,
  type SceneBounds,
  type Vector3Tuple,
  ObjectHitTargets,
  ObjectLabels,
  SceneBoundsController,
  SelectionOverlays,
  buildObjectLabels,
  createEmptyBatchProgress,
  createEmptyBounds,
  expandBounds,
  finalizeBounds,
  labelText,
  resolveObjectPosition,
  updateMeasuredBoundsMap,
} from "./shared";

function sceneBoundsEqual(left: SceneBounds | null, right: SceneBounds): boolean {
  if (!left) {
    return false;
  }
  return (
    left.center[0] === right.center[0] &&
    left.center[1] === right.center[1] &&
    left.center[2] === right.center[2] &&
    left.size[0] === right.size[0] &&
    left.size[1] === right.size[1] &&
    left.size[2] === right.size[2]
  );
}

function computeSceneWeaverBounds(
  renderScene: RenderableWholeSceneGlbSceneManifest,
  measuredSceneBounds: SceneBounds | null,
  measuredObjectBounds: Record<string, SceneBounds>,
): SceneBounds {
  if (measuredSceneBounds) {
    return measuredSceneBounds;
  }

  const min: Vector3Tuple = [
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
  ];
  const max: Vector3Tuple = [
    Number.NEGATIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
  ];
  const minMax = { min, max };

  for (const object of renderScene.objects) {
    const measured = measuredObjectBounds[object.id];
    if (measured) {
      const half: Vector3Tuple = [
        Math.max(measured.size[0] / 2, 0.1),
        Math.max(measured.size[1] / 2, 0.1),
        Math.max(measured.size[2] / 2, 0.1),
      ];
      expandBounds(minMax, [
        measured.center[0] - half[0],
        measured.center[1] - half[1],
        measured.center[2] - half[2],
      ]);
      expandBounds(minMax, [
        measured.center[0] + half[0],
        measured.center[1] + half[1],
        measured.center[2] + half[2],
      ]);
      continue;
    }

    const fallbackSize = object.size ?? [0.25, 0.25, 0.25];
    const half: Vector3Tuple = [
      Math.max(fallbackSize[0] / 2, 0.1),
      Math.max(fallbackSize[1] / 2, 0.1),
      Math.max(fallbackSize[2] / 2, 0.1),
    ];
    expandBounds(minMax, [
      object.position[0] - half[0],
      object.position[1] - half[1],
      object.position[2] - half[2],
    ]);
    expandBounds(minMax, [
      object.position[0] + half[0],
      object.position[1] + half[1],
      object.position[2] + half[2],
    ]);
  }

  const width = Math.max(renderScene.room?.dimensions?.width ?? 0, 1);
  const length = Math.max(renderScene.room?.dimensions?.length ?? 0, 1);
  const height = Math.max(renderScene.room?.dimensions?.height ?? 2.8, 2.8);
  expandBounds(minMax, [0, 0, -length]);
  expandBounds(minMax, [width, height, 0]);

  if (!Number.isFinite(min[0]) || !Number.isFinite(max[0])) {
    return createEmptyBounds();
  }

  return finalizeBounds(min, max);
}

function buildWholeSceneObjectAssets(
  renderScene: RenderableWholeSceneGlbSceneManifest,
  positionOverrides?: Record<string, Vector3Tuple>,
): AssetPlacement[] {
  return renderScene.objects
    .filter((object) => Boolean(object.asset_path))
    .map(
      (object): AssetPlacement => ({
        key: object.id,
        assetPath: object.asset_path!,
        position: resolveObjectPosition(object.id, object.position, positionOverrides),
        rotationYDeg: object.rotation_y_deg,
        quaternion: object.quaternion,
        scale: object.scale ?? [1, 1, 1],
      }),
    );
}

export function SceneWeaverPreviewContent({
  renderScene,
  showObjectLabels,
  selectedObjectId,
  onSelectedObjectChange,
  objectPositionOverrides,
  onRenderProgressChange,
}: {
  renderScene: RenderableWholeSceneGlbSceneManifest;
  showObjectLabels: boolean;
  selectedObjectId: string | null;
  onSelectedObjectChange: (id: string | null) => void;
  objectPositionOverrides?: Record<string, Vector3Tuple>;
  onRenderProgressChange: (snapshot: RenderProgressSnapshot) => void;
}) {
  const [hoveredObjectId, setHoveredObjectId] = useState<string | null>(null);
  const [fitVersion, setFitVersion] = useState(0);
  const [measuredSceneBounds, setMeasuredSceneBounds] = useState<SceneBounds | null>(null);
  const [measuredObjectBounds, setMeasuredObjectBounds] = useState<Record<string, SceneBounds>>({});
  const sceneAssets = useMemo(
    (): AssetPlacement[] => [
      {
        key: `${renderScene.scene_uid}::scene-glb`,
        assetPath: renderScene.scene_glb,
        position: [0, 0, 0],
        rotationYDeg: 0,
        scale: [1, 1, 1],
      },
    ],
    [renderScene.scene_glb, renderScene.scene_uid],
  );
  const objectAssets = useMemo(
    () => buildWholeSceneObjectAssets(renderScene, objectPositionOverrides),
    [objectPositionOverrides, renderScene],
  );
  const [sceneBatchProgress, setSceneBatchProgress] = useState(() =>
    createEmptyBatchProgress(sceneAssets.length),
  );
  const [objectBatchProgress, setObjectBatchProgress] = useState(() =>
    createEmptyBatchProgress(objectAssets.length),
  );
  const inspectableObjects = useMemo(
    (): InspectableObject[] =>
      renderScene.objects.map((object) => ({
        id: object.id,
        label: labelText(object.object_type || object.description, object.id),
        position:
          measuredObjectBounds[object.id]?.center ??
          resolveObjectPosition(object.id, object.position, objectPositionOverrides),
        size: measuredObjectBounds[object.id]?.size ?? object.size ?? [0.25, 0.25, 0.25],
      })),
    [measuredObjectBounds, objectPositionOverrides, renderScene.objects],
  );
  const objectLabels = useMemo(
    () => buildObjectLabels(inspectableObjects, showObjectLabels, selectedObjectId, hoveredObjectId),
    [hoveredObjectId, inspectableObjects, selectedObjectId, showObjectLabels],
  );
  const sceneBounds = useMemo(
    () => computeSceneWeaverBounds(renderScene, measuredSceneBounds, measuredObjectBounds),
    [measuredObjectBounds, measuredSceneBounds, renderScene],
  );

  useEffect(() => {
    const total = sceneAssets.length + objectAssets.length;
    const sceneCompleted = Math.min(sceneBatchProgress.readyCount, sceneAssets.length);
    const objectCompleted = Math.min(objectBatchProgress.readyCount, objectAssets.length);
    const completed = sceneCompleted + objectCompleted;
    const ready =
      (sceneAssets.length === 0 || sceneBatchProgress.complete) &&
      (objectAssets.length === 0 || objectBatchProgress.complete);
    const progress = total === 0 ? 100 : Math.round((completed / total) * 100);
    const datasetLabel = renderScene.dataset === "hssd" ? "HSSD stage" : "SceneWeaver";
    let detail = `Mounted ${datasetLabel} GLB`;
    if (!ready) {
      detail =
        sceneCompleted < sceneAssets.length
          ? `Mounted ${sceneCompleted}/${sceneAssets.length} scene assets`
          : `Mounted ${objectCompleted}/${objectAssets.length} object assets`;
    }
    onRenderProgressChange({
      ready,
      stage: ready ? "Scene ready" : "Preparing scene mesh",
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
    renderScene.dataset,
    sceneAssets.length,
    sceneBatchProgress.complete,
    sceneBatchProgress.readyCount,
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
        <BatchedAssetModels
          key={`${renderScene.scene_uid}::sceneweaver-scene`}
          items={sceneAssets}
          batchSize={1}
          materialProfile="sceneweaver"
          onItemBounds={(_, bounds) =>
            setMeasuredSceneBounds((current) => (sceneBoundsEqual(current, bounds) ? current : bounds))
          }
          onProgress={setSceneBatchProgress}
          onComplete={() => {
            setFitVersion((current) => current + 1);
          }}
        />
        {objectAssets.length > 0 ? (
          <BatchedAssetModels
            key={`${renderScene.scene_uid}::whole-scene-objects`}
            items={objectAssets}
            batchSize={24}
            materialProfile="sceneweaver"
            onItemBounds={(id, bounds) =>
              setMeasuredObjectBounds((current) => updateMeasuredBoundsMap(current, id, bounds))
            }
            onProgress={setObjectBatchProgress}
          />
        ) : null}
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
      </group>
    </Bounds>
  );
}
