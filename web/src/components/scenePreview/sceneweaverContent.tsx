import { useEffect, useMemo, useState } from "react";
import type { RenderableSceneWeaverSceneManifest } from "../../types";
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
  renderScene: RenderableSceneWeaverSceneManifest,
  measuredSceneBounds: SceneBounds | null,
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
    const half: Vector3Tuple = [
      Math.max(object.size[0] / 2, 0.1),
      Math.max(object.size[1] / 2, 0.1),
      Math.max(object.size[2] / 2, 0.1),
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

export function SceneWeaverPreviewContent({
  renderScene,
  showObjectLabels,
  onRenderProgressChange,
}: {
  renderScene: RenderableSceneWeaverSceneManifest;
  showObjectLabels: boolean;
  onRenderProgressChange: (snapshot: RenderProgressSnapshot) => void;
}) {
  const [hoveredObjectId, setHoveredObjectId] = useState<string | null>(null);
  const [selectedObjectId, setSelectedObjectId] = useState<string | null>(null);
  const [fitVersion, setFitVersion] = useState(0);
  const [measuredSceneBounds, setMeasuredSceneBounds] = useState<SceneBounds | null>(null);
  const sceneAsset = useMemo(
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
  const [batchProgress, setBatchProgress] = useState(() =>
    createEmptyBatchProgress(sceneAsset.length),
  );
  const inspectableObjects = useMemo(
    (): InspectableObject[] =>
      renderScene.objects.map((object) => ({
        id: object.id,
        label: labelText(object.object_type || object.description, object.id),
        position: object.position,
        size: object.size,
      })),
    [renderScene.objects],
  );
  const objectLabels = useMemo(
    () => buildObjectLabels(inspectableObjects, showObjectLabels, selectedObjectId, hoveredObjectId),
    [hoveredObjectId, inspectableObjects, selectedObjectId, showObjectLabels],
  );
  const sceneBounds = useMemo(
    () => computeSceneWeaverBounds(renderScene, measuredSceneBounds),
    [measuredSceneBounds, renderScene],
  );

  useEffect(() => {
    const total = sceneAsset.length;
    const completed = Math.min(batchProgress.readyCount, total);
    const ready = total === 0 || batchProgress.complete;
    const progress = total === 0 ? 100 : Math.round((completed / total) * 100);
    onRenderProgressChange({
      ready,
      stage: ready ? "Scene ready" : "Preparing scene mesh",
      detail: ready ? "Mounted exported SceneWeaver GLB" : `Mounted ${completed}/${total} scene assets`,
      completed,
      total,
      progress,
    });
  }, [batchProgress.complete, batchProgress.readyCount, onRenderProgressChange, sceneAsset.length]);

  function handleObjectSelect(id: string, additive: boolean) {
    setSelectedObjectId((current) => {
      if (additive && current === id) {
        return null;
      }
      return id;
    });
  }

  return (
    <Bounds key={renderScene.scene_uid} clip observe={false} margin={1.18}>
      <SceneBoundsController sceneKey={renderScene.scene_uid} fitVersion={fitVersion} />
      <group
        onPointerMissed={() => {
          setHoveredObjectId(null);
          setSelectedObjectId(null);
          document.body.style.cursor = "default";
        }}
      >
        <mesh position={sceneBounds.center} visible={false}>
          <boxGeometry args={sceneBounds.size} />
          <meshBasicMaterial transparent opacity={0} depthWrite={false} />
        </mesh>
        <BatchedAssetModels
          key={`${renderScene.scene_uid}::sceneweaver-scene`}
          items={sceneAsset}
          batchSize={1}
          materialProfile="sceneweaver"
          onItemBounds={(_, bounds) =>
            setMeasuredSceneBounds((current) => (sceneBoundsEqual(current, bounds) ? current : bounds))
          }
          onProgress={setBatchProgress}
          onComplete={() => {
            setFitVersion((current) => current + 1);
          }}
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
      </group>
    </Bounds>
  );
}
