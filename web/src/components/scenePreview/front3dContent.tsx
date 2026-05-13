import { useEffect, useMemo, useState } from "react";
import type { Renderable3dFrontSceneManifest, SceneManifest } from "../../types";
import {
  BatchedAssetModels,
  Bounds,
  type AssetPlacement,
  type InspectableObject,
  type RenderProgressSnapshot,
  type SceneBounds,
  type Vector3Tuple,
  type WallDisplayMode,
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
  roomShellOpacity,
  updateMeasuredBoundsMap,
} from "./shared";

function computeFront3dBounds(
  scene: SceneManifest | null,
  measuredShellBounds: Record<string, SceneBounds>,
): SceneBounds {
  const min: Vector3Tuple = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
  const max: Vector3Tuple = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
  const minMax = { min, max };

  function includeBox(center: Vector3Tuple, size: Vector3Tuple) {
    const half: Vector3Tuple = [size[0] / 2, size[1] / 2, size[2] / 2];
    expandBounds(minMax, [center[0] - half[0], center[1] - half[1], center[2] - half[2]]);
    expandBounds(minMax, [center[0] + half[0], center[1] + half[1], center[2] + half[2]]);
  }

  for (const bounds of Object.values(measuredShellBounds)) {
    includeBox(bounds.center, bounds.size);
  }

  if (scene) {
    for (const room of scene.normalized.rooms) {
      const width = Math.max(room.dimensions?.width ?? 2, 1);
      const length = Math.max(room.dimensions?.length ?? 2, 1);
      const height = Math.max(room.dimensions?.height ?? room.ceiling_height ?? 2.8, 2);
      includeBox(
        [room.position?.x ?? 0, (room.position?.y ?? 0) + height / 2, room.position?.z ?? 0],
        [width, height, length],
      );
    }

    for (const object of scene.normalized.objects) {
      if (!object.bbox_min || !object.bbox_max) {
        continue;
      }
      expandBounds(minMax, object.bbox_min);
      expandBounds(minMax, object.bbox_max);
    }
  }

  if (!Number.isFinite(min[0]) || !Number.isFinite(max[0])) {
    return createEmptyBounds();
  }

  return finalizeBounds(min, max);
}

export function Front3DPreviewContent({
  scene,
  renderScene,
  wallOpacity,
  wallDisplayMode,
  showObjectLabels,
  onRenderProgressChange,
}: {
  scene: SceneManifest | null;
  renderScene: Renderable3dFrontSceneManifest;
  wallOpacity: number;
  wallDisplayMode: WallDisplayMode;
  showObjectLabels: boolean;
  onRenderProgressChange: (snapshot: RenderProgressSnapshot) => void;
}) {
  const [hoveredObjectId, setHoveredObjectId] = useState<string | null>(null);
  const [selectedObjectId, setSelectedObjectId] = useState<string | null>(null);
  const [fitVersion, setFitVersion] = useState(0);
  const [shellsReady, setShellsReady] = useState(renderScene.room_shells.length === 0);
  const [measuredShellBounds, setMeasuredShellBounds] = useState<Record<string, SceneBounds>>({});
  const [measuredObjectBounds, setMeasuredObjectBounds] = useState<Record<string, SceneBounds>>({});
  const shellAssets = useMemo(
    () =>
      renderScene.room_shells.map(
        (shell): AssetPlacement => {
          const shellOpacity = roomShellOpacity(shell.category, wallDisplayMode, wallOpacity);
          const hidden =
            (shell.category === "wall" ||
              shell.category === "window" ||
              shell.category === "door" ||
              shell.category === "feature") &&
            wallDisplayMode === "hidden";
          return {
            key: `${shell.id}::${shell.asset_path}`,
            assetPath: shell.asset_path,
            position: shell.position,
            rotationYDeg: shell.rotation_y_deg,
            scale: shell.scale,
            opacity: shellOpacity,
            wireframe:
              (shell.category === "wall" ||
                shell.category === "window" ||
                shell.category === "door" ||
                shell.category === "feature") &&
              wallDisplayMode === "wireframe",
            visible: !hidden,
            doubleSided: shell.category !== "floor",
            transparentDepthWrite: false,
            forceSinglePass: shellOpacity < 0.999,
            polygonOffset: shellOpacity < 0.999,
            polygonOffsetFactor: shellOpacity < 0.999 ? -1 : 0,
            polygonOffsetUnits: shellOpacity < 0.999 ? -1 : 0,
          };
        },
      ),
    [renderScene, wallDisplayMode, wallOpacity],
  );
  const objectAssets = useMemo(
    () =>
      renderScene.objects.map(
        (object): AssetPlacement => ({
          key: object.id,
          assetPath: object.asset_path,
          position: object.position,
          rotationYDeg: object.rotation_y_deg,
          scale: object.scale,
          quaternion: object.quaternion,
        }),
      ),
    [renderScene],
  );
  const [shellBatchProgress, setShellBatchProgress] = useState(() =>
    createEmptyBatchProgress(renderScene.room_shells.length),
  );
  const [objectBatchProgress, setObjectBatchProgress] = useState(() =>
    createEmptyBatchProgress(renderScene.objects.length),
  );
  const sceneBounds = useMemo(
    () => computeFront3dBounds(scene, measuredShellBounds),
    [measuredShellBounds, scene],
  );
  const inspectableObjects = useMemo((): InspectableObject[] => {
    return renderScene.objects.map((object) => {
      const measured = measuredObjectBounds[object.id];
      const sourceObject = (scene?.normalized.objects ?? []).find((item) => item.id === object.id);
      const bboxMin = sourceObject?.bbox_min ?? null;
      const bboxMax = sourceObject?.bbox_max ?? null;
      const preferredLabel =
        sourceObject?.name || sourceObject?.type || object.object_type || object.description || object.id;

      if (bboxMin && bboxMax) {
        return {
          id: object.id,
          label: labelText(preferredLabel, object.id),
          position: measured?.center ?? [
            (bboxMin[0] + bboxMax[0]) / 2,
            (bboxMin[1] + bboxMax[1]) / 2,
            (bboxMin[2] + bboxMax[2]) / 2,
          ],
          size: measured?.size ?? [
            Math.max(Math.abs(bboxMax[0] - bboxMin[0]), 0.18),
            Math.max(Math.abs(bboxMax[1] - bboxMin[1]), 0.18),
            Math.max(Math.abs(bboxMax[2] - bboxMin[2]), 0.18),
          ],
        };
      }

      return {
        id: object.id,
        label: labelText(preferredLabel, object.id),
        position: measured?.center ?? object.position,
        size: measured?.size ?? [
          Math.max(Math.abs(object.scale[0]), 0.45),
          Math.max(Math.abs(object.scale[1]), 0.45),
          Math.max(Math.abs(object.scale[2]), 0.45),
        ],
      };
    });
  }, [measuredObjectBounds, renderScene, scene]);
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
    if (!shellsReady) {
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
    shellsReady,
  ]);

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
          key={`${renderScene.scene_uid}::3dfront-shells`}
          items={shellAssets}
          batchSize={24}
          materialProfile="3dfront"
          onItemBounds={(id, bounds) =>
            setMeasuredShellBounds((current) => updateMeasuredBoundsMap(current, id, bounds))
          }
          onProgress={setShellBatchProgress}
          onComplete={() => {
            setShellsReady(true);
            setFitVersion((current) => current + 1);
          }}
        />
        {shellsReady ? (
          <>
            <BatchedAssetModels
              key={`${renderScene.scene_uid}::3dfront-objects`}
              items={objectAssets}
              batchSize={20}
              onItemBounds={(id, bounds) =>
                setMeasuredObjectBounds((current) => updateMeasuredBoundsMap(current, id, bounds))
              }
              materialProfile="3dfront"
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
