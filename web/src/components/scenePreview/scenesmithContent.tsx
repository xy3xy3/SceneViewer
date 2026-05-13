import { useEffect, useMemo, useState } from "react";
import type { RenderableSceneSmithSceneManifest, SceneManifest } from "../../types";
import {
  BatchedAssetModels,
  Bounds,
  type AssetPlacement,
  type InspectableObject,
  type RenderProgressSnapshot,
  type SceneBounds,
  type SceneSmithShellTransform,
  type Vector3Tuple,
  type WallDisplayMode,
  ObjectHitTargets,
  ObjectLabels,
  SceneBoundsController,
  SelectionOverlays,
  buildObjectLabels,
  compactSceneSmithName,
  createEmptyBatchProgress,
  createEmptyBounds,
  expandBounds,
  finalizeBounds,
  labelText,
  loadSceneSmithShellTransforms,
  resolveWallOpacity,
  sceneSmithToThree,
  updateMeasuredBoundsMap,
} from "./shared";

function computeSceneSmithBounds(
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
      const translation = room.frame?.translation ?? [0, 0, 0];
      const width = Math.max(room.dimensions?.width ?? 2, 1);
      const length = Math.max(room.dimensions?.length ?? 2, 1);
      const height = Math.max(room.dimensions?.height ?? room.ceiling_height ?? 2.8, 2);
      includeBox([translation[0], translation[1] + height / 2, translation[2]], [width, height, length]);
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

export function SceneSmithPreviewContent({
  scene,
  renderScene,
  wallOpacity,
  wallDisplayMode,
  showObjectLabels,
  onRenderProgressChange,
}: {
  scene: SceneManifest | null;
  renderScene: RenderableSceneSmithSceneManifest;
  wallOpacity: number;
  wallDisplayMode: WallDisplayMode;
  showObjectLabels: boolean;
  onRenderProgressChange: (snapshot: RenderProgressSnapshot) => void;
}) {
  const [hoveredObjectId, setHoveredObjectId] = useState<string | null>(null);
  const [selectedObjectId, setSelectedObjectId] = useState<string | null>(null);
  const [fitVersion, setFitVersion] = useState(0);
  const roomGeometryPaths = useMemo(
    () =>
      (scene?.normalized.rooms ?? [])
        .map((room) => room.room_geometry_sdf)
        .filter((value): value is string => Boolean(value)),
    [scene],
  );
  const needsShellTransforms = roomGeometryPaths.length > 0;
  const [shellsReady, setShellsReady] = useState(renderScene.room_shells.length === 0);
  const [shellTransformsLoaded, setShellTransformsLoaded] = useState(!needsShellTransforms);
  const [shellTransformMap, setShellTransformMap] = useState<Record<string, SceneSmithShellTransform>>({});
  const [measuredShellBounds, setMeasuredShellBounds] = useState<Record<string, SceneBounds>>({});
  const [measuredObjectBounds, setMeasuredObjectBounds] = useState<Record<string, SceneBounds>>({});
  const shellAssets = useMemo(() => {
    return renderScene.room_shells.map(
      (shell): AssetPlacement => {
        const transform = shellTransformMap[shell.asset_path];
        const wallOpacityValue = resolveWallOpacity(wallDisplayMode, wallOpacity);
        const position = transform
          ? ([
              shell.position[0] + transform.position[0],
              shell.position[1] + transform.position[1],
              shell.position[2] + transform.position[2],
            ] as Vector3Tuple)
          : shell.position;

        return {
          key: `${shell.id}::${shell.asset_path}`,
          assetPath: shell.asset_path,
          position,
          rotationYDeg: shell.rotation_y_deg + (transform?.rotationYDeg ?? 0),
          scale: shell.scale,
          opacity: shell.category === "wall" ? wallOpacityValue : 1,
          wireframe: shell.category === "wall" && wallDisplayMode === "wireframe",
          visible: shell.category !== "wall" || wallDisplayMode !== "hidden",
          doubleSided: shell.category === "wall" || shell.category === "window",
          transparentDepthWrite: false,
          forceSinglePass: shell.category === "wall" || shell.category === "window",
          polygonOffset: shell.category === "wall" && wallOpacityValue < 0.999,
          polygonOffsetFactor: shell.category === "wall" ? -1 : 0,
          polygonOffsetUnits: shell.category === "wall" ? -1 : 0,
        };
      },
    );
  }, [renderScene, shellTransformMap, wallDisplayMode, wallOpacity]);
  const objectAssets = useMemo(
    () =>
      renderScene.objects.map(
        (object): AssetPlacement => ({
          key: object.id,
          assetPath: object.asset_path,
          position: object.position,
          rotationYDeg: object.rotation_y_deg,
          scale: object.scale,
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
    () => computeSceneSmithBounds(scene, measuredShellBounds),
    [measuredShellBounds, scene],
  );
  const inspectableObjects = useMemo((): InspectableObject[] => {
    const sourceObjects = new Map((scene?.normalized.objects ?? []).map((object) => [object.id, object] as const));
    return renderScene.objects.map((object) => {
      const measured = measuredObjectBounds[object.id];
      const sourceObject = sourceObjects.get(object.id);
      const bboxMin = sourceObject?.bbox_min ?? null;
      const bboxMax = sourceObject?.bbox_max ?? null;
      const preferredLabel =
        compactSceneSmithName(sourceObject?.name, sourceObject?.room_id) ||
        compactSceneSmithName(object.description, sourceObject?.room_id) ||
        sourceObject?.object_type ||
        object.object_type ||
        sourceObject?.description ||
        object.description ||
        object.id;

      if (bboxMin && bboxMax) {
        return {
          id: object.id,
          label: labelText(preferredLabel, object.id),
          position:
            measured?.center ??
            sceneSmithToThree([
              (bboxMin[0] + bboxMax[0]) / 2,
              (bboxMin[1] + bboxMax[1]) / 2,
              (bboxMin[2] + bboxMax[2]) / 2,
            ]),
          size: measured?.size ?? [
            Math.max(Math.abs(bboxMax[0] - bboxMin[0]), 0.18),
            Math.max(Math.abs(bboxMax[2] - bboxMin[2]), 0.18),
            Math.max(Math.abs(bboxMax[1] - bboxMin[1]), 0.18),
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
    if (!needsShellTransforms) {
      setShellTransformsLoaded(true);
      setShellTransformMap({});
      return;
    }

    let cancelled = false;

    async function loadTransforms() {
      try {
        const nextMap = await loadSceneSmithShellTransforms(roomGeometryPaths);
        if (!cancelled) {
          setShellTransformMap(nextMap);
        }
      } catch {
        if (!cancelled) {
          setShellTransformMap({});
        }
      } finally {
        if (!cancelled) {
          setShellTransformsLoaded(true);
        }
      }
    }

    void loadTransforms();

    return () => {
      cancelled = true;
    };
  }, [needsShellTransforms, roomGeometryPaths]);

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
