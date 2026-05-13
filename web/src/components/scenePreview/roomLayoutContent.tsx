import { useEffect, useMemo, useState } from "react";
import type {
  RenderableHsmSceneManifest,
  RenderableSageSceneManifest,
} from "../../types";
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
  SageOpenings,
  SageRoomShell,
  SceneBoundsController,
  SelectionOverlays,
  buildObjectLabels,
  computeRoomFootprint,
  createEmptyBatchProgress,
  createEmptyBounds,
  expandBounds,
  finalizeBounds,
  labelText,
  updateMeasuredBoundsMap,
} from "./shared";

type RoomLayoutRenderScene = RenderableSageSceneManifest | RenderableHsmSceneManifest;

function computeRoomLayoutBounds(renderScene: RoomLayoutRenderScene): SceneBounds {
  const min: Vector3Tuple = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
  const max: Vector3Tuple = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
  const minMax = { min, max };

  function includeBox(center: Vector3Tuple, size: Vector3Tuple) {
    const half: Vector3Tuple = [size[0] / 2, size[1] / 2, size[2] / 2];
    expandBounds(minMax, [center[0] - half[0], center[1] - half[1], center[2] - half[2]]);
    expandBounds(minMax, [center[0] + half[0], center[1] + half[1], center[2] + half[2]]);
  }

  for (const room of renderScene.rooms) {
    const footprint = computeRoomFootprint(room);
    const ceilingHeight = Math.max(room.ceiling_height ?? room.dimensions?.height ?? 2.8, 2);
    includeBox(
      [footprint.center[0], ceilingHeight / 2, footprint.center[1]],
      [footprint.size[0], ceilingHeight, footprint.size[1]],
    );
  }

  if (renderScene.dataset === "sage") {
    for (const object of renderScene.objects) {
      includeBox(object.position, [
        Math.max(object.native_size[0] * object.scale[0], 0.2),
        Math.max(object.native_size[1] * object.scale[1], 0.2),
        Math.max(object.native_size[2] * object.scale[2], 0.2),
      ]);
    }
  } else {
    for (const object of renderScene.objects) {
      includeBox(object.position, [
        Math.max(Math.abs(object.scale[0]), 0.45),
        Math.max(Math.abs(object.scale[1]), 0.45),
        Math.max(Math.abs(object.scale[2]), 0.45),
      ]);
    }
  }

  if (!Number.isFinite(min[0]) || !Number.isFinite(max[0])) {
    return createEmptyBounds();
  }

  return finalizeBounds(min, max);
}

export function RoomLayoutPreviewContent({
  renderScene,
  wallOpacity,
  wallDisplayMode,
  showObjectLabels,
  onRenderProgressChange,
}: {
  renderScene: RoomLayoutRenderScene;
  wallOpacity: number;
  wallDisplayMode: WallDisplayMode;
  showObjectLabels: boolean;
  onRenderProgressChange: (snapshot: RenderProgressSnapshot) => void;
}) {
  const [hoveredObjectId, setHoveredObjectId] = useState<string | null>(null);
  const [selectedObjectId, setSelectedObjectId] = useState<string | null>(null);
  const [fitVersion, setFitVersion] = useState(0);
  const [measuredObjectBounds, setMeasuredObjectBounds] = useState<Record<string, SceneBounds>>({});
  const roomLayoutAssets = useMemo(() => {
    if (renderScene.dataset === "sage") {
      return renderScene.objects.map(
        (object): AssetPlacement => ({
          key: object.id,
          assetPath: object.asset_path,
          position: object.position,
          rotationYDeg: object.rotation_y_deg,
          scale: object.scale,
        }),
      );
    }

    return renderScene.objects.map(
      (object): AssetPlacement => ({
        key: object.id,
        assetPath: object.asset_path,
        position: object.position,
        rotationYDeg: object.rotation_y_deg,
        quaternion: object.quaternion,
        scale: object.scale,
      }),
    );
  }, [renderScene]);
  const [batchProgress, setBatchProgress] = useState(() =>
    createEmptyBatchProgress(roomLayoutAssets.length),
  );
  const sceneBounds = useMemo(() => computeRoomLayoutBounds(renderScene), [renderScene]);
  const inspectableObjects = useMemo((): InspectableObject[] => {
    if (renderScene.dataset === "sage") {
      return renderScene.objects.map((object) => {
        const measured = measuredObjectBounds[object.id];
        return {
          id: object.id,
          label: labelText(object.type || object.description, object.id),
          position: measured?.center ?? object.position,
          size: measured?.size ?? [
            Math.max(object.native_size[0] * object.scale[0], 0.18),
            Math.max(object.native_size[1] * object.scale[1], 0.18),
            Math.max(object.native_size[2] * object.scale[2], 0.18),
          ],
        };
      });
    }

    return renderScene.objects.map((object) => {
      const measured = measuredObjectBounds[object.id];
      const preferredLabel =
        object.name ||
        object.semantic_label ||
        object.object_type ||
        object.description ||
        object.category ||
        object.source_id ||
        object.id;
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
  }, [measuredObjectBounds, renderScene]);
  const objectLabels = useMemo(
    () => buildObjectLabels(inspectableObjects, showObjectLabels, selectedObjectId, hoveredObjectId),
    [hoveredObjectId, inspectableObjects, selectedObjectId, showObjectLabels],
  );

  useEffect(() => {
    const total = roomLayoutAssets.length;
    const completed = Math.min(batchProgress.readyCount, total);
    const ready = total === 0 || batchProgress.complete;
    const progress = total === 0 ? 100 : Math.round((completed / total) * 100);
    onRenderProgressChange({
      ready,
      stage: ready ? "Scene ready" : "Preparing objects",
      detail:
        total === 0 ? "No object assets need staged rendering" : `Mounted ${completed}/${total} object assets`,
      completed,
      total,
      progress,
    });
  }, [batchProgress.complete, batchProgress.readyCount, onRenderProgressChange, roomLayoutAssets.length]);

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
        {renderScene.rooms.map((room) => (
          <SageRoomShell
            key={room.id}
            room={room}
            wallOpacity={wallOpacity}
            wallDisplayMode={wallDisplayMode}
          />
        ))}
        <SageOpenings rooms={renderScene.rooms} />
        <BatchedAssetModels
          key={`${renderScene.scene_uid}::room-layout-objects`}
          items={roomLayoutAssets}
          batchSize={24}
          onItemBounds={(id, bounds) =>
            setMeasuredObjectBounds((current) => updateMeasuredBoundsMap(current, id, bounds))
          }
          materialProfile={renderScene.dataset === "sage" ? "sage" : "hsm"}
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
