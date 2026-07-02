import type { RenderableSceneSmithSceneManifest, SceneManifest } from "../../../types";
import {
  resolveWallOpacity,
  resolveObjectPosition,
  resolveObjectQuaternion,
  resolveObjectRotation,
  type AssetPlacement,
  type InspectableObject,
  type ObjectForwardArrowPlacement,
  type QuaternionTuple,
  type SceneSmithShellTransform,
  type Vector3Tuple,
  type WallDisplayMode,
} from "../shared";

export function sceneSmithRoomGeometryPaths(scene: SceneManifest | null): string[] {
  return (scene?.normalized.rooms ?? [])
    .map((room) => room.room_geometry_sdf)
    .filter((value): value is string => Boolean(value));
}

export function buildSceneSmithShellAssets({
  renderScene,
  shellTransformMap,
  shellTransformsLoaded,
  wallDisplayMode,
  wallOpacity,
}: {
  renderScene: RenderableSceneSmithSceneManifest;
  shellTransformMap: Record<string, SceneSmithShellTransform>;
  shellTransformsLoaded: boolean;
  wallDisplayMode: WallDisplayMode;
  wallOpacity: number;
}): AssetPlacement[] {
  return renderScene.room_shells.flatMap((shell): AssetPlacement[] => {
    const transform = shellTransformMap[shell.asset_path];
    if (shell.category === "window" && shellTransformsLoaded && !transform) {
      return [];
    }

    const wallOpacityValue = resolveWallOpacity(wallDisplayMode, wallOpacity);
    const position = transform
      ? ([
          shell.position[0] + transform.position[0],
          shell.position[1] + transform.position[1],
          shell.position[2] + transform.position[2],
        ] as Vector3Tuple)
      : shell.position;

    return [
      {
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
      },
    ];
  });
}

export function buildSceneSmithObjectAssets(
  renderScene: RenderableSceneSmithSceneManifest,
  positionOverrides?: Record<string, Vector3Tuple>,
  rotationOverrides?: Record<string, number>,
  quaternionOverrides?: Record<string, QuaternionTuple>,
): AssetPlacement[] {
  return renderScene.objects.map((object): AssetPlacement => ({
    key: object.id,
    assetPath: object.asset_path,
    position: resolveObjectPosition(object.id, object.position, positionOverrides),
    rotationYDeg: resolveObjectRotation(object.id, object.rotation_y_deg, rotationOverrides),
    quaternion: resolveObjectQuaternion(object.id, object.quaternion, quaternionOverrides),
    scale: object.scale,
  }));
}

function isVector3Tuple(value: unknown): value is Vector3Tuple {
  return (
    Array.isArray(value) &&
    value.length >= 3 &&
    value.slice(0, 3).every((entry) => typeof entry === "number" && Number.isFinite(entry))
  );
}

export function buildSceneSmithForwardArrowPlacements({
  renderScene,
  inspectableObjects,
  selectedObjectId,
  positionOverrides,
}: {
  renderScene: RenderableSceneSmithSceneManifest;
  inspectableObjects: InspectableObject[];
  selectedObjectId: string | null;
  positionOverrides?: Record<string, Vector3Tuple>;
}): ObjectForwardArrowPlacement[] {
  const inspectableById = new Map(inspectableObjects.map((object) => [object.id, object] as const));

  return renderScene.objects.flatMap((object): ObjectForwardArrowPlacement[] => {
    if (!isVector3Tuple(object.forward_direction)) {
      return [];
    }

    const inspectable = inspectableById.get(object.id);
    return [
      {
        id: object.id,
        position:
          inspectable?.position ??
          resolveObjectPosition(object.id, object.position, positionOverrides),
        direction: object.forward_direction,
        size: inspectable?.size ?? [
          Math.max(Math.abs(object.scale[0]), 0.45),
          Math.max(Math.abs(object.scale[1]), 0.45),
          Math.max(Math.abs(object.scale[2]), 0.45),
        ],
        active: object.id === selectedObjectId,
      },
    ];
  });
}
