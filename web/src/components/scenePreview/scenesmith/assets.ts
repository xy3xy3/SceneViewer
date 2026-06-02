import type { RenderableSceneSmithSceneManifest, SceneManifest } from "../../../types";
import {
  resolveWallOpacity,
  resolveObjectPosition,
  type AssetPlacement,
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
): AssetPlacement[] {
  return renderScene.objects.map((object): AssetPlacement => ({
    key: object.id,
    assetPath: object.asset_path,
    position: resolveObjectPosition(object.id, object.position, positionOverrides),
    rotationYDeg: object.rotation_y_deg,
    quaternion: object.quaternion,
    scale: object.scale,
  }));
}
