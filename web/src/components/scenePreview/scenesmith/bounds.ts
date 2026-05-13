import type { SceneManifest } from "../../../types";
import {
  createEmptyBounds,
  expandBounds,
  finalizeBounds,
  type SceneBounds,
  type Vector3Tuple,
} from "../shared";

export function computeSceneSmithBounds(
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
