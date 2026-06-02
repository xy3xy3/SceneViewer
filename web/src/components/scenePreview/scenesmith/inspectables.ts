import type { RenderableSceneSmithSceneManifest, SceneManifest } from "../../../types";
import {
  compactSceneSmithName,
  labelText,
  resolveObjectPosition,
  sceneSmithToThree,
  type InspectableObject,
  type SceneBounds,
  type Vector3Tuple,
} from "../shared";

export function buildSceneSmithInspectableObjects({
  scene,
  renderScene,
  measuredObjectBounds,
  positionOverrides,
}: {
  scene: SceneManifest | null;
  renderScene: RenderableSceneSmithSceneManifest;
  measuredObjectBounds: Record<string, SceneBounds>;
  positionOverrides?: Record<string, Vector3Tuple>;
}): InspectableObject[] {
  const sourceObjects = new Map((scene?.normalized.objects ?? []).map((object) => [object.id, object] as const));

  return renderScene.objects.map((object) => {
    const measured = measuredObjectBounds[object.id];
    const resolvedPosition = resolveObjectPosition(object.id, object.position, positionOverrides);
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
      position: measured?.center ?? resolvedPosition,
      size: measured?.size ?? [
        Math.max(Math.abs(object.scale[0]), 0.45),
        Math.max(Math.abs(object.scale[1]), 0.45),
        Math.max(Math.abs(object.scale[2]), 0.45),
      ],
    };
  });
}
