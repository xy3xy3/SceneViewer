import { useEffect, useState } from "react";
import { loadSceneSmithShellTransforms, type SceneSmithShellTransform } from "../shared";

export function useSceneSmithShellTransforms(roomGeometryPaths: string[]) {
  const needsShellTransforms = roomGeometryPaths.length > 0;
  const [shellTransformsLoaded, setShellTransformsLoaded] = useState(!needsShellTransforms);
  const [shellTransformMap, setShellTransformMap] = useState<Record<string, SceneSmithShellTransform>>({});

  useEffect(() => {
    if (!needsShellTransforms) {
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

  return {
    needsShellTransforms,
    shellTransformsLoaded,
    shellTransformMap,
  };
}
