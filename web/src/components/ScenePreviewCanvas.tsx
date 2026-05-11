import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import {
  Bounds,
  Environment,
  Grid,
  Lightformer,
  OrbitControls,
  useBounds,
  useGLTF,
  useTexture,
} from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import * as THREE from "three";
import { toRepoAssetUrl } from "../lib/repoAssets";
import type {
  RenderableSceneManifest,
  RenderableSageDoor,
  RenderableSageRoom,
  SceneManifest,
  SageDoor,
  SageWall,
  SageWindow,
} from "../types";

interface ScenePreviewCanvasProps {
  scene: SceneManifest | null;
  renderScene: RenderableSceneManifest | null;
}

type Vector3Tuple = [number, number, number];

type WallPanel = {
  id: string;
  position: Vector3Tuple;
  rotationY: number;
  size: Vector3Tuple;
};

type MarkerPanel = {
  id: string;
  position: Vector3Tuple;
  rotationY: number;
  size: Vector3Tuple;
  color: string;
};

type RoomFootprint = {
  center: [number, number];
  size: [number, number];
};

type AssetPlacement = {
  key: string;
  assetPath: string;
  position: Vector3Tuple;
  rotationYDeg: number;
  scale: Vector3Tuple;
};

type SceneBounds = {
  center: Vector3Tuple;
  size: Vector3Tuple;
};

type MaterialProfile = "sage" | "scenesmith";

function wallToPanel(wall: SageWall): WallPanel {
  const start: Vector3Tuple = [wall.start_point.x, 0, wall.start_point.y];
  const end: Vector3Tuple = [wall.end_point.x, 0, wall.end_point.y];
  const dx = end[0] - start[0];
  const dz = end[2] - start[2];
  const length = Math.max(Math.hypot(dx, dz), 0.05);
  const rotationY = Math.atan2(dz, dx);
  return {
    id: wall.id,
    position: [(start[0] + end[0]) / 2, wall.height / 2, (start[2] + end[2]) / 2],
    rotationY,
    size: [length, wall.height, Math.max(wall.thickness, 0.06)],
  };
}

function openingToMarker(
  opening: SageDoor | SageWindow | RenderableSageDoor,
  wall: SageWall | undefined,
  color: string,
): MarkerPanel | null {
  if (!wall) {
    return null;
  }
  const dx = wall.end_point.x - wall.start_point.x;
  const dz = wall.end_point.y - wall.start_point.y;
  const yaw = Math.atan2(dz, dx);
  const t = opening.position_on_wall;
  const centerX = wall.start_point.x + dx * t;
  const centerZ = wall.start_point.y + dz * t;
  return {
    id: opening.id,
    position: [centerX, opening.height / 2, centerZ],
    rotationY: yaw,
    size: [Math.max(opening.width, 0.2), opening.height, 0.08],
    color,
  };
}

function computeRoomFootprint(room: RenderableSageRoom): RoomFootprint {
  const points = room.walls.flatMap((wall) => [
    [wall.start_point.x, wall.start_point.y] as const,
    [wall.end_point.x, wall.end_point.y] as const,
  ]);

  if (points.length === 0) {
    const width = Math.max(room.dimensions?.width ?? 1, 0.5);
    const length = Math.max(room.dimensions?.length ?? 1, 0.5);
    return {
      center: [width / 2, length / 2],
      size: [width, length],
    };
  }

  const xs = points.map(([x]) => x);
  const zs = points.map(([, z]) => z);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minZ = Math.min(...zs);
  const maxZ = Math.max(...zs);
  return {
    center: [(minX + maxX) / 2, (minZ + maxZ) / 2],
    size: [Math.max(maxX - minX, 0.5), Math.max(maxZ - minZ, 0.5)],
  };
}

function prepareScene(root: THREE.Object3D, profile: MaterialProfile): THREE.Object3D {
  const clone = root.clone(true);
  clone.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      child.castShadow = false;
      child.receiveShadow = false;

      if (child.geometry) {
        if (!child.geometry.getAttribute("normal")) {
          child.geometry.computeVertexNormals();
        }
        child.geometry.normalizeNormals();
        child.geometry.computeBoundingSphere();
      }

      const sourceMaterials = Array.isArray(child.material) ? child.material : [child.material];
      const nextMaterials = sourceMaterials.map((sourceMaterial) => {
        if (!sourceMaterial) {
          return sourceMaterial;
        }

        const material = sourceMaterial.clone();
        if (material instanceof THREE.MeshStandardMaterial) {
          if (material.map) {
            material.map.colorSpace = THREE.SRGBColorSpace;
            material.map.needsUpdate = true;
          }
          if (profile === "sage") {
            material.metalness = 0;
            material.roughness = 1;
          } else {
            material.metalness = Math.min(material.metalness ?? 0, 0.08);
            material.roughness = Math.max(material.roughness ?? 0.92, 0.78);
          }
          material.envMapIntensity = 0.9;
          material.side = THREE.FrontSide;
        }
        return material;
      });

      child.material = Array.isArray(child.material) ? nextMaterials : nextMaterials[0];
      child.frustumCulled = false;
    }
  });
  return clone;
}

function PreviewEnvironment() {
  return (
    <Environment resolution={128}>
      <Lightformer
        form="rect"
        intensity={2.4}
        color="#fff7ed"
        position={[8, 10, 6]}
        rotation={[-Math.PI / 5, Math.PI / 3, 0]}
        scale={[10, 10, 1]}
      />
      <Lightformer
        form="rect"
        intensity={1.8}
        color="#e0f2fe"
        position={[-9, 7, -6]}
        rotation={[-Math.PI / 6, -Math.PI / 3, 0]}
        scale={[12, 12, 1]}
      />
      <Lightformer
        form="ring"
        intensity={1.2}
        color="#f8fafc"
        position={[0, 12, 0]}
        scale={10}
      />
    </Environment>
  );
}

function useTiledTexture(textureUrl: string, repeatX: number, repeatY: number) {
  const texture = useTexture(textureUrl);
  return useMemo(() => {
    const tiled = texture.clone();
    tiled.colorSpace = THREE.SRGBColorSpace;
    tiled.wrapS = THREE.RepeatWrapping;
    tiled.wrapT = THREE.RepeatWrapping;
    tiled.repeat.set(Math.max(repeatX, 1), Math.max(repeatY, 1));
    tiled.needsUpdate = true;
    return tiled;
  }, [repeatX, repeatY, texture]);
}

function TexturedFloor({
  center,
  size,
  textureUrl,
}: {
  center: [number, number];
  size: [number, number];
  textureUrl: string;
}) {
  const texture = useTiledTexture(textureUrl, size[0] / 1.4, size[1] / 1.4);
  return (
    <mesh
      position={[center[0], 0.005, center[1]]}
      rotation={[-Math.PI / 2, 0, 0]}
      receiveShadow
    >
      <planeGeometry args={size} />
      <meshStandardMaterial map={texture} roughness={0.92} metalness={0.02} />
    </mesh>
  );
}

function TexturedWall({
  panel,
  textureUrl,
}: {
  panel: WallPanel;
  textureUrl: string;
}) {
  const texture = useTiledTexture(textureUrl, panel.size[0] / 1.2, panel.size[1] / 1.2);
  return (
    <mesh position={panel.position} rotation={[0, -panel.rotationY, 0]} castShadow receiveShadow>
      <boxGeometry args={panel.size} />
      <meshStandardMaterial map={texture} roughness={0.96} metalness={0.02} />
    </mesh>
  );
}

function AssetModel({
  assetPath,
  position,
  rotationYDeg,
  scale,
  onReady,
  materialProfile,
}: {
  assetPath: string;
  position: Vector3Tuple;
  rotationYDeg: number;
  scale: Vector3Tuple;
  onReady?: () => void;
  materialProfile: MaterialProfile;
}) {
  const url = toRepoAssetUrl(assetPath);
  if (!url) {
    return null;
  }

  const gltf = useGLTF(url);
  const object = useMemo(() => prepareScene(gltf.scene, materialProfile), [gltf.scene, materialProfile]);

  useEffect(() => {
    onReady?.();
  }, [onReady]);

  return (
    <primitive
      object={object}
      position={position}
      rotation={[0, THREE.MathUtils.degToRad(rotationYDeg), 0]}
      scale={scale}
    />
  );
}

function BatchedAssetModels({
  items,
  batchSize,
  onComplete,
  materialProfile,
}: {
  items: AssetPlacement[];
  batchSize: number;
  onComplete?: () => void;
  materialProfile: MaterialProfile;
}) {
  const [visibleCount, setVisibleCount] = useState(Math.min(batchSize, items.length));
  const [readyCount, setReadyCount] = useState(0);
  const loadedKeysRef = useRef<Set<string>>(new Set());
  const completedRef = useRef(false);

  useEffect(() => {
    loadedKeysRef.current = new Set();
    completedRef.current = false;
    setVisibleCount(Math.min(batchSize, items.length));
    setReadyCount(0);
  }, [batchSize, items]);

  useEffect(() => {
    if (visibleCount >= items.length || readyCount < visibleCount) {
      return;
    }

    const timer = window.setTimeout(() => {
      setVisibleCount((current) => Math.min(current + batchSize, items.length));
    }, 120);

    return () => {
      window.clearTimeout(timer);
    };
  }, [batchSize, items.length, readyCount, visibleCount]);

  function handleReady(key: string) {
    if (loadedKeysRef.current.has(key)) {
      return;
    }
    loadedKeysRef.current.add(key);
    setReadyCount(loadedKeysRef.current.size);
  }

  useEffect(() => {
    if (completedRef.current || items.length === 0 || readyCount < items.length) {
      return;
    }
    completedRef.current = true;
    onComplete?.();
  }, [items.length, onComplete, readyCount]);

  return (
    <>
      {items.slice(0, visibleCount).map((item) => (
        <Suspense key={item.key} fallback={null}>
          <AssetModel
            assetPath={item.assetPath}
            position={item.position}
            rotationYDeg={item.rotationYDeg}
            scale={item.scale}
            onReady={() => handleReady(item.key)}
            materialProfile={materialProfile}
          />
        </Suspense>
      ))}
    </>
  );
}

function SceneBoundsController({
  sceneKey,
  fitVersion,
}: {
  sceneKey: string;
  fitVersion: number;
}) {
  const bounds = useBounds();

  useEffect(() => {
    let cancelled = false;
    const frame = window.requestAnimationFrame(() => {
      window.requestAnimationFrame(() => {
        if (cancelled) {
          return;
        }
        bounds.refresh().clip().fit();
      });
    });
    return () => {
      cancelled = true;
      window.cancelAnimationFrame(frame);
    };
  }, [bounds, fitVersion, sceneKey]);

  return null;
}

function createEmptyBounds(): SceneBounds {
  return {
    center: [0, 1.5, 0],
    size: [2, 3, 2],
  };
}

function expandBounds(
  bounds: {
    min: Vector3Tuple;
    max: Vector3Tuple;
  },
  point: Vector3Tuple,
) {
  bounds.min[0] = Math.min(bounds.min[0], point[0]);
  bounds.min[1] = Math.min(bounds.min[1], point[1]);
  bounds.min[2] = Math.min(bounds.min[2], point[2]);
  bounds.max[0] = Math.max(bounds.max[0], point[0]);
  bounds.max[1] = Math.max(bounds.max[1], point[1]);
  bounds.max[2] = Math.max(bounds.max[2], point[2]);
}

function finalizeBounds(min: Vector3Tuple, max: Vector3Tuple): SceneBounds {
  const sizeX = Math.max(max[0] - min[0], 1);
  const sizeY = Math.max(max[1] - min[1], 1);
  const sizeZ = Math.max(max[2] - min[2], 1);
  return {
    center: [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2],
    size: [sizeX, sizeY, sizeZ],
  };
}

function computeSceneBounds(
  scene: SceneManifest | null,
  renderScene: RenderableSceneManifest,
): SceneBounds {
  const min: Vector3Tuple = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
  const max: Vector3Tuple = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];

  function includeBox(center: Vector3Tuple, size: Vector3Tuple) {
    const half: Vector3Tuple = [size[0] / 2, size[1] / 2, size[2] / 2];
    expandBounds(minMax, [center[0] - half[0], center[1] - half[1], center[2] - half[2]]);
    expandBounds(minMax, [center[0] + half[0], center[1] + half[1], center[2] + half[2]]);
  }

  const minMax = { min, max };

  if (renderScene.dataset === "sage") {
    for (const room of renderScene.rooms) {
      const footprint = computeRoomFootprint(room);
      const ceilingHeight = Math.max(room.ceiling_height ?? room.dimensions?.height ?? 2.8, 2);
      includeBox(
        [footprint.center[0], ceilingHeight / 2, footprint.center[1]],
        [footprint.size[0], ceilingHeight, footprint.size[1]],
      );
    }

    for (const object of renderScene.objects) {
      const size: Vector3Tuple = [
        Math.max(object.native_size[0] * object.scale[0], 0.2),
        Math.max(object.native_size[1] * object.scale[1], 0.2),
        Math.max(object.native_size[2] * object.scale[2], 0.2),
      ];
      includeBox(object.position, size);
    }
  } else if (scene) {
    for (const room of scene.normalized.rooms) {
      const translation = room.frame?.translation ?? [0, 0, 0];
      const width = Math.max(room.dimensions?.width ?? 2, 1);
      const length = Math.max(room.dimensions?.length ?? 2, 1);
      const height = Math.max(room.dimensions?.height ?? room.ceiling_height ?? 2.8, 2);
      includeBox(
        [translation[0], translation[1] + height / 2, translation[2]],
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

function SageRoomShell({ room }: { room: RenderableSageRoom }) {
  const footprint = useMemo(() => computeRoomFootprint(room), [room]);
  const wallPanels = useMemo(() => room.walls.map((wall) => wallToPanel(wall)), [room.walls]);
  const floorTextureUrl = toRepoAssetUrl(room.floor_texture_path);
  const wallTextureUrl = toRepoAssetUrl(room.wall_texture_path);

  return (
    <group>
      {floorTextureUrl ? (
        <TexturedFloor center={footprint.center} size={footprint.size} textureUrl={floorTextureUrl} />
      ) : (
        <mesh
          position={[footprint.center[0], 0.002, footprint.center[1]]}
          rotation={[-Math.PI / 2, 0, 0]}
          receiveShadow
        >
          <planeGeometry args={footprint.size} />
          <meshStandardMaterial color="#475569" roughness={0.98} />
        </mesh>
      )}

      {wallPanels.map((panel) =>
        wallTextureUrl ? (
          <TexturedWall key={panel.id} panel={panel} textureUrl={wallTextureUrl} />
        ) : (
          <mesh
            key={panel.id}
            position={panel.position}
            rotation={[0, -panel.rotationY, 0]}
            castShadow
            receiveShadow
          >
            <boxGeometry args={panel.size} />
            <meshStandardMaterial color="#cbd5e1" roughness={0.95} />
          </mesh>
        ),
      )}
    </group>
  );
}

function SageOpenings({ rooms }: { rooms: RenderableSageRoom[] }) {
  const markers = useMemo(() => {
    const wallsById = new Map<string, SageWall>();
    for (const room of rooms) {
      for (const wall of room.walls) {
        wallsById.set(wall.id, wall);
      }
    }

    const nextMarkers: MarkerPanel[] = [];
    for (const room of rooms) {
      for (const door of room.doors) {
        const marker = openingToMarker(door, wallsById.get(door.wall_id), "#22c55e");
        if (marker) {
          nextMarkers.push(marker);
        }
      }
      for (const window of room.windows) {
        const marker = openingToMarker(window, wallsById.get(window.wall_id), "#38bdf8");
        if (marker) {
          nextMarkers.push(marker);
        }
      }
    }
    return nextMarkers;
  }, [rooms]);

  return (
    <>
      {markers.map((marker) => (
        <mesh
          key={marker.id}
          position={marker.position}
          rotation={[0, -marker.rotationY, 0]}
          castShadow
          receiveShadow
        >
          <boxGeometry args={marker.size} />
          <meshStandardMaterial color={marker.color} transparent opacity={0.55} />
        </mesh>
      ))}
    </>
  );
}

function PreviewContent({
  scene,
  renderScene,
}: {
  scene: SceneManifest | null;
  renderScene: RenderableSceneManifest;
}) {
  const dataset = renderScene.dataset;
  const [fitVersion, setFitVersion] = useState(0);
  const [scenesmithShellsReady, setScenesmithShellsReady] = useState(dataset !== "scenesmith");
  const sageAssets = useMemo(() => {
    if (dataset !== "sage") {
      return [];
    }

    return renderScene.objects.map(
      (object): AssetPlacement => ({
        key: object.id,
        assetPath: object.asset_path,
        position: object.position,
        rotationYDeg: object.rotation_y_deg,
        scale: object.scale,
      }),
    );
  }, [dataset, renderScene]);
  const scenesmithShellAssets = useMemo(() => {
    if (dataset !== "scenesmith") {
      return [];
    }

    return renderScene.room_shells.map(
      (shell): AssetPlacement => ({
        key: `${shell.id}::${shell.asset_path}`,
        assetPath: shell.asset_path,
        position: shell.position,
        rotationYDeg: shell.rotation_y_deg,
        scale: shell.scale,
      }),
    );
  }, [dataset, renderScene]);

  const scenesmithObjectAssets = useMemo(() => {
    if (dataset !== "scenesmith") {
      return [];
    }

    return renderScene.objects.map(
      (object): AssetPlacement => ({
        key: object.id,
        assetPath: object.asset_path,
        position: object.position,
        rotationYDeg: object.rotation_y_deg,
        scale: object.scale,
      }),
    );
  }, [dataset, renderScene]);
  const sceneBounds = useMemo(() => computeSceneBounds(scene, renderScene), [renderScene, scene]);
  const scenesmithShellCount = dataset === "scenesmith" ? renderScene.room_shells.length : 0;

  useEffect(() => {
    setScenesmithShellsReady(dataset !== "scenesmith" || scenesmithShellCount === 0);
    setFitVersion((current) => current + 1);
  }, [dataset, renderScene.scene_uid, scenesmithShellCount]);

  return (
    <>
      <ambientLight intensity={1.18} />
      <hemisphereLight args={["#f8fafc", "#334155", 1.05]} />
      <directionalLight position={[12, 16, 10]} intensity={0.85} castShadow={false} />
      <directionalLight position={[-10, 10, -8]} intensity={0.45} castShadow={false} />
      <PreviewEnvironment />
      <Grid
        args={[64, 64]}
        position={[0, -0.002, 0]}
        cellSize={0.75}
        cellThickness={0.45}
        cellColor="#1e293b"
        sectionSize={3}
        sectionThickness={1}
        sectionColor="#334155"
        fadeDistance={88}
        fadeStrength={1}
      />

      <Bounds key={renderScene.scene_uid} clip observe={false} margin={1.18}>
        <SceneBoundsController sceneKey={renderScene.scene_uid} fitVersion={fitVersion} />
        <group>
          <mesh position={sceneBounds.center} visible={false}>
            <boxGeometry args={sceneBounds.size} />
            <meshBasicMaterial transparent opacity={0} depthWrite={false} />
          </mesh>
          {dataset === "sage" ? (
            <>
              {renderScene.rooms.map((room) => (
                <SageRoomShell key={room.id} room={room} />
              ))}
              <SageOpenings rooms={renderScene.rooms} />
              <BatchedAssetModels
                items={sageAssets}
                batchSize={24}
                materialProfile="sage"
              />
            </>
          ) : (
            <>
              <BatchedAssetModels
                items={scenesmithShellAssets}
                batchSize={24}
                materialProfile="scenesmith"
                onComplete={() => {
                  setScenesmithShellsReady(true);
                  setFitVersion((current) => current + 1);
                }}
              />
              {scenesmithShellsReady ? (
                <BatchedAssetModels
                  items={scenesmithObjectAssets}
                  batchSize={20}
                  materialProfile="scenesmith"
                />
              ) : null}
            </>
          )}
        </group>
      </Bounds>
    </>
  );
}

export function ScenePreviewCanvas({ scene, renderScene }: ScenePreviewCanvasProps) {
  if (!scene && !renderScene) {
    return (
      <div className="canvas-empty">
        <p>先选择一个场景</p>
      </div>
    );
  }

  if (!renderScene) {
    return (
      <div className="canvas-empty">
        <p>正在加载可渲染资产...</p>
      </div>
    );
  }

  const dataset = renderScene.dataset;
  const badge =
    dataset === "sage"
      ? `SAGE: ${renderScene.objects.length} textured objects`
      : `SceneSmith: ${renderScene.room_shells.length} room shells + ${renderScene.objects.length} objects`;

  return (
    <div className="canvas-shell">
      <Canvas
        shadows
        camera={{ position: [8, 7, 8], fov: 42, near: 0.1, far: 300 }}
        gl={{ antialias: true }}
      >
        <color attach="background" args={["#020617"]} />
        <PreviewContent scene={scene} renderScene={renderScene} />
        <OrbitControls
          makeDefault
          enableDamping
          enablePan
          dampingFactor={0.08}
          screenSpacePanning
          mouseButtons={{
            LEFT: THREE.MOUSE.ROTATE,
            MIDDLE: THREE.MOUSE.DOLLY,
            RIGHT: THREE.MOUSE.PAN,
          }}
        />
      </Canvas>

      <div className="canvas-caption">
        <div>
          <strong>Three.js Preview</strong>
          <span>左键旋转，右键平移，滚轮缩放</span>
        </div>
        <span className="canvas-badge">{badge}</span>
      </div>
    </div>
  );
}
