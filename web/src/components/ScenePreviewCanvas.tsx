import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import {
  Bounds,
  Environment,
  Edges,
  Grid,
  Html,
  Lightformer,
  OrbitControls,
  useBounds,
  useGLTF,
  useProgress,
  useTexture,
} from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import type { ThreeEvent } from "@react-three/fiber";
import * as THREE from "three";
import { fetchRepoText, toRepoAssetUrl } from "../lib/repoAssets";
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
  wallOpacity: number;
  wallDisplayMode: "solid" | "transparent" | "hidden" | "wireframe";
  showObjectLabels: boolean;
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
  opacity?: number;
  wireframe?: boolean;
  visible?: boolean;
  doubleSided?: boolean;
  transparentDepthWrite?: boolean;
  forceSinglePass?: boolean;
  polygonOffset?: boolean;
  polygonOffsetFactor?: number;
  polygonOffsetUnits?: number;
};

type SceneBounds = {
  center: Vector3Tuple;
  size: Vector3Tuple;
};

type MaterialProfile = "sage" | "scenesmith";

type ObjectLabelPlacement = {
  id: string;
  label: string;
  position: Vector3Tuple;
};

type InspectableObject = ObjectLabelPlacement & {
  size: Vector3Tuple;
};

type BatchProgressSnapshot = {
  readyCount: number;
  visibleCount: number;
  totalCount: number;
  complete: boolean;
};

type ResourceProgressSnapshot = {
  active: boolean;
  item: string;
  loaded: number;
  total: number;
  progress: number;
};

type RenderProgressSnapshot = {
  ready: boolean;
  stage: string;
  detail: string;
  completed: number;
  total: number;
  progress: number;
};

type ProgressStageId = "download" | "parse" | "mount" | "ready";

type SceneSmithShellTransform = {
  position: Vector3Tuple;
  rotationYDeg: number;
};

function createEmptyBatchProgress(totalCount = 0): BatchProgressSnapshot {
  return {
    readyCount: 0,
    visibleCount: 0,
    totalCount,
    complete: totalCount === 0,
  };
}

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

function resolveWallOpacity(
  wallDisplayMode: "solid" | "transparent" | "hidden" | "wireframe",
  wallOpacity: number,
): number {
  return wallDisplayMode === "hidden" ? 0 : wallOpacity;
}

function prepareScene(
  root: THREE.Object3D,
  profile: MaterialProfile,
  {
    opacity = 1,
    wireframe = false,
    doubleSided = false,
    transparentDepthWrite = false,
    forceSinglePass = false,
    polygonOffset = false,
    polygonOffsetFactor = 1,
    polygonOffsetUnits = 1,
  }: {
    opacity?: number;
    wireframe?: boolean;
    doubleSided?: boolean;
    transparentDepthWrite?: boolean;
    forceSinglePass?: boolean;
    polygonOffset?: boolean;
    polygonOffsetFactor?: number;
    polygonOffsetUnits?: number;
  } = {},
): THREE.Object3D {
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
        const isTransparent = opacity < 0.999;
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
        }
        material.side = doubleSided || isTransparent ? THREE.DoubleSide : THREE.FrontSide;
        material.transparent = isTransparent;
        material.opacity = opacity;
        material.depthWrite = isTransparent ? transparentDepthWrite : true;
        material.wireframe = wireframe;
        if (isTransparent) {
          material.forceSinglePass = forceSinglePass;
        }
        if (polygonOffset) {
          material.polygonOffset = true;
          material.polygonOffsetFactor = polygonOffsetFactor;
          material.polygonOffsetUnits = polygonOffsetUnits;
        }
        return material;
      });

      child.material = Array.isArray(child.material) ? nextMaterials : nextMaterials[0];
      child.frustumCulled = false;
    }
  });
  return clone;
}

function sceneSmithToThree(vector: [number, number, number]): Vector3Tuple {
  return [vector[0], vector[2], vector[1]];
}

function resolveRepoRelativePath(basePath: string, relativePath: string): string {
  const baseParts = basePath.split("/");
  baseParts.pop();

  for (const segment of relativePath.split("/")) {
    if (!segment || segment === ".") {
      continue;
    }
    if (segment === "..") {
      if (baseParts.length > 0) {
        baseParts.pop();
      }
      continue;
    }
    baseParts.push(segment);
  }

  return baseParts.join("/");
}

function parseSceneSmithRoomGeometryTransforms(
  xmlText: string,
  sdfPath: string,
): Record<string, SceneSmithShellTransform> {
  const parser = new DOMParser();
  const document = parser.parseFromString(xmlText, "application/xml");
  const transforms: Record<string, SceneSmithShellTransform> = {};

  for (const visual of Array.from(document.querySelectorAll("visual"))) {
    const uri = visual.querySelector("geometry > mesh > uri")?.textContent?.trim();
    if (!uri) {
      continue;
    }

    const pose = visual.querySelector("pose")?.textContent?.trim();
    const [x = 0, y = 0, z = 0, , , yaw = 0] = (pose ?? "")
      .split(/\s+/)
      .map((value) => Number(value))
      .filter((value) => Number.isFinite(value));

    const assetPath = resolveRepoRelativePath(sdfPath, uri);
    transforms[assetPath] = {
      position: sceneSmithToThree([x, y, z]),
      rotationYDeg: THREE.MathUtils.radToDeg(yaw),
    };
  }

  return transforms;
}

function labelText(value: string | null | undefined, fallback: string): string {
  const text = (value || "").trim();
  if (!text) {
    return fallback;
  }
  if (text.length <= 28) {
    return text;
  }
  return `${text.slice(0, 25)}...`;
}

function compactSceneSmithName(
  value: string | null | undefined,
  roomId?: string | null,
): string | null {
  const text = (value || "").trim();
  if (!text) {
    return null;
  }

  let next = text;
  if (roomId) {
    const prefix = `${roomId}_`;
    if (next.startsWith(prefix)) {
      next = next.slice(prefix.length);
    }
  }

  next = next.replace(/_\d+$/, "");
  return next || text || null;
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
  opacity,
  wireframe,
}: {
  panel: WallPanel;
  textureUrl: string;
  opacity: number;
  wireframe: boolean;
}) {
  const texture = useTiledTexture(textureUrl, panel.size[0] / 1.2, panel.size[1] / 1.2);
  return (
    <mesh position={panel.position} rotation={[0, -panel.rotationY, 0]} castShadow receiveShadow>
      <boxGeometry args={panel.size} />
      <meshStandardMaterial
        map={texture}
        roughness={0.96}
        metalness={0.02}
        transparent={opacity < 0.999}
        opacity={opacity}
        depthWrite={opacity >= 0.55}
        wireframe={wireframe}
      />
    </mesh>
  );
}

function AssetModel({
  assetPath,
  position,
  rotationYDeg,
  scale,
  onReady,
  onBounds,
  materialProfile,
  opacity,
  wireframe,
  visible = true,
  doubleSided,
  transparentDepthWrite,
  forceSinglePass,
  polygonOffset,
  polygonOffsetFactor,
  polygonOffsetUnits,
}: {
  assetPath: string;
  position: Vector3Tuple;
  rotationYDeg: number;
  scale: Vector3Tuple;
  onReady?: () => void;
  onBounds?: (bounds: { center: Vector3Tuple; size: Vector3Tuple }) => void;
  materialProfile: MaterialProfile;
  opacity?: number;
  wireframe?: boolean;
  visible?: boolean;
  doubleSided?: boolean;
  transparentDepthWrite?: boolean;
  forceSinglePass?: boolean;
  polygonOffset?: boolean;
  polygonOffsetFactor?: number;
  polygonOffsetUnits?: number;
}) {
  const url = toRepoAssetUrl(assetPath);
  if (!url) {
    return null;
  }

  const gltf = useGLTF(url);
  const groupRef = useRef<THREE.Group | null>(null);
  const object = useMemo(
    () =>
      prepareScene(gltf.scene, materialProfile, {
        opacity: opacity ?? 1,
        wireframe: wireframe ?? false,
        doubleSided: doubleSided ?? false,
        transparentDepthWrite: transparentDepthWrite ?? false,
        forceSinglePass: forceSinglePass ?? false,
        polygonOffset: polygonOffset ?? false,
        polygonOffsetFactor: polygonOffsetFactor ?? 1,
        polygonOffsetUnits: polygonOffsetUnits ?? 1,
      }),
    [
      doubleSided,
      forceSinglePass,
      gltf.scene,
      materialProfile,
      opacity,
      polygonOffset,
      polygonOffsetFactor,
      polygonOffsetUnits,
      transparentDepthWrite,
      wireframe,
    ],
  );

  useEffect(() => {
    onReady?.();
  }, [onReady]);

  useEffect(() => {
    const group = groupRef.current;
    if (!group || !onBounds) {
      return;
    }

    group.updateWorldMatrix(true, true);
    const box = new THREE.Box3().setFromObject(group);
    if (box.isEmpty()) {
      return;
    }

    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    box.getCenter(center);
    box.getSize(size);

    onBounds({
      center: [center.x, center.y, center.z],
      size: [
        Math.max(size.x, 0.18),
        Math.max(size.y, 0.18),
        Math.max(size.z, 0.18),
      ],
    });
  }, [object, onBounds, position, rotationYDeg, scale]);

  return (
    <group
      ref={groupRef}
      visible={visible}
      position={position}
      rotation={[0, THREE.MathUtils.degToRad(rotationYDeg), 0]}
      scale={scale}
    >
      <primitive object={object} />
    </group>
  );
}

function BatchedAssetModels({
  items,
  batchSize,
  onComplete,
  onItemBounds,
  materialProfile,
  onProgress,
}: {
  items: AssetPlacement[];
  batchSize: number;
  onComplete?: () => void;
  onItemBounds?: (key: string, bounds: { center: Vector3Tuple; size: Vector3Tuple }) => void;
  materialProfile: MaterialProfile;
  onProgress?: (snapshot: BatchProgressSnapshot) => void;
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

  useEffect(() => {
    onProgress?.({
      readyCount,
      visibleCount,
      totalCount: items.length,
      complete: items.length === 0 || readyCount >= items.length,
    });
  }, [items.length, onProgress, readyCount, visibleCount]);

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
            onBounds={onItemBounds ? (bounds) => onItemBounds(item.key, bounds) : undefined}
            materialProfile={materialProfile}
            opacity={item.opacity}
            wireframe={item.wireframe}
            visible={item.visible}
            doubleSided={item.doubleSided}
            transparentDepthWrite={item.transparentDepthWrite}
            forceSinglePass={item.forceSinglePass}
            polygonOffset={item.polygonOffset}
            polygonOffsetFactor={item.polygonOffsetFactor}
            polygonOffsetUnits={item.polygonOffsetUnits}
          />
        </Suspense>
      ))}
    </>
  );
}

function LoadingProgressReporter({
  sceneKey,
  onChange,
}: {
  sceneKey: string;
  onChange: (snapshot: ResourceProgressSnapshot) => void;
}) {
  const { active, item, loaded, total, progress } = useProgress();

  useEffect(() => {
    const timeoutId = window.setTimeout(() => {
      onChange({
        active,
        item,
        loaded,
        total,
        progress,
      });
    }, 0);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [active, item, loaded, onChange, progress, sceneKey, total]);

  return null;
}

function formatProgressItemLabel(item: string): string {
  const normalized = item.split("?")[0]?.split("/").filter(Boolean).pop() ?? item;
  if (normalized.length <= 42) {
    return normalized;
  }
  return `${normalized.slice(0, 39)}...`;
}

function inferParsingState(progress: ResourceProgressSnapshot): boolean {
  return !progress.active && progress.total > 0 && progress.loaded > 0;
}

function ObjectLabels({ items }: { items: ObjectLabelPlacement[] }) {
  return (
    <>
      {items.map((item) => (
        <group key={item.id} position={item.position}>
          <Html transform sprite distanceFactor={8} center>
            <div className="object-label" title={item.label}>
              {item.label}
            </div>
          </Html>
        </group>
      ))}
    </>
  );
}

function SelectionOverlays({
  items,
  activeId,
  hoveredId,
}: {
  items: InspectableObject[];
  activeId: string | null;
  hoveredId: string | null;
}) {
  return (
    <>
      {items
        .filter((item) => item.id === activeId || (item.id === hoveredId && item.id !== activeId))
        .map((item) => {
          const isActive = item.id === activeId;
          return (
            <mesh key={item.id} position={item.position}>
              <boxGeometry
                args={[
                  Math.max(item.size[0] + 0.06, 0.18),
                  Math.max(item.size[1] + 0.06, 0.18),
                  Math.max(item.size[2] + 0.06, 0.18),
                ]}
              />
              <meshBasicMaterial transparent opacity={0} depthWrite={false} />
              <Edges
                color={isActive ? "#38bdf8" : "#f8fafc"}
                lineWidth={1}
                scale={1}
                threshold={15}
              />
            </mesh>
          );
        })}
    </>
  );
}

function ObjectHitTargets({
  items,
  activeId,
  onHoverChange,
  onSelect,
}: {
  items: InspectableObject[];
  activeId: string | null;
  onHoverChange: (id: string | null) => void;
  onSelect: (id: string, additive: boolean) => void;
}) {
  return (
    <>
      {items.map((item) => (
        <mesh
          key={item.id}
          position={item.position}
          onPointerOver={(event: ThreeEvent<PointerEvent>) => {
            event.stopPropagation();
            onHoverChange(item.id);
            document.body.style.cursor = "pointer";
          }}
          onPointerOut={(event: ThreeEvent<PointerEvent>) => {
            event.stopPropagation();
            onHoverChange(null);
            document.body.style.cursor = "default";
          }}
          onClick={(event: ThreeEvent<MouseEvent>) => {
            event.stopPropagation();
            onSelect(item.id, Boolean(event.nativeEvent.ctrlKey || event.nativeEvent.metaKey));
          }}
        >
          <boxGeometry
            args={[
              Math.max(item.size[0], 0.24),
              Math.max(item.size[1], 0.24),
              Math.max(item.size[2], 0.24),
            ]}
          />
          <meshBasicMaterial
            transparent
            opacity={activeId === item.id ? 0.04 : 0}
            color="#38bdf8"
            depthWrite={false}
          />
        </mesh>
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
  measuredShellBounds: Record<string, { center: Vector3Tuple; size: Vector3Tuple }> = {},
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
  } else {
    const shellBounds = Object.values(measuredShellBounds);
    for (const bounds of shellBounds) {
      includeBox(bounds.center, bounds.size);
    }

    if (scene) {
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
  }

  if (!Number.isFinite(min[0]) || !Number.isFinite(max[0])) {
    return createEmptyBounds();
  }

  return finalizeBounds(min, max);
}

function SageRoomShell({
  room,
  wallOpacity,
  wallDisplayMode,
}: {
  room: RenderableSageRoom;
  wallOpacity: number;
  wallDisplayMode: "solid" | "transparent" | "hidden" | "wireframe";
}) {
  const footprint = useMemo(() => computeRoomFootprint(room), [room]);
  const wallPanels = useMemo(() => room.walls.map((wall) => wallToPanel(wall)), [room.walls]);
  const floorTextureUrl = toRepoAssetUrl(room.floor_texture_path);
  const wallTextureUrl = toRepoAssetUrl(room.wall_texture_path);
  const wallOpacityValue = resolveWallOpacity(wallDisplayMode, wallOpacity);
  const wallWireframe = wallDisplayMode === "wireframe";
  const wallVisible = wallDisplayMode !== "hidden";

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

      {wallVisible
        ? wallPanels.map((panel) =>
        wallTextureUrl ? (
          <TexturedWall
            key={panel.id}
            panel={panel}
            textureUrl={wallTextureUrl}
            opacity={wallOpacityValue}
            wireframe={wallWireframe}
          />
        ) : (
          <mesh
            key={panel.id}
            position={panel.position}
            rotation={[0, -panel.rotationY, 0]}
            castShadow
            receiveShadow
          >
            <boxGeometry args={panel.size} />
            <meshStandardMaterial
              color="#cbd5e1"
              roughness={0.95}
              transparent={wallOpacityValue < 0.999}
              opacity={wallOpacityValue}
              depthWrite={wallOpacityValue >= 0.55}
              wireframe={wallWireframe}
            />
          </mesh>
        ),
      )
        : null}
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
  wallOpacity,
  wallDisplayMode,
  showObjectLabels,
  onRenderProgressChange,
}: {
  scene: SceneManifest | null;
  renderScene: RenderableSceneManifest;
  wallOpacity: number;
  wallDisplayMode: "solid" | "transparent" | "hidden" | "wireframe";
  showObjectLabels: boolean;
  onRenderProgressChange: (snapshot: RenderProgressSnapshot) => void;
}) {
  const dataset = renderScene.dataset;
  const [fitVersion, setFitVersion] = useState(0);
  const [scenesmithShellsReady, setScenesmithShellsReady] = useState(dataset !== "scenesmith");
  const [shellTransformsLoaded, setShellTransformsLoaded] = useState(dataset !== "scenesmith");
  const [shellTransformMap, setShellTransformMap] = useState<Record<string, SceneSmithShellTransform>>({});
  const [hoveredObjectId, setHoveredObjectId] = useState<string | null>(null);
  const [selectedObjectId, setSelectedObjectId] = useState<string | null>(null);
  const [measuredShellBounds, setMeasuredShellBounds] = useState<
    Record<string, { center: Vector3Tuple; size: Vector3Tuple }>
  >({});
  const [measuredObjectBounds, setMeasuredObjectBounds] = useState<
    Record<string, { center: Vector3Tuple; size: Vector3Tuple }>
  >({});
  const [sageBatchProgress, setSageBatchProgress] = useState<BatchProgressSnapshot>(
    createEmptyBatchProgress(),
  );
  const [shellBatchProgress, setShellBatchProgress] = useState<BatchProgressSnapshot>(
    createEmptyBatchProgress(),
  );
  const [objectBatchProgress, setObjectBatchProgress] = useState<BatchProgressSnapshot>(
    createEmptyBatchProgress(),
  );
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
  }, [dataset, renderScene, shellTransformMap, wallDisplayMode, wallOpacity]);

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
  const sceneBounds = useMemo(
    () => computeSceneBounds(scene, renderScene, measuredShellBounds),
    [measuredShellBounds, renderScene, scene],
  );
  const scenesmithShellCount = dataset === "scenesmith" ? renderScene.room_shells.length : 0;
  const inspectableObjects = useMemo((): InspectableObject[] => {
    if (renderScene.dataset === "sage") {
      return renderScene.objects.map((object) => {
        const measured = measuredObjectBounds[object.id];
        const size: Vector3Tuple = [
          Math.max(object.native_size[0] * object.scale[0], 0.18),
          Math.max(object.native_size[1] * object.scale[1], 0.18),
          Math.max(object.native_size[2] * object.scale[2], 0.18),
        ];
        return {
          id: object.id,
          label: labelText(object.type || object.description, object.id),
          position: measured?.center ?? object.position,
          size: measured?.size ?? size,
        };
      });
    }

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
        const centerX = (bboxMin[0] + bboxMax[0]) / 2;
        const centerY = (bboxMin[2] + bboxMax[2]) / 2;
        const centerZ = (bboxMin[1] + bboxMax[1]) / 2;
        return {
          id: object.id,
          label: labelText(preferredLabel, object.id),
          position: measured?.center ?? sceneSmithToThree([centerX, centerZ, centerY]),
          size: measured?.size ?? ([
            Math.max(Math.abs(bboxMax[0] - bboxMin[0]), 0.18),
            Math.max(Math.abs(bboxMax[2] - bboxMin[2]), 0.18),
            Math.max(Math.abs(bboxMax[1] - bboxMin[1]), 0.18),
          ] as Vector3Tuple),
        };
      }

      return {
        id: object.id,
        label: labelText(preferredLabel, object.id),
        position: measured?.center ?? object.position,
        size: measured?.size ?? ([
          Math.max(Math.abs(object.scale[0]), 0.45),
          Math.max(Math.abs(object.scale[1]), 0.45),
          Math.max(Math.abs(object.scale[2]), 0.45),
        ] as Vector3Tuple),
      };
    });
  }, [measuredObjectBounds, renderScene, scene]);

  const objectLabels = useMemo(() => {
    const alwaysVisibleIds = new Set<string>();
    if (showObjectLabels) {
      for (const item of inspectableObjects) {
        alwaysVisibleIds.add(item.id);
      }
    }
    if (selectedObjectId) {
      alwaysVisibleIds.add(selectedObjectId);
    } else if (hoveredObjectId) {
      alwaysVisibleIds.add(hoveredObjectId);
    }

    return inspectableObjects
      .filter((item) => alwaysVisibleIds.has(item.id))
      .map((item) => ({
        id: item.id,
        label: item.label,
        position: [
          item.position[0],
          item.position[1] + Math.max(item.size[1] / 2 + 0.14, 0.2),
          item.position[2],
        ] as Vector3Tuple,
      }));
  }, [hoveredObjectId, inspectableObjects, selectedObjectId, showObjectLabels]);

  useEffect(() => {
    if (dataset !== "scenesmith" || !scene) {
      setShellTransformMap({});
      setShellTransformsLoaded(true);
      return;
    }

    const roomGeometryPaths = (scene.normalized.rooms ?? [])
      .map((room) => room.room_geometry_sdf)
      .filter((value): value is string => Boolean(value));

    if (roomGeometryPaths.length === 0) {
      setShellTransformMap({});
      setShellTransformsLoaded(true);
      return;
    }

    let cancelled = false;
    setShellTransformsLoaded(false);
    setShellTransformMap({});

    async function loadShellTransforms() {
      try {
        const entries = await Promise.all(
          roomGeometryPaths.map(async (roomGeometryPath) => {
            const xmlText = await fetchRepoText(roomGeometryPath);
            return parseSceneSmithRoomGeometryTransforms(xmlText, roomGeometryPath);
          }),
        );

        if (cancelled) {
          return;
        }

        setShellTransformMap(Object.assign({}, ...entries));
      } catch {
        if (cancelled) {
          return;
        }
        setShellTransformMap({});
      } finally {
        if (!cancelled) {
          setShellTransformsLoaded(true);
        }
      }
    }

    void loadShellTransforms();

    return () => {
      cancelled = true;
    };
  }, [dataset, scene, renderScene.scene_uid]);

  useEffect(() => {
    if (dataset === "sage") {
      setSageBatchProgress(createEmptyBatchProgress(sageAssets.length));
      setShellBatchProgress(createEmptyBatchProgress());
      setObjectBatchProgress(createEmptyBatchProgress());
      return;
    }

    setSageBatchProgress(createEmptyBatchProgress());
    setShellBatchProgress(createEmptyBatchProgress(scenesmithShellAssets.length));
    setObjectBatchProgress(createEmptyBatchProgress(scenesmithObjectAssets.length));
  }, [dataset, renderScene.scene_uid, sageAssets.length, scenesmithObjectAssets.length, scenesmithShellAssets.length]);

  useEffect(() => {
    setScenesmithShellsReady(dataset !== "scenesmith" || scenesmithShellCount === 0);
    setFitVersion((current) => current + 1);
  }, [dataset, renderScene.scene_uid, scenesmithShellCount, shellTransformsLoaded]);

  useEffect(() => {
    setHoveredObjectId(null);
    setSelectedObjectId(null);
    setShellTransformsLoaded(dataset !== "scenesmith");
    setShellTransformMap({});
    setMeasuredShellBounds({});
    setMeasuredObjectBounds({});
    document.body.style.cursor = "default";
  }, [dataset, renderScene.scene_uid]);

  useEffect(() => {
    if (dataset === "sage") {
      const total = sageAssets.length;
      const completed = Math.min(sageBatchProgress.readyCount, total);
      const ready = total === 0 || sageBatchProgress.complete;
      const progress = total === 0 ? 100 : Math.round((completed / total) * 100);
      onRenderProgressChange({
        ready,
        stage: ready ? "Scene ready" : "Preparing objects",
        detail:
          total === 0
            ? "No object assets need staged rendering"
            : `Mounted ${completed}/${total} object assets`,
        completed,
        total,
        progress,
      });
      return;
    }

    const shellCompleted = Math.min(shellBatchProgress.readyCount, scenesmithShellAssets.length);
    const objectCompleted = Math.min(objectBatchProgress.readyCount, scenesmithObjectAssets.length);
    const total = scenesmithShellAssets.length + scenesmithObjectAssets.length;
    const completed = shellCompleted + (scenesmithShellsReady ? objectCompleted : 0);
    const ready =
      (scenesmithShellAssets.length === 0 || shellBatchProgress.complete) &&
      (scenesmithObjectAssets.length === 0 || objectBatchProgress.complete);
    const progress = total === 0 ? 100 : Math.round((completed / total) * 100);

    let stage = "Scene ready";
    let detail = "All room shells and objects are mounted";
    if (!shellTransformsLoaded) {
      stage = "Preparing room shells";
      detail = "Loading SceneSmith room shell transforms";
    } else if (!scenesmithShellsReady) {
      stage = "Preparing room shells";
      detail =
        scenesmithShellAssets.length === 0
          ? "No room shell assets to stage"
          : `Mounted ${shellCompleted}/${scenesmithShellAssets.length} room shell assets`;
    } else if (!ready) {
      stage = "Preparing objects";
      detail =
        scenesmithObjectAssets.length === 0
          ? "No object assets to stage"
          : `Mounted ${objectCompleted}/${scenesmithObjectAssets.length} object assets`;
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
    dataset,
    objectBatchProgress.complete,
    objectBatchProgress.readyCount,
    onRenderProgressChange,
    sageAssets.length,
    sageBatchProgress.complete,
    sageBatchProgress.readyCount,
    scenesmithObjectAssets.length,
    scenesmithShellAssets.length,
    scenesmithShellsReady,
    shellTransformsLoaded,
    shellBatchProgress.complete,
    shellBatchProgress.readyCount,
  ]);

  function handleObjectSelect(id: string, additive: boolean) {
    setSelectedObjectId((current) => {
      if (additive && current === id) {
        return null;
      }
      return id;
    });
  }

  function handleObjectBounds(
    id: string,
    bounds: { center: Vector3Tuple; size: Vector3Tuple },
  ) {
    setMeasuredObjectBounds((current) => {
      const prev = current[id];
      if (
        prev &&
        prev.center.every((value, index) => Math.abs(value - bounds.center[index]) < 0.0001) &&
        prev.size.every((value, index) => Math.abs(value - bounds.size[index]) < 0.0001)
      ) {
        return current;
      }
      return {
        ...current,
        [id]: bounds,
      };
    });
  }

  function handleShellBounds(
    id: string,
    bounds: { center: Vector3Tuple; size: Vector3Tuple },
  ) {
    setMeasuredShellBounds((current) => {
      const prev = current[id];
      if (
        prev &&
        prev.center.every((value, index) => Math.abs(value - bounds.center[index]) < 0.0001) &&
        prev.size.every((value, index) => Math.abs(value - bounds.size[index]) < 0.0001)
      ) {
        return current;
      }
      return {
        ...current,
        [id]: bounds,
      };
    });
  }

  return (
    <>
      <ambientLight intensity={1.18} />
      <hemisphereLight args={["#f8fafc", "#334155", 1.05]} />
      <directionalLight position={[12, 16, 10]} intensity={0.85} castShadow={false} />
      <directionalLight position={[-10, 10, -8]} intensity={0.45} castShadow={false} />
      <PreviewEnvironment />
      <Grid
        args={[64, 64]}
        position={[0, -0.08, 0]}
        renderOrder={-20}
        material-depthTest={true}
        material-depthWrite={false}
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
          {dataset === "sage" ? (
            <>
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
                items={sageAssets}
                batchSize={24}
                onItemBounds={handleObjectBounds}
                materialProfile="sage"
                onProgress={setSageBatchProgress}
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
          ) : (
            <>
              {shellTransformsLoaded ? (
                <BatchedAssetModels
                  items={scenesmithShellAssets}
                  batchSize={24}
                  materialProfile="scenesmith"
                  onItemBounds={handleShellBounds}
                  onProgress={setShellBatchProgress}
                  onComplete={() => {
                    setScenesmithShellsReady(true);
                    setFitVersion((current) => current + 1);
                  }}
                />
              ) : null}
              {scenesmithShellsReady ? (
                <>
                  <BatchedAssetModels
                    items={scenesmithObjectAssets}
                    batchSize={20}
                    onItemBounds={handleObjectBounds}
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
            </>
          )}
        </group>
      </Bounds>
    </>
  );
}

export function ScenePreviewCanvas({
  scene,
  renderScene,
  wallOpacity,
  wallDisplayMode,
  showObjectLabels,
}: ScenePreviewCanvasProps) {
  const [resourceProgress, setResourceProgress] = useState<ResourceProgressSnapshot>({
    active: false,
    item: "",
    loaded: 0,
    total: 0,
    progress: 100,
  });
  const [renderProgress, setRenderProgress] = useState<RenderProgressSnapshot>({
    ready: false,
    stage: "Waiting for scene",
    detail: "Choose a scene to start rendering",
    completed: 0,
    total: 0,
    progress: 0,
  });

  useEffect(() => {
    if (!renderScene) {
      setResourceProgress({
        active: false,
        item: "",
        loaded: 0,
        total: 0,
        progress: 0,
      });
      setRenderProgress({
        ready: false,
        stage: "Loading scene data",
        detail: "Waiting for renderable scene manifest",
        completed: 0,
        total: 0,
        progress: 0,
      });
      return;
    }

    setResourceProgress({
      active: false,
      item: "",
      loaded: 0,
      total: 0,
      progress: 100,
    });
    setRenderProgress({
      ready: false,
      stage: "Preparing scene",
      detail: "Initializing asset batches",
      completed: 0,
      total: 0,
      progress: 0,
    });
  }, [renderScene?.scene_uid]);

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
  const resourceActive = resourceProgress.active && resourceProgress.total > 0;
  const resourceValue =
    resourceProgress.total > 0
      ? Math.max(0, Math.min(100, Math.round(resourceProgress.progress)))
      : resourceActive
        ? Math.max(0, Math.min(100, Math.round(resourceProgress.progress)))
        : 100;
  const renderValue = Math.max(0, Math.min(100, Math.round(renderProgress.progress)));
  const overallProgress = Math.round(resourceValue * 0.45 + renderValue * 0.55);
  const previewReady = !resourceActive && renderProgress.ready;
  const parsingActive = inferParsingState(resourceProgress) && !renderProgress.ready && renderValue === 0;
  const currentStage: ProgressStageId = previewReady
    ? "ready"
    : resourceActive
      ? "download"
      : parsingActive
        ? "parse"
        : "mount";
  const statusTitle =
    currentStage === "download"
      ? "Downloading assets"
      : currentStage === "parse"
        ? "Parsing models"
        : currentStage === "mount"
          ? "Mounting scene"
          : "Scene ready";
  const statusDetail =
    currentStage === "download"
    ? `${
        resourceProgress.loaded && resourceProgress.total
          ? `${resourceProgress.loaded}/${resourceProgress.total} files`
          : "Fetching textures and models"
      }${resourceProgress.item ? ` · ${formatProgressItemLabel(resourceProgress.item)}` : ""}`
    : currentStage === "parse"
      ? "Assets downloaded, preparing geometry and materials"
      : renderProgress.detail;

  return (
    <div className="canvas-shell">
      <div className={`canvas-progress ${previewReady ? "is-ready" : "is-busy"}`}>
        <div className="canvas-progress-header">
          <strong>{statusTitle}</strong>
          <span>{previewReady ? "Ready" : `${overallProgress}%`}</span>
        </div>
        <div
          className="canvas-progress-bar"
          role="progressbar"
          aria-label="Scene preview loading progress"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={overallProgress}
        >
          <div className="canvas-progress-fill" style={{ width: `${overallProgress}%` }} />
        </div>
        <div className="canvas-progress-meta">
          <span>{statusDetail}</span>
          <span>{`Downloading assets -> Parsing models -> Mounting scene`}</span>
        </div>
        <div className="canvas-progress-stages" aria-hidden="true">
          <span
            className={`canvas-progress-stage ${
              currentStage === "download" ? "is-current" : resourceActive ? "is-current" : "is-done"
            }`}
          >
            Downloading assets
          </span>
          <span
            className={`canvas-progress-stage ${
              currentStage === "parse"
                ? "is-current"
                : currentStage === "mount" || currentStage === "ready"
                  ? "is-done"
                  : ""
            }`}
          >
            Parsing models
          </span>
          <span
            className={`canvas-progress-stage ${
              currentStage === "mount"
                ? "is-current"
                : currentStage === "ready"
                  ? "is-done"
                  : ""
            }`}
          >
            Mounting scene
          </span>
        </div>
      </div>
      <Canvas
        shadows
        camera={{ position: [8, 7, 8], fov: 42, near: 0.1, far: 300 }}
        gl={{ antialias: true }}
      >
        <color attach="background" args={["#020617"]} />
        <LoadingProgressReporter sceneKey={renderScene.scene_uid} onChange={setResourceProgress} />
        <PreviewContent
          scene={scene}
          renderScene={renderScene}
          wallOpacity={wallOpacity}
          wallDisplayMode={wallDisplayMode}
          showObjectLabels={showObjectLabels}
          onRenderProgressChange={setRenderProgress}
        />
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
          <span>悬停看单个标签，点击固定，Ctrl/Cmd+点击取消固定</span>
        </div>
        <span className="canvas-badge">{badge}</span>
      </div>
    </div>
  );
}
