import {
  Component,
  Suspense,
  startTransition,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ErrorInfo,
  type ReactNode,
} from "react";
import {
  Bounds,
  Edges,
  Environment,
  Html,
  Lightformer,
  useBounds,
  useGLTF,
  useTexture,
} from "@react-three/drei";
import type { ThreeEvent } from "@react-three/fiber";
import * as THREE from "three";
import { fetchRepoText, toRepoAssetUrl } from "../../lib/repoAssets";
import type {
  Renderable3dFrontRoomShell,
  RenderableSceneSmithRoomShell,
  RenderableSageDoor,
  RenderableSageRoom,
  SageDoor,
  SageWall,
  SageWindow,
} from "../../types";

export type WallDisplayMode = "solid" | "transparent" | "hidden" | "wireframe";
export type Vector3Tuple = [number, number, number];
export type QuaternionTuple = [number, number, number, number];

export type WallPanel = {
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

export type RoomFootprint = {
  center: [number, number];
  size: [number, number];
};

export type AssetPlacement = {
  key: string;
  assetPath: string;
  position: Vector3Tuple;
  rotationYDeg: number;
  scale: Vector3Tuple;
  quaternion?: QuaternionTuple | null;
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

export type SceneBounds = {
  center: Vector3Tuple;
  size: Vector3Tuple;
};

export type MaterialProfile = "sage" | "scenesmith" | "sceneweaver" | "3dfront" | "hsm";

export type ObjectLabelPlacement = {
  id: string;
  label: string;
  position: Vector3Tuple;
};

export type InspectableObject = ObjectLabelPlacement & {
  size: Vector3Tuple;
};

export type BatchProgressSnapshot = {
  readyCount: number;
  visibleCount: number;
  totalCount: number;
  complete: boolean;
};

export type ResourceProgressSnapshot = {
  active: boolean;
  item: string;
  loaded: number;
  total: number;
  progress: number;
};

export type RenderProgressSnapshot = {
  ready: boolean;
  stage: string;
  detail: string;
  completed: number;
  total: number;
  progress: number;
};

export type ProgressStageId = "download" | "parse" | "mount" | "ready";

export type SceneSmithShellTransform = {
  position: Vector3Tuple;
  rotationYDeg: number;
};

export function createEmptyBatchProgress(totalCount = 0): BatchProgressSnapshot {
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

export function computeRoomFootprint(room: RenderableSageRoom): RoomFootprint {
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

export function resolveWallOpacity(wallDisplayMode: WallDisplayMode, wallOpacity: number): number {
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
          if (material.emissiveMap) {
            material.emissiveMap.colorSpace = THREE.SRGBColorSpace;
            material.emissiveMap.needsUpdate = true;
          }
          if (profile === "sage") {
            material.metalness = 0;
            material.roughness = 1;
          } else if (profile === "sceneweaver") {
            material.metalness = material.metalness ?? 0;
            material.roughness = material.roughness ?? 1;
          } else {
            material.metalness = Math.min(material.metalness ?? 0, 0.08);
            material.roughness = Math.max(material.roughness ?? 0.92, 0.78);
          }
          material.envMapIntensity = profile === "sceneweaver" ? 0.72 : 0.9;
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

  if (profile === "hsm") {
    const bounds = new THREE.Box3().setFromObject(clone);
    if (!bounds.isEmpty()) {
      const centerX = (bounds.min.x + bounds.max.x) / 2;
      const centerZ = (bounds.min.z + bounds.max.z) / 2;
      clone.position.set(-centerX, -bounds.min.y, -centerZ);
    }
  }

  return clone;
}

export function sceneSmithToThree(vector: [number, number, number]): Vector3Tuple {
  return [vector[0], vector[2], -vector[1]];
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

export async function loadSceneSmithShellTransforms(
  roomGeometryPaths: string[],
): Promise<Record<string, SceneSmithShellTransform>> {
  const entries = await Promise.all(
    roomGeometryPaths.map(async (roomGeometryPath) => {
      const xmlText = await fetchRepoText(roomGeometryPath);
      return parseSceneSmithRoomGeometryTransforms(xmlText, roomGeometryPath);
    }),
  );

  return Object.assign({}, ...entries);
}

export function labelText(value: string | null | undefined, fallback: string): string {
  const text = (value || "").trim();
  if (!text) {
    return fallback;
  }
  if (text.length <= 28) {
    return text;
  }
  return `${text.slice(0, 25)}...`;
}

export function compactSceneSmithName(
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

export function PreviewEnvironment() {
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
      <Lightformer form="ring" intensity={1.2} color="#f8fafc" position={[0, 12, 0]} scale={10} />
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
    <mesh position={[center[0], 0.005, center[1]]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
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
  quaternion,
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
  quaternion?: QuaternionTuple | null;
  onReady?: () => void;
  onBounds?: (bounds: SceneBounds) => void;
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

  return (
    <AssetModelErrorBoundary
      resetKey={url}
      onError={onReady}
      fallback={
        <MissingAssetPlaceholder
          position={position}
          rotationYDeg={rotationYDeg}
          scale={scale}
          quaternion={quaternion}
          visible={visible}
          onBounds={onBounds}
        />
      }
    >
      <AssetModelContent
        url={url}
        position={position}
        rotationYDeg={rotationYDeg}
        scale={scale}
        quaternion={quaternion}
        onReady={onReady}
        onBounds={onBounds}
        materialProfile={materialProfile}
        opacity={opacity}
        wireframe={wireframe}
        visible={visible}
        doubleSided={doubleSided}
        transparentDepthWrite={transparentDepthWrite}
        forceSinglePass={forceSinglePass}
        polygonOffset={polygonOffset}
        polygonOffsetFactor={polygonOffsetFactor}
        polygonOffsetUnits={polygonOffsetUnits}
      />
    </AssetModelErrorBoundary>
  );
}

class AssetModelErrorBoundary extends Component<
  {
    resetKey: string;
    onError?: () => void;
    fallback: ReactNode;
    children: ReactNode;
  },
  { hasError: boolean }
> {
  state = { hasError: false };

  private reported = false;

  static getDerivedStateFromError(): { hasError: boolean } {
    return { hasError: true };
  }

  componentDidCatch(_error: Error, _info: ErrorInfo) {
    if (this.reported) {
      return;
    }
    this.reported = true;
    this.props.onError?.();
  }

  componentDidUpdate(prevProps: Readonly<{ resetKey: string }>) {
    if (prevProps.resetKey === this.props.resetKey || !this.state.hasError) {
      return;
    }
    this.reported = false;
    this.setState({ hasError: false });
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback;
    }
    return this.props.children;
  }
}

function MissingAssetPlaceholder({
  position,
  rotationYDeg,
  scale,
  quaternion,
  visible = true,
  onBounds,
}: {
  position: Vector3Tuple;
  rotationYDeg: number;
  scale: Vector3Tuple;
  quaternion?: QuaternionTuple | null;
  visible?: boolean;
  onBounds?: (bounds: SceneBounds) => void;
}) {
  const placeholderSize = useMemo(
    () =>
      [
        Math.max(Math.abs(scale[0]), 0.24),
        Math.max(Math.abs(scale[1]), 0.24),
        Math.max(Math.abs(scale[2]), 0.24),
      ] as Vector3Tuple,
    [scale],
  );
  const resolvedQuaternion = useMemo(
    () =>
      quaternion
        ? new THREE.Quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
        : undefined,
    [quaternion],
  );

  useEffect(() => {
    onBounds?.({
      center: position,
      size: placeholderSize,
    });
  }, [onBounds, placeholderSize, position]);

  return (
    <group
      visible={visible}
      position={position}
      rotation={quaternion ? undefined : [0, THREE.MathUtils.degToRad(rotationYDeg), 0]}
      quaternion={resolvedQuaternion}
    >
      <mesh>
        <boxGeometry args={placeholderSize} />
        <meshStandardMaterial
          color="#f97316"
          roughness={1}
          metalness={0}
          transparent
          opacity={0.16}
          wireframe
        />
      </mesh>
    </group>
  );
}

function AssetModelContent({
  url,
  position,
  rotationYDeg,
  scale,
  quaternion,
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
  url: string;
  position: Vector3Tuple;
  rotationYDeg: number;
  scale: Vector3Tuple;
  quaternion?: QuaternionTuple | null;
  onReady?: () => void;
  onBounds?: (bounds: SceneBounds) => void;
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
  const resolvedQuaternion = useMemo(
    () =>
      quaternion
        ? new THREE.Quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
        : undefined,
    [quaternion],
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
      size: [Math.max(size.x, 0.18), Math.max(size.y, 0.18), Math.max(size.z, 0.18)],
    });
  }, [object, onBounds, position, quaternion, rotationYDeg, scale]);

  return (
    <group
      ref={groupRef}
      visible={visible}
      position={position}
      rotation={quaternion ? undefined : [0, THREE.MathUtils.degToRad(rotationYDeg), 0]}
      quaternion={resolvedQuaternion}
      scale={scale}
    >
      <primitive object={object} />
    </group>
  );
}

export function BatchedAssetModels({
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
  onItemBounds?: (key: string, bounds: SceneBounds) => void;
  materialProfile: MaterialProfile;
  onProgress?: (snapshot: BatchProgressSnapshot) => void;
}) {
  const [visibleCount, setVisibleCount] = useState(Math.min(batchSize, items.length));
  const [readyCount, setReadyCount] = useState(0);
  const loadedKeysRef = useRef<Set<string>>(new Set());
  const completedRef = useRef(false);

  useEffect(() => {
    setVisibleCount(Math.min(batchSize, items.length));
    setReadyCount(0);
    loadedKeysRef.current = new Set();
    completedRef.current = false;
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
            quaternion={item.quaternion}
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

export function LoadingProgressReporter({
  sceneKey,
  onChange,
}: {
  sceneKey: string;
  onChange: (snapshot: ResourceProgressSnapshot) => void;
}) {
  const frameRef = useRef<number | null>(null);
  const latestSnapshotRef = useRef<ResourceProgressSnapshot | null>(null);

  useEffect(() => {
    const manager = THREE.DefaultLoadingManager;
    const previousOnStart = manager.onStart;
    const previousOnProgress = manager.onProgress;
    const previousOnLoad = manager.onLoad;
    const previousOnError = manager.onError;

    const schedule = (snapshot: ResourceProgressSnapshot) => {
      const current = latestSnapshotRef.current;
      if (
        current &&
        current.active === snapshot.active &&
        current.item === snapshot.item &&
        current.loaded === snapshot.loaded &&
        current.total === snapshot.total &&
        current.progress === snapshot.progress
      ) {
        return;
      }

      latestSnapshotRef.current = snapshot;
      if (frameRef.current !== null) {
        window.cancelAnimationFrame(frameRef.current);
      }
      frameRef.current = window.requestAnimationFrame(() => {
        frameRef.current = null;
        startTransition(() => {
          onChange(snapshot);
        });
      });
    };

    const createSnapshot = (
      active: boolean,
      item: string,
      loaded: number,
      total: number,
    ): ResourceProgressSnapshot => {
      const safeLoaded = Math.max(loaded, 0);
      const safeTotal = Math.max(total, safeLoaded);
      return {
        active,
        item,
        loaded: safeLoaded,
        total: safeTotal,
        progress: active && safeTotal > 0 ? (safeLoaded / safeTotal) * 100 : active ? 0 : 100,
      };
    };

    schedule({
      active: false,
      item: "",
      loaded: 0,
      total: 0,
      progress: 100,
    });

    manager.onStart = (url, loaded, total) => {
      previousOnStart?.(url, loaded, total);
      schedule(createSnapshot(true, url, loaded, total));
    };
    manager.onProgress = (url, loaded, total) => {
      previousOnProgress?.(url, loaded, total);
      schedule(createSnapshot(true, url, loaded, total));
    };
    manager.onLoad = () => {
      previousOnLoad?.();
      const latest = latestSnapshotRef.current;
      schedule(
        createSnapshot(
          false,
          latest?.item ?? "",
          latest?.total ?? latest?.loaded ?? 0,
          latest?.total ?? latest?.loaded ?? 0,
        ),
      );
    };
    manager.onError = (url) => {
      previousOnError?.(url);
      const latest = latestSnapshotRef.current;
      schedule(
        createSnapshot(false, url, latest?.loaded ?? 0, latest?.total ?? latest?.loaded ?? 0),
      );
    };

    return () => {
      if (frameRef.current !== null) {
        window.cancelAnimationFrame(frameRef.current);
        frameRef.current = null;
      }
      manager.onStart = previousOnStart;
      manager.onProgress = previousOnProgress;
      manager.onLoad = previousOnLoad;
      manager.onError = previousOnError;
    };
  }, [onChange, sceneKey]);

  return null;
}

export function formatProgressItemLabel(item: string): string {
  const normalized = item.split("?")[0]?.split("/").filter(Boolean).pop() ?? item;
  if (normalized.length <= 42) {
    return normalized;
  }
  return `${normalized.slice(0, 39)}...`;
}

export function inferParsingState(progress: ResourceProgressSnapshot): boolean {
  return !progress.active && progress.total > 0 && progress.loaded > 0;
}

export function ObjectLabels({ items }: { items: ObjectLabelPlacement[] }) {
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

export function SelectionOverlays({
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

export function ObjectHitTargets({
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

export function SceneBoundsController({
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

export function createEmptyBounds(): SceneBounds {
  return {
    center: [0, 1.5, 0],
    size: [2, 3, 2],
  };
}

export function expandBounds(
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

export function finalizeBounds(min: Vector3Tuple, max: Vector3Tuple): SceneBounds {
  const sizeX = Math.max(max[0] - min[0], 1);
  const sizeY = Math.max(max[1] - min[1], 1);
  const sizeZ = Math.max(max[2] - min[2], 1);
  return {
    center: [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2],
    size: [sizeX, sizeY, sizeZ],
  };
}

export function buildObjectLabels(
  inspectableObjects: InspectableObject[],
  showObjectLabels: boolean,
  selectedObjectId: string | null,
  hoveredObjectId: string | null,
): ObjectLabelPlacement[] {
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
}

export function updateMeasuredBoundsMap(
  current: Record<string, SceneBounds>,
  id: string,
  bounds: SceneBounds,
): Record<string, SceneBounds> {
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
}

export function resolveObjectPosition(
  objectId: string,
  fallback: Vector3Tuple,
  positionOverrides?: Record<string, Vector3Tuple>,
): Vector3Tuple {
  return positionOverrides?.[objectId] ?? fallback;
}

export function resolveObjectRotation(
  objectId: string,
  fallback: number,
  rotationOverrides?: Record<string, number>,
): number {
  return rotationOverrides?.[objectId] ?? fallback;
}

export function normalizeQuaternionTuple(
  quaternion: QuaternionTuple,
  epsilon = 0.000001,
): QuaternionTuple | null {
  const length = Math.hypot(quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
  if (!Number.isFinite(length) || length <= epsilon) {
    return null;
  }

  return [
    quaternion[0] / length,
    quaternion[1] / length,
    quaternion[2] / length,
    quaternion[3] / length,
  ];
}

export function resolveObjectQuaternion(
  objectId: string,
  fallback: QuaternionTuple | null | undefined,
  quaternionOverrides?: Record<string, QuaternionTuple>,
): QuaternionTuple | null {
  const override = quaternionOverrides?.[objectId];
  if (override) {
    return normalizeQuaternionTuple(override);
  }

  if (!fallback) {
    return null;
  }

  return normalizeQuaternionTuple(fallback);
}

export function SageRoomShell({
  room,
  wallOpacity,
  wallDisplayMode,
}: {
  room: RenderableSageRoom;
  wallOpacity: number;
  wallDisplayMode: WallDisplayMode;
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

export function roomShellOpacity(
  category: Renderable3dFrontRoomShell["category"] | RenderableSceneSmithRoomShell["category"],
  wallDisplayMode: WallDisplayMode,
  wallOpacity: number,
): number {
  if (category === "wall" || category === "window" || category === "door" || category === "feature") {
    return resolveWallOpacity(wallDisplayMode, wallOpacity);
  }
  if (category === "ceiling") {
    return Math.max(0.24, Math.min(0.72, wallOpacity));
  }
  return 1;
}

export function SageOpenings({ rooms }: { rooms: RenderableSageRoom[] }) {
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

export { Bounds };
