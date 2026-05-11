import { Suspense, useMemo } from "react";
import { Bounds, Clone, Grid, OrbitControls, useGLTF } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import * as THREE from "three";
import { toRepoAssetUrl } from "../lib/repoAssets";
import type {
  SceneManifest,
  SageDoor,
  SageWall,
  SageWindow,
} from "../types";

interface ScenePreviewCanvasProps {
  scene: SceneManifest | null;
}

type PreviewBox = {
  id: string;
  label: string;
  color: string;
  position: [number, number, number];
  rotationY: number;
  size: [number, number, number];
};

type RoomShell = {
  id: string;
  translation: [number, number, number];
  urls: string[];
};

type WallPanel = {
  id: string;
  position: [number, number, number];
  rotationY: number;
  size: [number, number, number];
};

type MarkerPanel = {
  id: string;
  position: [number, number, number];
  rotationY: number;
  size: [number, number, number];
  color: string;
};

function scenesmithToThree(vector: [number, number, number] | undefined): [number, number, number] {
  if (!vector) {
    return [0, 0, 0];
  }
  return [vector[0] ?? 0, vector[2] ?? 0, vector[1] ?? 0];
}

function wallToPanel(wall: SageWall): WallPanel {
  const start: [number, number, number] = [wall.start_point.x, 0, wall.start_point.y];
  const end: [number, number, number] = [wall.end_point.x, 0, wall.end_point.y];
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
  opening: SageDoor | SageWindow,
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

function defaultSceneSmithSize(objectType: string | null | undefined): [number, number, number] {
  switch (objectType) {
    case "wall_mounted":
      return [0.56, 0.42, 0.12];
    case "ceiling_mounted":
      return [0.34, 0.24, 0.34];
    case "manipuland":
    case "manipulands":
      return [0.2, 0.16, 0.2];
    default:
      return [0.9, 0.82, 0.9];
  }
}

function colorForObjectType(objectType: string | null | undefined): string {
  switch (objectType) {
    case "wall_mounted":
      return "#f59e0b";
    case "ceiling_mounted":
      return "#38bdf8";
    case "manipuland":
    case "manipulands":
      return "#f97316";
    case "furniture":
      return "#8b5cf6";
    default:
      return "#94a3b8";
  }
}

function buildSagePreview(scene: SceneManifest) {
  const boxes: PreviewBox[] = [];
  const wallPanels: WallPanel[] = [];
  const markers: MarkerPanel[] = [];

  const wallsById = new Map<string, SageWall>();
  for (const room of scene.normalized.rooms) {
    for (const wall of room.walls ?? []) {
      wallsById.set(wall.id, wall);
      wallPanels.push(wallToPanel(wall));
    }
    for (const door of room.doors ?? []) {
      const marker = openingToMarker(door, wallsById.get(door.wall_id), "#22c55e");
      if (marker) {
        markers.push(marker);
      }
    }
    for (const window of room.windows ?? []) {
      const marker = openingToMarker(window, wallsById.get(window.wall_id), "#38bdf8");
      if (marker) {
        markers.push(marker);
      }
    }
  }

  for (const object of scene.normalized.objects) {
    const position = object.position ?? { x: 0, y: 0, z: 0 };
    const dimensions = object.dimensions ?? {};
    const width = Math.max(dimensions.width ?? 0.4, 0.15);
    const depth = Math.max(dimensions.length ?? 0.4, 0.15);
    const height = Math.max(dimensions.height ?? 0.4, 0.15);
    boxes.push({
      id: object.id,
      label: object.type || object.name || object.id,
      color: "#8b5cf6",
      position: [position.x, position.z + height / 2, position.y],
      rotationY: THREE.MathUtils.degToRad(-(object.rotation?.z ?? 0)),
      size: [width, height, depth],
    });
  }

  return { boxes, wallPanels, markers, roomShells: [] as RoomShell[] };
}

function buildScenesmithPreview(scene: SceneManifest) {
  const roomFrames = new Map<string, [number, number, number]>();
  const roomShells: RoomShell[] = [];
  const boxes: PreviewBox[] = [];

  for (const room of scene.normalized.rooms) {
    const frame = scenesmithToThree(room.frame?.translation);
    roomFrames.set(room.id, frame);

    const urls = [
      room.floor_plan_assets?.floor_gltf,
      ...(room.floor_plan_assets?.wall_gltfs ?? []),
      ...(room.floor_plan_assets?.window_gltfs ?? []),
    ]
      .map((path) => toRepoAssetUrl(path))
      .filter((value): value is string => Boolean(value));

    if (urls.length > 0) {
      roomShells.push({
        id: room.id,
        translation: frame,
        urls,
      });
    }
  }

  for (const object of scene.normalized.objects) {
    const roomFrame = roomFrames.get(object.room_id) ?? [0, 0, 0];
    const local = scenesmithToThree(object.transform?.translation);
    const defaultSize = defaultSceneSmithSize(object.object_type);
    const bboxSize =
      object.bbox_min && object.bbox_max
        ? ([
            Math.abs(object.bbox_max[0] - object.bbox_min[0]),
            Math.abs(object.bbox_max[2] - object.bbox_min[2]),
            Math.abs(object.bbox_max[1] - object.bbox_min[1]),
          ] as [number, number, number])
        : null;
    const size: [number, number, number] = [
      Math.max(bboxSize?.[0] ?? defaultSize[0], 0.12),
      Math.max(bboxSize?.[1] ?? defaultSize[1], 0.12),
      Math.max(bboxSize?.[2] ?? defaultSize[2], 0.12),
    ];
    const angleDeg = object.transform?.rotation_angle_axis?.angle_deg ?? 0;
    const position: [number, number, number] = [
      roomFrame[0] + local[0],
      roomFrame[1] + local[1] + size[1] / 2,
      roomFrame[2] + local[2],
    ];

    boxes.push({
      id: object.id,
      label: object.name || object.object_type || object.id,
      color: colorForObjectType(object.object_type),
      position,
      rotationY: THREE.MathUtils.degToRad(-angleDeg),
      size,
    });
  }

  return { boxes, roomShells, wallPanels: [] as WallPanel[], markers: [] as MarkerPanel[] };
}

function ShellModel({ url, translation }: { url: string; translation: [number, number, number] }) {
  const gltf = useGLTF(url);
  return <Clone object={gltf.scene} position={translation} />;
}

function PreviewContent({ scene }: { scene: SceneManifest }) {
  const preview = useMemo(() => {
    if (scene.dataset === "sage") {
      return buildSagePreview(scene);
    }
    return buildScenesmithPreview(scene);
  }, [scene]);

  return (
    <>
      <ambientLight intensity={0.8} />
      <directionalLight position={[10, 12, 6]} intensity={1.2} castShadow />
      <directionalLight position={[-8, 10, -6]} intensity={0.5} />
      <Grid
        args={[48, 48]}
        position={[0, -0.001, 0]}
        cellSize={0.6}
        cellThickness={0.5}
        cellColor="#1e293b"
        sectionSize={3}
        sectionThickness={1}
        sectionColor="#334155"
        fadeDistance={70}
        fadeStrength={1}
      />

      <Bounds fit clip observe margin={1.2}>
        <group>
          {preview.wallPanels.map((panel) => (
            <mesh
              key={panel.id}
              position={panel.position}
              rotation={[0, -panel.rotationY, 0]}
            >
              <boxGeometry args={panel.size} />
              <meshStandardMaterial
                color="#cbd5e1"
                transparent
                opacity={0.3}
                roughness={0.82}
              />
            </mesh>
          ))}

          {preview.markers.map((marker) => (
            <mesh
              key={marker.id}
              position={marker.position}
              rotation={[0, -marker.rotationY, 0]}
            >
              <boxGeometry args={marker.size} />
              <meshStandardMaterial color={marker.color} transparent opacity={0.7} />
            </mesh>
          ))}

          <Suspense fallback={null}>
            {preview.roomShells.flatMap((roomShell) =>
              roomShell.urls.map((url) => (
                <ShellModel
                  key={`${roomShell.id}-${url}`}
                  url={url}
                  translation={roomShell.translation}
                />
              )),
            )}
          </Suspense>

          {preview.boxes.map((box) => (
            <mesh
              key={box.id}
              position={box.position}
              rotation={[0, box.rotationY, 0]}
              castShadow
              receiveShadow
            >
              <boxGeometry args={box.size} />
              <meshStandardMaterial
                color={box.color}
                roughness={0.68}
                metalness={0.08}
                transparent
                opacity={0.9}
              />
            </mesh>
          ))}
        </group>
      </Bounds>
    </>
  );
}

export function ScenePreviewCanvas({ scene }: ScenePreviewCanvasProps) {
  if (!scene) {
    return (
      <div className="canvas-empty">
        <p>先选择一个场景</p>
      </div>
    );
  }

  return (
    <div className="canvas-shell">
      <Canvas
        shadows
        camera={{ position: [8, 7, 8], fov: 42, near: 0.1, far: 200 }}
        gl={{ antialias: true }}
      >
        <color attach="background" args={["#020617"]} />
        <PreviewContent scene={scene} />
        <OrbitControls makeDefault enableDamping dampingFactor={0.08} />
      </Canvas>

      <div className="canvas-caption">
        <div>
          <strong>Three.js Preview</strong>
          <span>拖拽旋转，滚轮缩放，右键平移</span>
        </div>
        {scene.dataset === "scenesmith" ? (
          <span className="canvas-badge">SceneSmith: room shell + object proxies</span>
        ) : (
          <span className="canvas-badge">SAGE: layout shell + object proxies</span>
        )}
      </div>
    </div>
  );
}
