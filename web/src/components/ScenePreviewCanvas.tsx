import { useEffect, useMemo, useState } from "react";
import { Grid, OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import * as THREE from "three";
import type {
  Renderable3dFrontSceneManifest,
  RenderableSceneManifest,
  RenderableSceneWeaverSceneManifest,
  RenderableSceneSmithSceneManifest,
  SceneManifest,
} from "../types";
import { Front3DPreviewContent } from "./scenePreview/front3dContent";
import { RoomLayoutPreviewContent } from "./scenePreview/roomLayoutContent";
import { SceneWeaverPreviewContent } from "./scenePreview/sceneweaverContent";
import { SceneSmithPreviewContent } from "./scenePreview/scenesmithContent";
import {
  LoadingProgressReporter,
  type ProgressStageId,
  PreviewEnvironment,
  type RenderProgressSnapshot,
  type ResourceProgressSnapshot,
  formatProgressItemLabel,
  inferParsingState,
} from "./scenePreview/shared";

interface ScenePreviewCanvasProps {
  scene: SceneManifest | null;
  renderScene: RenderableSceneManifest | null;
  wallOpacity: number;
  wallDisplayMode: "solid" | "transparent" | "hidden" | "wireframe";
  showObjectLabels: boolean;
  onProgressChange?: (snapshot: ScenePreviewProgressSnapshot) => void;
}

type ScenePreviewViewportProps = Omit<ScenePreviewCanvasProps, "renderScene"> & {
  renderScene: RenderableSceneManifest;
};

export interface ScenePreviewProgressSnapshot {
  sceneUid: string;
  currentStage: ProgressStageId;
  statusTitle: string;
  statusDetail: string;
  overallProgress: number;
  previewReady: boolean;
  resourceActive: boolean;
}

function createProgressSnapshot(
  sceneUid: string,
  resourceProgress: ResourceProgressSnapshot,
  renderProgress: RenderProgressSnapshot,
): ScenePreviewProgressSnapshot {
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
  const parsingActive =
    inferParsingState(resourceProgress) && !renderProgress.ready && renderValue === 0;
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

  return {
    sceneUid,
    currentStage,
    statusTitle,
    statusDetail,
    overallProgress,
    previewReady,
    resourceActive,
  };
}

export function ScenePreviewProgressIndicator({
  progress,
  className = "",
}: {
  progress: ScenePreviewProgressSnapshot | null;
  className?: string;
}) {
  if (!progress) {
    return null;
  }

  const { currentStage, overallProgress, previewReady, resourceActive, statusDetail, statusTitle } =
    progress;
  const classes = ["canvas-progress", className, previewReady ? "is-ready" : "is-busy"]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={classes}>
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
            currentStage === "mount" ? "is-current" : currentStage === "ready" ? "is-done" : ""
          }`}
        >
          Mounting scene
        </span>
      </div>
    </div>
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
      {renderScene.dataset === "sage" || renderScene.dataset === "hsm" ? (
        <RoomLayoutPreviewContent
          renderScene={renderScene}
          wallOpacity={wallOpacity}
          wallDisplayMode={wallDisplayMode}
          showObjectLabels={showObjectLabels}
          onRenderProgressChange={onRenderProgressChange}
        />
      ) : renderScene.dataset === "3dfront" ? (
        <Front3DPreviewContent
          scene={scene}
          renderScene={renderScene as Renderable3dFrontSceneManifest}
          wallOpacity={wallOpacity}
          wallDisplayMode={wallDisplayMode}
          showObjectLabels={showObjectLabels}
          onRenderProgressChange={onRenderProgressChange}
        />
      ) : renderScene.dataset === "sceneweaver" ? (
        <SceneWeaverPreviewContent
          key={renderScene.scene_uid}
          renderScene={renderScene as RenderableSceneWeaverSceneManifest}
          showObjectLabels={showObjectLabels}
          onRenderProgressChange={onRenderProgressChange}
        />
      ) : (
        <SceneSmithPreviewContent
          key={renderScene.scene_uid}
          scene={scene}
          renderScene={renderScene as RenderableSceneSmithSceneManifest}
          wallOpacity={wallOpacity}
          wallDisplayMode={wallDisplayMode}
          showObjectLabels={showObjectLabels}
          onRenderProgressChange={onRenderProgressChange}
        />
      )}
    </>
  );
}

function ScenePreviewViewport({
  scene,
  renderScene,
  wallOpacity,
  wallDisplayMode,
  showObjectLabels,
  onProgressChange,
}: ScenePreviewViewportProps) {
  const [resourceProgress, setResourceProgress] = useState<ResourceProgressSnapshot>({
    active: false,
    item: "",
    loaded: 0,
    total: 0,
    progress: 100,
  });
  const [renderProgress, setRenderProgress] = useState<RenderProgressSnapshot>({
    ready: false,
    stage: "Preparing scene",
    detail: "Initializing asset batches",
    completed: 0,
    total: 0,
    progress: 0,
  });

  const dataset = renderScene.dataset;
  const badge =
    dataset === "sage"
      ? `SAGE: ${renderScene.objects.length} textured objects`
      : dataset === "hsm"
        ? `HSM: ${renderScene.rooms.length} procedural rooms + ${renderScene.objects.length} HSSD objects`
        : dataset === "3dfront"
          ? `3D-FRONT: ${renderScene.room_shells.length} room shells + ${renderScene.objects.length} objects`
          : dataset === "sceneweaver"
            ? `SceneWeaver: 1 exported GLB + ${renderScene.objects.length} layout objects`
            : `SceneSmith: ${renderScene.room_shells.length} room shells + ${renderScene.objects.length} objects`;
  const progressSnapshot = useMemo(
    () => createProgressSnapshot(renderScene.scene_uid, resourceProgress, renderProgress),
    [renderProgress, renderScene.scene_uid, resourceProgress],
  );

  useEffect(() => {
    onProgressChange?.(progressSnapshot);
  }, [onProgressChange, progressSnapshot]);

  return (
    <div className="canvas-shell">
      <Canvas
        shadows
        camera={{ position: [8, 7, 8], fov: 42, near: 0.1, far: 300 }}
        gl={{ antialias: true }}
      >
        <color attach="background" args={["#020617"]} />
        <LoadingProgressReporter sceneKey={renderScene.scene_uid} onChange={setResourceProgress} />
        <PreviewContent
          key={renderScene.scene_uid}
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

export function ScenePreviewCanvas(props: ScenePreviewCanvasProps) {
  const { scene, renderScene } = props;

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

  return <ScenePreviewViewport key={renderScene.scene_uid} {...props} renderScene={renderScene} />;
}
