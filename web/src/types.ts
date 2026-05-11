export interface DatasetCatalog {
  schema_version: number;
  generated_at_utc: string;
  datasets: DatasetCatalogEntry[];
}

export interface DatasetCatalogEntry {
  dataset: string;
  scene_count: number;
  skipped_count: number;
  index_path: string;
}

export interface RenderableDatasetCatalog {
  schema_version: number;
  generated_at_utc: string;
  datasets: RenderableDatasetCatalogEntry[];
}

export interface RenderableDatasetCatalogEntry {
  dataset: string;
  scene_count: number;
  index_path: string;
}

export interface DatasetIndex {
  schema_version: number;
  generated_at_utc: string;
  dataset: string;
  scene_count: number;
  skipped_count: number;
  scenes: SceneSummary[];
  skipped_scenes: Array<Record<string, unknown>>;
}

export interface SceneSummary {
  scene_id: string;
  scene_uid: string;
  subset?: string | null;
  description?: string | null;
  title?: string | null;
  preview_image?: string | null;
  scene_manifest: string;
  stats: Record<string, number>;
}

export interface RenderableDatasetIndex {
  schema_version: number;
  generated_at_utc: string;
  dataset: "sage" | "scenesmith";
  scene_count: number;
  shared_asset_count?: number;
  scenes: RenderableSceneSummary[];
}

export interface RenderableSceneSummary {
  scene_id: string;
  scene_uid: string;
  subset?: string | null;
  render_manifest: string;
  object_count: number;
  room_count?: number;
  room_shell_count?: number;
}

export interface SceneManifest {
  schema_version: number;
  generated_at_utc: string;
  dataset: "sage" | "scenesmith";
  scene_id: string;
  scene_uid: string;
  subset?: string | null;
  description?: string | null;
  display: {
    title?: string | null;
    subtitle?: string | null;
    preview_images?: string[];
  };
  stats: Record<string, number>;
  assets: Record<string, string | string[] | null>;
  normalized: {
    kind?: string | null;
    scene_state_mode?: string | null;
    layout?: Record<string, unknown> | null;
    building_style?: string | null;
    created_from_text?: string | null;
    policy_analysis?: string | null;
    rooms: NormalizedRoom[];
    objects: NormalizedObject[];
  };
}

export interface NormalizedRoom {
  id: string;
  room_type?: string | null;
  position?: { x: number; y: number; z: number };
  dimensions?: { width?: number; length?: number; height?: number };
  ceiling_height?: number | null;
  floor_material?: string | null;
  walls?: SageWall[];
  doors?: SageDoor[];
  windows?: SageWindow[];
  object_ids?: string[];
  object_count?: number;
  frame?: {
    frame_name?: string;
    translation?: [number, number, number];
    base_frame?: string;
  } | null;
  room_geometry_sdf?: string | null;
  floor_plan_assets?: {
    floor_gltf?: string | null;
    wall_gltfs?: string[];
    window_gltfs?: string[];
  } | null;
  generated_assets?: Record<string, string[]>;
  floor?: unknown;
  objects?: string[];
}

export interface SageWall {
  id: string;
  start_point: { x: number; y: number; z?: number };
  end_point: { x: number; y: number; z?: number };
  height: number;
  thickness: number;
  material?: string | null;
}

export interface SageDoor {
  id: string;
  wall_id: string;
  position_on_wall: number;
  width: number;
  height: number;
  door_type?: string | null;
}

export interface SageWindow {
  id: string;
  wall_id: string;
  position_on_wall: number;
  width: number;
  height: number;
}

export interface NormalizedObject {
  id: string;
  room_id: string;
  type?: string | null;
  name?: string | null;
  description?: string | null;
  position?: { x: number; y: number; z: number };
  rotation?: { x?: number; y?: number; z?: number };
  dimensions?: { width?: number; length?: number; height?: number };
  object_type?: string | null;
  transform?: {
    translation?: [number, number, number];
    rotation_angle_axis?: {
      angle_deg?: number;
      axis?: [number, number, number];
    } | null;
    base_frame?: string | null;
  } | null;
  bbox_min?: [number, number, number] | null;
  bbox_max?: [number, number, number] | null;
  gltf_path?: string | null;
  sdf_path?: string | null;
  mesh?: {
    ply?: string | null;
    texture?: string | null;
  } | null;
  metadata?: Record<string, unknown>;
}

export interface RenderableSageDoor {
  id: string;
  wall_id: string;
  position_on_wall: number;
  width: number;
  height: number;
  texture_path?: string | null;
}

export interface RenderableSageObject {
  id: string;
  asset_path: string;
  position: [number, number, number];
  rotation_y_deg: number;
  scale: [number, number, number];
  native_size: [number, number, number];
  description?: string | null;
  type?: string | null;
  source_id: string;
}

export interface RenderableSageRoom {
  id: string;
  room_type?: string | null;
  dimensions?: { width?: number | null; length?: number | null; height?: number | null };
  ceiling_height?: number | null;
  floor_texture_path?: string | null;
  wall_texture_path?: string | null;
  walls: SageWall[];
  doors: RenderableSageDoor[];
  windows: SageWindow[];
}

export interface RenderableSceneSmithAsset {
  id: string;
  asset_path: string;
  position: [number, number, number];
  rotation_y_deg: number;
  scale: [number, number, number];
  room_id: string;
}

export interface RenderableSceneSmithRoomShell extends RenderableSceneSmithAsset {
  category: "floor" | "wall" | "window";
}

export interface RenderableSceneSmithObject extends RenderableSceneSmithAsset {
  object_type?: string | null;
  description?: string | null;
}

export interface RenderableSageSceneManifest {
  schema_version: number;
  generated_at_utc: string;
  dataset: "sage";
  scene_id: string;
  scene_uid: string;
  source_scene_manifest: string;
  objects: RenderableSageObject[];
  rooms: RenderableSageRoom[];
}

export interface RenderableSceneSmithSceneManifest {
  schema_version: number;
  generated_at_utc: string;
  dataset: "scenesmith";
  scene_id: string;
  scene_uid: string;
  subset?: string | null;
  source_scene_manifest: string;
  room_shells: RenderableSceneSmithRoomShell[];
  objects: RenderableSceneSmithObject[];
}

export type RenderableSceneManifest =
  | RenderableSageSceneManifest
  | RenderableSceneSmithSceneManifest;
