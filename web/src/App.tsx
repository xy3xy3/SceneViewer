import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import {
  Boxes,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Copy,
  Database,
  Image as ImageIcon,
  Layers3,
  Tag,
  ScrollText,
  RotateCcw,
} from "lucide-react";
import {
  ScenePreviewCanvas,
  type ScenePointerDebugSnapshot,
  type ScenePreviewDebugObjectSnapshot,
  ScenePreviewProgressIndicator,
  type ScenePreviewProgressSnapshot,
} from "./components/ScenePreviewCanvas";
import {
  normalizeQuaternionTuple,
  type QuaternionTuple,
  type Vector3Tuple,
} from "./components/scenePreview/shared";
import { fetchRepoJson, toRepoAssetUrl } from "./lib/repoAssets";
import type {
  DatasetCatalog,
  DatasetIndex,
  NormalizedRoom,
  RenderableDatasetCatalog,
  RenderableDatasetIndex,
  RenderableSceneManifest,
  RenderableSceneSummary,
  SceneManifest,
  SceneSummary,
} from "./types";
import "./index.css";

type WallDisplayMode = "solid" | "transparent" | "hidden" | "wireframe";
type DebugAxis = "x" | "y" | "z";
type QuaternionAxis = "x" | "y" | "z" | "w";
type CoordinateDraft = Record<DebugAxis, string>;
type EulerDraft = Record<DebugAxis, string>;
type QuaternionDraft = Record<QuaternionAxis, string>;
type RotationInputSource = "euler" | "quaternion" | null;

type RenderableDebugObject = {
  id: string;
  type?: string | null;
  label: string;
  originalPosition: Vector3Tuple;
  originalQuaternion: QuaternionTuple;
  originalRotationYDeg: number;
  sourceId?: string | null;
};

type ObjectFinderEntry = {
  id: string;
  type: string;
  label: string;
  sourceId?: string | null;
  searchText: string;
};

type ObjectFinderGroup = {
  type: string;
  items: ObjectFinderEntry[];
};

type SearchableSelectOption = {
  value: string;
  label: string;
  searchText?: string;
};

const DEBUG_AXES: Array<{ axis: DebugAxis; index: 0 | 1 | 2 }> = [
  { axis: "x", index: 0 },
  { axis: "y", index: 1 },
  { axis: "z", index: 2 },
];

const QUATERNION_AXES: QuaternionAxis[] = ["x", "y", "z", "w"];
const EULER_ORDER = "XYZ";
const WALL_DISPLAY_MODE_OPTIONS: SearchableSelectOption[] = [
  { value: "solid", label: "Solid" },
  { value: "transparent", label: "Transparent" },
  { value: "hidden", label: "Hidden" },
  { value: "wireframe", label: "Wireframe" },
];

function formatDatasetLabel(dataset: string): string {
  if (dataset === "hsm") {
    return "HSM";
  }
  if (dataset === "sage") {
    return "SAGE";
  }
  if (dataset === "scenesmith") {
    return "SceneSmith";
  }
  if (dataset === "sceneweaver") {
    return "SceneWeaver";
  }
  if (dataset === "hssd") {
    return "HSSD";
  }
  return "3D-FRONT";
}

function formatSceneLabel(
  scene: RenderableSceneSummary,
  metadata?: SceneSummary | null,
): string {
  if (scene.subset) {
    return `${scene.subset} / ${scene.scene_id}`;
  }
  return metadata?.title || scene.scene_id;
}

function normalizeSearchText(value: string): string {
  return value.trim().toLowerCase();
}

function isFuzzyTermMatch(searchText: string, term: string): boolean {
  if (searchText.includes(term)) {
    return true;
  }

  let termIndex = 0;
  for (const character of searchText) {
    if (character === term[termIndex]) {
      termIndex += 1;
      if (termIndex === term.length) {
        return true;
      }
    }
  }

  return false;
}

function filterSearchableOptions(
  options: SearchableSelectOption[],
  query: string,
): SearchableSelectOption[] {
  const terms = normalizeSearchText(query).split(/\s+/).filter(Boolean);
  if (!terms.length) {
    return options;
  }

  return options.filter((option) => {
    const searchText = normalizeSearchText(
      [option.label, option.value, option.searchText ?? ""].join(" "),
    );
    return terms.every((term) => isFuzzyTermMatch(searchText, term));
  });
}

function SearchableSelect({
  ariaLabel,
  disabled = false,
  emptyMessage = "No matches",
  onChange,
  options,
  placeholder = "Choose an option",
  value,
}: {
  ariaLabel: string;
  disabled?: boolean;
  emptyMessage?: string;
  onChange: (value: string) => void;
  options: SearchableSelectOption[];
  placeholder?: string;
  value: string;
}) {
  const listboxId = useId();
  const rootRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(-1);
  const selectedOption = useMemo(
    () => options.find((option) => option.value === value) ?? null,
    [options, value],
  );
  const filteredOptions = useMemo(
    () => filterSearchableOptions(options, query),
    [options, query],
  );
  const dropdownOpen = isOpen && !disabled;
  const inputValue = dropdownOpen ? query : selectedOption?.label ?? "";
  const inputPlaceholder = dropdownOpen ? selectedOption?.label ?? placeholder : placeholder;
  const visibleActiveIndex =
    activeIndex >= 0 && activeIndex < filteredOptions.length
      ? activeIndex
      : filteredOptions.length
        ? 0
        : -1;
  const activeOption = visibleActiveIndex >= 0 ? filteredOptions[visibleActiveIndex] : null;

  const closeDropdown = useCallback(() => {
    setIsOpen(false);
    setQuery("");
    setActiveIndex(-1);
  }, []);

  const openDropdown = useCallback(() => {
    if (disabled) {
      return;
    }

    setIsOpen(true);
    setActiveIndex(filteredOptions.length ? 0 : -1);
  }, [disabled, filteredOptions.length]);

  const selectOption = useCallback(
    (option: SearchableSelectOption) => {
      onChange(option.value);
      closeDropdown();
      inputRef.current?.blur();
    },
    [closeDropdown, onChange],
  );

  useEffect(() => {
    if (!dropdownOpen) {
      return;
    }

    function handleDocumentPointerDown(event: PointerEvent) {
      if (!rootRef.current?.contains(event.target as Node)) {
        closeDropdown();
      }
    }

    document.addEventListener("pointerdown", handleDocumentPointerDown);
    return () => document.removeEventListener("pointerdown", handleDocumentPointerDown);
  }, [closeDropdown, dropdownOpen]);

  function handleInputFocus() {
    setQuery("");
    if (!disabled) {
      setIsOpen(true);
      setActiveIndex(options.length ? 0 : -1);
    }
  }

  function handleInputChange(nextQuery: string) {
    const nextFilteredOptions = filterSearchableOptions(options, nextQuery);
    setQuery(nextQuery);
    setActiveIndex(nextFilteredOptions.length ? 0 : -1);
    if (!isOpen) {
      setIsOpen(true);
    }
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLInputElement>) {
    if (disabled) {
      return;
    }

    if (event.key === "ArrowDown") {
      event.preventDefault();
      if (!dropdownOpen) {
        openDropdown();
        return;
      }

      if (filteredOptions.length) {
        setActiveIndex((current) => (current + 1) % filteredOptions.length);
      }
      return;
    }

    if (event.key === "ArrowUp") {
      event.preventDefault();
      if (!dropdownOpen) {
        openDropdown();
        return;
      }

      if (filteredOptions.length) {
        setActiveIndex((current) => {
          const normalizedIndex = current < 0 ? 0 : current;
          return (normalizedIndex - 1 + filteredOptions.length) % filteredOptions.length;
        });
      }
      return;
    }

    if (event.key === "Enter") {
      if (!dropdownOpen) {
        openDropdown();
        return;
      }

      event.preventDefault();
      if (activeOption) {
        selectOption(activeOption);
      }
      return;
    }

    if (event.key === "Escape") {
      if (dropdownOpen) {
        event.preventDefault();
        closeDropdown();
      }
      return;
    }

    if (event.key === "Tab") {
      closeDropdown();
    }
  }

  return (
    <div
      ref={rootRef}
      className={`searchable-select${dropdownOpen ? " is-open" : ""}${
        disabled ? " is-disabled" : ""
      }`}
    >
      <div className="searchable-select-field">
        <input
          ref={inputRef}
          type="text"
          value={inputValue}
          placeholder={inputPlaceholder}
          onFocus={handleInputFocus}
          onChange={(event) => handleInputChange(event.currentTarget.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          role="combobox"
          aria-label={ariaLabel}
          aria-autocomplete="list"
          aria-expanded={dropdownOpen}
          aria-controls={listboxId}
          aria-activedescendant={
            dropdownOpen && visibleActiveIndex >= 0
              ? `${listboxId}-option-${visibleActiveIndex}`
              : undefined
          }
        />
        <ChevronDown className="searchable-select-icon" size={16} aria-hidden="true" />
      </div>
      {dropdownOpen ? (
        <div id={listboxId} className="searchable-select-menu" role="listbox">
          {filteredOptions.length ? (
            filteredOptions.map((option, index) => (
              <button
                id={`${listboxId}-option-${index}`}
                key={option.value}
                type="button"
                className={`searchable-select-option${
                  index === visibleActiveIndex ? " is-active" : ""
                }${option.value === value ? " is-selected" : ""}`}
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => selectOption(option)}
                role="option"
                aria-selected={option.value === value}
              >
                {option.label}
              </button>
            ))
          ) : (
            <div className="searchable-select-empty">{emptyMessage}</div>
          )}
        </div>
      ) : null}
    </div>
  );
}

function collectPreviewImages(scene: SceneManifest | null): string[] {
  return (scene?.display.preview_images ?? [])
    .map((path) => toRepoAssetUrl(path))
    .filter((value): value is string => Boolean(value));
}

function buildObjectFinderGroups(
  objects: RenderableDebugObject[],
  filterText: string,
): ObjectFinderGroup[] {
  const normalizedFilter = filterText.trim().toLowerCase();
  const groups = new Map<string, ObjectFinderEntry[]>();

  for (const object of objects) {
    const type = object.type?.trim() || "unknown";
    const entry: ObjectFinderEntry = {
      id: object.id,
      type,
      label: object.label,
      sourceId: object.sourceId,
      searchText: [type, object.label, object.id, object.sourceId ?? ""].join(" ").toLowerCase(),
    };

    if (normalizedFilter && !entry.searchText.includes(normalizedFilter)) {
      continue;
    }

    const existing = groups.get(type);
    if (existing) {
      existing.push(entry);
    } else {
      groups.set(type, [entry]);
    }
  }

  return [...groups.entries()]
    .map(([type, items]) => ({
      type,
      items: [...items].sort(
        (left, right) =>
          left.label.localeCompare(right.label, undefined, { sensitivity: "base" }) ||
          left.id.localeCompare(right.id, undefined, { sensitivity: "base" }),
      ),
    }))
    .sort(
      (left, right) =>
        right.items.length - left.items.length ||
        left.type.localeCompare(right.type, undefined, { sensitivity: "base" }),
    );
}

function sourceAssetEntries(scene: SceneManifest | null) {
  if (!scene) {
    return [];
  }

  return Object.entries(scene.assets)
    .map(([key, value]) => {
      if (Array.isArray(value)) {
        return [key, `${value.length} files`] as const;
      }
      return [key, value] as const;
    })
    .filter(([, value]) => Boolean(value));
}

function roomSubtitle(room: NormalizedRoom, dataset: SceneManifest["dataset"]): string {
  if (dataset === "sage" || dataset === "hsm") {
    const width = room.dimensions?.width;
    const length = room.dimensions?.length;
    const objectCount = room.object_ids?.length ?? 0;
    return `${room.room_type || "room"} · ${width ?? "?"}m x ${length ?? "?"}m · ${objectCount} objects`;
  }

  if (dataset === "3dfront") {
    const width = room.dimensions?.width;
    const length = room.dimensions?.length;
    const objectCount = room.object_count ?? room.object_ids?.length ?? 0;
    const shellCount = room.shell_refs?.length ?? 0;
    return `${room.room_type || "room"} · ${width?.toFixed?.(2) ?? "?"}m x ${length?.toFixed?.(2) ?? "?"}m · ${objectCount} objects · ${shellCount} shells`;
  }

  if (dataset === "sceneweaver") {
    const width = room.dimensions?.width;
    const length = room.dimensions?.length;
    const objectCount = room.object_count ?? room.object_ids?.length ?? 0;
    return `${room.room_type || "room"} · ${width?.toFixed?.(2) ?? "?"}m x ${length?.toFixed?.(2) ?? "?"}m · ${objectCount} objects`;
  }

  if (dataset === "hssd") {
    const objectCount = room.object_count ?? room.object_ids?.length ?? 0;
    return `${room.room_type || "stage"} · ${objectCount} objects · stage GLB preview`;
  }

  const objectCount = room.object_count ?? room.objects?.length ?? 0;
  return `${objectCount} objects · ${room.floor_plan_assets?.wall_gltfs?.length ?? 0} wall meshes`;
}

function formatCoordinate(value: number): string {
  if (!Number.isFinite(value)) {
    return "-";
  }
  return value.toFixed(3);
}

function formatCoordinateInput(value: number): string {
  if (!Number.isFinite(value)) {
    return "0";
  }
  return value.toFixed(3);
}

function formatVector(value: Vector3Tuple): string {
  return `${formatCoordinate(value[0])}, ${formatCoordinate(value[1])}, ${formatCoordinate(value[2])}`;
}

function formatQuaternion(value: QuaternionTuple): string {
  return value.map((entry) => formatCoordinate(entry)).join(", ");
}

function positionsEqual(left: Vector3Tuple, right: Vector3Tuple, epsilon = 0.0001): boolean {
  return left.every((value, index) => Math.abs(value - right[index]) <= epsilon);
}

function quaternionFromRotationYDeg(rotationYDeg: number): QuaternionTuple {
  const halfRadians = THREE.MathUtils.degToRad(rotationYDeg) / 2;
  return [0, Math.sin(halfRadians), 0, Math.cos(halfRadians)];
}

function resolveDebugQuaternion(
  quaternion: QuaternionTuple | null | undefined,
  rotationYDeg: number,
): QuaternionTuple {
  if (quaternion) {
    const normalized = normalizeQuaternionTuple(quaternion);
    if (normalized) {
      return normalized;
    }
  }

  return quaternionFromRotationYDeg(rotationYDeg);
}

function quaternionToRotationYDeg(quaternion: QuaternionTuple): number {
  return quaternionToEulerDeg(quaternion)[1];
}

function quaternionToEulerDeg(quaternion: QuaternionTuple): Vector3Tuple {
  const normalized = normalizeQuaternionTuple(quaternion) ?? quaternionFromRotationYDeg(0);
  const euler = new THREE.Euler().setFromQuaternion(
    new THREE.Quaternion(normalized[0], normalized[1], normalized[2], normalized[3]),
    EULER_ORDER,
  );
  return [
    THREE.MathUtils.radToDeg(euler.x),
    THREE.MathUtils.radToDeg(euler.y),
    THREE.MathUtils.radToDeg(euler.z),
  ];
}

function eulerDegToQuaternion(eulerDeg: Vector3Tuple): QuaternionTuple {
  const euler = new THREE.Euler(
    THREE.MathUtils.degToRad(eulerDeg[0]),
    THREE.MathUtils.degToRad(eulerDeg[1]),
    THREE.MathUtils.degToRad(eulerDeg[2]),
    EULER_ORDER,
  );
  const quaternion = new THREE.Quaternion().setFromEuler(euler);
  return [quaternion.x, quaternion.y, quaternion.z, quaternion.w];
}

function quaternionsEqual(
  left: QuaternionTuple,
  right: QuaternionTuple,
  epsilon = 0.0001,
): boolean {
  const normalizedLeft = normalizeQuaternionTuple(left);
  const normalizedRight = normalizeQuaternionTuple(right);
  if (!normalizedLeft || !normalizedRight) {
    return false;
  }

  const sameOrientation =
    normalizedLeft.every((value, index) => Math.abs(value - normalizedRight[index]) <= epsilon) ||
    normalizedLeft.every((value, index) => Math.abs(value + normalizedRight[index]) <= epsilon);

  return sameOrientation;
}

function quaternionDraftFromValue(value: QuaternionTuple): QuaternionDraft {
  return {
    x: formatCoordinateInput(value[0]),
    y: formatCoordinateInput(value[1]),
    z: formatCoordinateInput(value[2]),
    w: formatCoordinateInput(value[3]),
  };
}

function parseQuaternionDraft(value: QuaternionDraft): QuaternionTuple | null {
  const parsed = [Number(value.x), Number(value.y), Number(value.z), Number(value.w)] as QuaternionTuple;
  if (parsed.some((entry) => !Number.isFinite(entry))) {
    return null;
  }

  return normalizeQuaternionTuple(parsed);
}

function eulerDraftFromValue(value: Vector3Tuple): EulerDraft {
  return {
    x: formatCoordinateInput(value[0]),
    y: formatCoordinateInput(value[1]),
    z: formatCoordinateInput(value[2]),
  };
}

function parseEulerDraft(value: EulerDraft): Vector3Tuple | null {
  const parsed = [Number(value.x), Number(value.y), Number(value.z)] as Vector3Tuple;
  if (parsed.some((entry) => !Number.isFinite(entry))) {
    return null;
  }

  return parsed;
}

function buildRenderableDebugObjects(
  scene: SceneManifest | null,
  renderScene: RenderableSceneManifest | null,
): RenderableDebugObject[] {
  if (!renderScene) {
    return [];
  }

  const sourceObjects = new Map((scene?.normalized.objects ?? []).map((object) => [object.id, object] as const));

  switch (renderScene.dataset) {
    case "sage":
      return renderScene.objects.map((object) => ({
        id: object.id,
        type: object.type ?? null,
        label: object.type || object.description || object.source_id || object.id,
        originalPosition: object.position,
        originalQuaternion: quaternionFromRotationYDeg(object.rotation_y_deg),
        originalRotationYDeg: object.rotation_y_deg,
        sourceId: object.source_id,
      }));
    case "hsm":
      return renderScene.objects.map((object) => ({
        id: object.id,
        type: object.object_type || object.semantic_label || object.category || null,
        label:
          object.name ||
          object.semantic_label ||
          object.object_type ||
          object.category ||
          object.description ||
          object.source_id ||
          object.id,
        originalPosition: object.position,
        originalQuaternion: resolveDebugQuaternion(object.quaternion, object.rotation_y_deg),
        originalRotationYDeg: object.rotation_y_deg,
        sourceId: object.source_id,
      }));
    case "scenesmith":
      return renderScene.objects.map((object) => {
        const sourceObject = sourceObjects.get(object.id);
        return {
          id: object.id,
          type: sourceObject?.object_type || object.object_type || null,
          label:
            sourceObject?.name ||
            sourceObject?.object_type ||
            object.object_type ||
            sourceObject?.description ||
            object.description ||
            object.id,
          originalPosition: object.position,
          originalQuaternion: resolveDebugQuaternion(object.quaternion, object.rotation_y_deg),
          originalRotationYDeg: object.rotation_y_deg,
        };
      });
    case "3dfront":
      return renderScene.objects.map((object) => {
        const sourceObject = sourceObjects.get(object.id);
        return {
          id: object.id,
          type: sourceObject?.type || sourceObject?.object_type || object.object_type || null,
          label:
            sourceObject?.name ||
            sourceObject?.type ||
            sourceObject?.object_type ||
            object.object_type ||
            object.description ||
            object.source_ref ||
            object.id,
          originalPosition: object.position,
          originalQuaternion: resolveDebugQuaternion(object.quaternion, object.rotation_y_deg),
          originalRotationYDeg: object.rotation_y_deg,
          sourceId: object.source_model_jid ?? object.source_ref,
        };
      });
    case "sceneweaver":
    case "hssd":
      return renderScene.objects.map((object) => ({
        id: object.id,
        type: object.object_type || null,
        label: object.object_type || object.description || object.source_id || object.id,
        originalPosition: object.position,
        originalQuaternion: resolveDebugQuaternion(object.quaternion, object.rotation_y_deg),
        originalRotationYDeg: object.rotation_y_deg,
        sourceId: object.source_id,
      }));
  }
}

async function copyText(text: string): Promise<boolean> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return true;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "absolute";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  document.body.removeChild(textarea);
  return copied;
}

export default function App() {
  const [catalog, setCatalog] = useState<DatasetCatalog | null>(null);
  const [datasetIndices, setDatasetIndices] = useState<Record<string, DatasetIndex>>({});
  const [renderDatasetIndices, setRenderDatasetIndices] = useState<
    Record<string, RenderableDatasetIndex>
  >({});
  const [sceneCache, setSceneCache] = useState<Record<string, SceneManifest>>({});
  const [renderSceneCache, setRenderSceneCache] = useState<Record<string, RenderableSceneManifest>>(
    {},
  );
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [selectedSceneUid, setSelectedSceneUid] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [wallOpacity, setWallOpacity] = useState(0.35);
  const [wallDisplayMode, setWallDisplayMode] = useState<WallDisplayMode>("transparent");
  const [showObjectLabels, setShowObjectLabels] = useState(false);
  const [selectedObjectId, setSelectedObjectId] = useState<string | null>(null);
  const [objectPositionOverrides, setObjectPositionOverrides] = useState<
    Record<string, Vector3Tuple>
  >({});
  const [objectQuaternionOverrides, setObjectQuaternionOverrides] = useState<
    Record<string, QuaternionTuple>
  >({});
  const [pointerDebug, setPointerDebug] = useState<ScenePointerDebugSnapshot | null>(null);
  const [positionDraft, setPositionDraft] = useState<CoordinateDraft | null>(null);
  const [eulerDraft, setEulerDraft] = useState<EulerDraft | null>(null);
  const [quaternionDraft, setQuaternionDraft] = useState<QuaternionDraft | null>(null);
  const previousSelectedObjectIdRef = useRef<string | null>(null);
  const rotationInputSourceRef = useRef<RotationInputSource>(null);
  const [copyState, setCopyState] = useState<"idle" | "copied" | "failed">("idle");
  const [previewProgress, setPreviewProgress] = useState<ScenePreviewProgressSnapshot | null>(
    null,
  );
  const [hiddenReadyProgressSceneUid, setHiddenReadyProgressSceneUid] = useState<string | null>(
    null,
  );
  const [objectFinderQuery, setObjectFinderQuery] = useState("");
  const [expandedObjectTypes, setExpandedObjectTypes] = useState<Record<string, boolean>>({});

  useEffect(() => {
    let cancelled = false;

    async function loadCatalog() {
      try {
        setLoading(true);
        const [preprocessedCatalog, nextRenderCatalog] = await Promise.all([
          fetchRepoJson<DatasetCatalog>("assets/preprocessed/datasets.json"),
          fetchRepoJson<RenderableDatasetCatalog>("assets/renderable/datasets.json"),
        ]);
        if (cancelled) {
          return;
        }

        const renderEntries = new Map(
          nextRenderCatalog.datasets.map((entry) => [entry.dataset, entry] as const),
        );
        const nextCatalog: DatasetCatalog = {
          ...preprocessedCatalog,
          datasets: preprocessedCatalog.datasets
            .filter((entry) => renderEntries.has(entry.dataset))
            .map((entry) => ({
              ...entry,
              scene_count: renderEntries.get(entry.dataset)?.scene_count ?? entry.scene_count,
              skipped_count: entry.skipped_count ?? 0,
            })),
        };

        const preprocessedIndices = await Promise.all(
          nextCatalog.datasets.map(async (entry) => [
            entry.dataset,
            await fetchRepoJson<DatasetIndex>(entry.index_path),
          ]),
        );
        const renderIndices = await Promise.all(
          nextCatalog.datasets.map(async (entry) => {
            const renderEntry = renderEntries.get(entry.dataset);
            if (!renderEntry) {
              throw new Error(`Missing renderable catalog entry for ${entry.dataset}`);
            }
            return [
              entry.dataset,
              await fetchRepoJson<RenderableDatasetIndex>(renderEntry.index_path),
            ] as const;
          }),
        );

        if (cancelled) {
          return;
        }

        const nextIndices = Object.fromEntries(preprocessedIndices);
        const nextRenderIndices = Object.fromEntries(renderIndices);
        setCatalog(nextCatalog);
        setDatasetIndices(nextIndices);
        setRenderDatasetIndices(nextRenderIndices);

        const defaultDataset = nextCatalog.datasets[0]?.dataset ?? "";
        const defaultScene = nextRenderIndices[defaultDataset]?.scenes[0]?.scene_uid ?? "";
        setSelectedDataset(defaultDataset);
        setSelectedSceneUid(defaultScene);
        setError(null);
      } catch (loadError) {
        if (!cancelled) {
          const message =
            loadError instanceof Error ? loadError.message : "Failed to load dataset catalog";
          setError(message);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void loadCatalog();

    return () => {
      cancelled = true;
    };
  }, []);

  const selectedDatasetIndex = renderDatasetIndices[selectedDataset];
  const selectedPreprocessedDatasetIndex = datasetIndices[selectedDataset];
  const selectedSceneRenderSummary = selectedDatasetIndex?.scenes.find(
    (scene) => scene.scene_uid === selectedSceneUid,
  );
  const selectedSceneIndex = selectedDatasetIndex?.scenes.findIndex(
    (scene) => scene.scene_uid === selectedSceneUid,
  ) ?? -1;
  const hasPreviousScene = selectedSceneIndex > 0;
  const hasNextScene =
    selectedSceneIndex >= 0 &&
    selectedSceneIndex < (selectedDatasetIndex?.scenes.length ?? 0) - 1;
  const selectedSceneSummary = selectedPreprocessedDatasetIndex?.scenes.find(
    (scene) => scene.scene_uid === selectedSceneUid,
  );

  const preprocessedSceneSummaryMap = useMemo(
    () =>
      new Map(
        (selectedPreprocessedDatasetIndex?.scenes ?? []).map((scene) => [scene.scene_uid, scene] as const),
      ),
    [selectedPreprocessedDatasetIndex],
  );
  const datasetSelectOptions = useMemo<SearchableSelectOption[]>(
    () =>
      (catalog?.datasets ?? []).map((entry) => ({
        value: entry.dataset,
        label: `${formatDatasetLabel(entry.dataset)} (${entry.scene_count})`,
        searchText: entry.dataset,
      })),
    [catalog],
  );
  const sceneSelectOptions = useMemo<SearchableSelectOption[]>(
    () =>
      (selectedDatasetIndex?.scenes ?? []).map((scene) => {
        const metadata = preprocessedSceneSummaryMap.get(scene.scene_uid);
        return {
          value: scene.scene_uid,
          label: formatSceneLabel(scene, metadata),
          searchText: [
            scene.scene_uid,
            scene.scene_id,
            scene.subset ?? "",
            metadata?.title ?? "",
            metadata?.description ?? "",
          ].join(" "),
        };
      }),
    [preprocessedSceneSummaryMap, selectedDatasetIndex],
  );

  useEffect(() => {
    if (!selectedSceneSummary) {
      return;
    }

    const summary = selectedSceneSummary;
    if (sceneCache[summary.scene_uid]) {
      return;
    }

    let cancelled = false;
    async function loadScene() {
      try {
        const manifest = await fetchRepoJson<SceneManifest>(summary.scene_manifest);
        if (cancelled) {
          return;
        }
        setSceneCache((current) => ({
          ...current,
          [summary.scene_uid]: manifest,
        }));
      } catch (loadError) {
        if (!cancelled) {
          const message =
            loadError instanceof Error ? loadError.message : "Failed to load scene manifest";
          setError(message);
        }
      }
    }

    void loadScene();

    return () => {
      cancelled = true;
    };
  }, [sceneCache, selectedSceneSummary]);

  useEffect(() => {
    if (!selectedSceneRenderSummary) {
      return;
    }

    const summary = selectedSceneRenderSummary;
    if (renderSceneCache[summary.scene_uid]) {
      return;
    }

    let cancelled = false;
    async function loadRenderScene() {
      try {
        const manifest = await fetchRepoJson<RenderableSceneManifest>(summary.render_manifest);
        if (cancelled) {
          return;
        }
        setRenderSceneCache((current) => ({
          ...current,
          [summary.scene_uid]: manifest,
        }));
      } catch (loadError) {
        if (!cancelled) {
          const message =
            loadError instanceof Error ? loadError.message : "Failed to load render manifest";
          setError(message);
        }
      }
    }

    void loadRenderScene();

    return () => {
      cancelled = true;
    };
  }, [renderSceneCache, selectedSceneRenderSummary]);

  const selectedScene = selectedSceneSummary
    ? sceneCache[selectedSceneSummary.scene_uid] ?? null
    : null;
  const selectedRenderScene = selectedSceneRenderSummary
    ? renderSceneCache[selectedSceneRenderSummary.scene_uid] ?? null
    : null;
  const visiblePreviewProgress =
    previewProgress?.sceneUid === selectedRenderScene?.scene_uid ? previewProgress : null;
  const topbarPreviewProgress =
    visiblePreviewProgress?.previewReady &&
    hiddenReadyProgressSceneUid === visiblePreviewProgress.sceneUid
      ? null
      : visiblePreviewProgress;

  const renderableDebugObjects = useMemo(
    () => buildRenderableDebugObjects(selectedScene, selectedRenderScene),
    [selectedRenderScene, selectedScene],
  );
  const previewImages = useMemo(() => collectPreviewImages(selectedScene), [selectedScene]);
  const objectFinderGroups = useMemo(
    () => buildObjectFinderGroups(renderableDebugObjects, objectFinderQuery),
    [objectFinderQuery, renderableDebugObjects],
  );
  const assetEntries = useMemo(() => sourceAssetEntries(selectedScene), [selectedScene]);
  const selectedSceneLabel = selectedSceneRenderSummary
    ? formatSceneLabel(
        selectedSceneRenderSummary,
        preprocessedSceneSummaryMap.get(selectedSceneRenderSummary.scene_uid),
      )
    : selectedSceneSummary?.title || selectedSceneUid || "Choose a scene";
  const renderableDebugObjectMap = useMemo(
    () => new Map(renderableDebugObjects.map((object) => [object.id, object] as const)),
    [renderableDebugObjects],
  );
  const selectedObjectDebugInfo = useMemo<ScenePreviewDebugObjectSnapshot | null>(() => {
    if (!selectedObjectId) {
      return null;
    }

    const selectedObject = renderableDebugObjectMap.get(selectedObjectId);
    if (!selectedObject) {
      return null;
    }

    const overridePosition = objectPositionOverrides[selectedObject.id];
    const overrideQuaternion = objectQuaternionOverrides[selectedObject.id];
    const currentQuaternion = overrideQuaternion ?? selectedObject.originalQuaternion;
    return {
      id: selectedObject.id,
      label: selectedObject.label,
      originalPosition: selectedObject.originalPosition,
      currentPosition: overridePosition ?? selectedObject.originalPosition,
      originalQuaternion: selectedObject.originalQuaternion,
      currentQuaternion,
      originalRotationYDeg: quaternionToRotationYDeg(selectedObject.originalQuaternion),
      currentRotationYDeg: quaternionToRotationYDeg(currentQuaternion),
      hasRotationOverride: overrideQuaternion !== undefined,
      hasOverride: Boolean(overridePosition),
    };
  }, [objectPositionOverrides, objectQuaternionOverrides, renderableDebugObjectMap, selectedObjectId]);
  const selectedObjectDelta = selectedObjectDebugInfo
    ? ([
        selectedObjectDebugInfo.currentPosition[0] - selectedObjectDebugInfo.originalPosition[0],
        selectedObjectDebugInfo.currentPosition[1] - selectedObjectDebugInfo.originalPosition[1],
        selectedObjectDebugInfo.currentPosition[2] - selectedObjectDebugInfo.originalPosition[2],
      ] as Vector3Tuple)
    : null;
  const hasPositionOverrides = Object.keys(objectPositionOverrides).length > 0;
  const hasRotationOverrides = Object.keys(objectQuaternionOverrides).length > 0;

  useEffect(() => {
    if (copyState === "idle") {
      return;
    }

    const timer = window.setTimeout(() => setCopyState("idle"), 1800);
    return () => window.clearTimeout(timer);
  }, [copyState]);

  useEffect(() => {
    setSelectedObjectId(null);
    setObjectPositionOverrides({});
    setObjectQuaternionOverrides({});
    setObjectFinderQuery("");
    setExpandedObjectTypes({});
    setPointerDebug(null);
    setPositionDraft(null);
    setEulerDraft(null);
    setQuaternionDraft(null);
    previousSelectedObjectIdRef.current = null;
    rotationInputSourceRef.current = null;
  }, [selectedSceneUid]);

  useEffect(() => {
    if (!selectedObjectId || renderableDebugObjectMap.has(selectedObjectId)) {
      return;
    }
    setSelectedObjectId(null);
  }, [renderableDebugObjectMap, selectedObjectId]);

  useEffect(() => {
    if (!selectedObjectId) {
      return;
    }

    const selectedObject = renderableDebugObjectMap.get(selectedObjectId);
    const selectedType = selectedObject?.type?.trim() || "unknown";
    setExpandedObjectTypes((current) =>
      current[selectedType]
        ? current
        : {
            ...current,
            [selectedType]: true,
          },
    );
  }, [renderableDebugObjectMap, selectedObjectId]);

  useEffect(() => {
    if (!selectedObjectDebugInfo) {
      setPositionDraft(null);
      setEulerDraft(null);
      setQuaternionDraft(null);
      previousSelectedObjectIdRef.current = null;
      rotationInputSourceRef.current = null;
      return;
    }

    setPositionDraft({
      x: formatCoordinateInput(selectedObjectDebugInfo.currentPosition[0]),
      y: formatCoordinateInput(selectedObjectDebugInfo.currentPosition[1]),
      z: formatCoordinateInput(selectedObjectDebugInfo.currentPosition[2]),
    });

    const nextEulerDraft = eulerDraftFromValue(
      quaternionToEulerDeg(selectedObjectDebugInfo.currentQuaternion),
    );
    const nextQuaternionDraft = quaternionDraftFromValue(selectedObjectDebugInfo.currentQuaternion);
    const selectedObjectChanged = previousSelectedObjectIdRef.current !== selectedObjectDebugInfo.id;

    if (selectedObjectChanged) {
      previousSelectedObjectIdRef.current = selectedObjectDebugInfo.id;
      setEulerDraft(nextEulerDraft);
      setQuaternionDraft(nextQuaternionDraft);
      rotationInputSourceRef.current = null;
      return;
    }

    setQuaternionDraft(nextQuaternionDraft);
    if (rotationInputSourceRef.current !== "euler") {
      setEulerDraft(nextEulerDraft);
    }
    rotationInputSourceRef.current = null;
  }, [
    selectedObjectDebugInfo?.id,
    selectedObjectDebugInfo?.currentPosition[0],
    selectedObjectDebugInfo?.currentPosition[1],
    selectedObjectDebugInfo?.currentPosition[2],
    selectedObjectDebugInfo?.currentQuaternion[0],
    selectedObjectDebugInfo?.currentQuaternion[1],
    selectedObjectDebugInfo?.currentQuaternion[2],
    selectedObjectDebugInfo?.currentQuaternion[3],
  ]);

  useEffect(() => {
    if (!visiblePreviewProgress?.previewReady) {
      return;
    }

    const timer = window.setTimeout(() => {
      setHiddenReadyProgressSceneUid(visiblePreviewProgress.sceneUid);
    }, 1000);

    return () => window.clearTimeout(timer);
  }, [visiblePreviewProgress?.previewReady, visiblePreviewProgress?.sceneUid]);

  function handleDatasetChange(dataset: string) {
    setSelectedDataset(dataset);
    const nextScene = renderDatasetIndices[dataset]?.scenes[0]?.scene_uid ?? "";
    setSelectedSceneUid(nextScene);
  }

  function handleWallOpacityChange(value: string) {
    setWallOpacity(Number(value) / 100);
  }

  const handleObjectTypeToggle = useCallback((type: string) => {
    setExpandedObjectTypes((current) => ({
      ...current,
      [type]: !current[type],
    }));
  }, []);

  const handlePreviewProgressChange = useCallback((snapshot: ScenePreviewProgressSnapshot) => {
    setPreviewProgress(snapshot);
  }, []);

  const handlePointerDebugChange = useCallback((snapshot: ScenePointerDebugSnapshot | null) => {
    setPointerDebug(snapshot);
  }, []);

  const updateSelectedObjectPosition = useCallback(
    (objectId: string, nextPosition: Vector3Tuple) => {
      const sourceObject = renderableDebugObjectMap.get(objectId);
      if (!sourceObject) {
        return;
      }

      setObjectPositionOverrides((current) => {
        if (positionsEqual(sourceObject.originalPosition, nextPosition)) {
          if (!(objectId in current)) {
            return current;
          }

          const next = { ...current };
          delete next[objectId];
          return next;
        }

        return {
          ...current,
          [objectId]: nextPosition,
        };
      });
    },
    [renderableDebugObjectMap],
  );

  const handlePositionDraftChange = useCallback(
    (axis: DebugAxis, value: string) => {
      if (!selectedObjectDebugInfo) {
        return;
      }

      setPositionDraft((current) => ({
        x: current?.x ?? formatCoordinateInput(selectedObjectDebugInfo.currentPosition[0]),
        y: current?.y ?? formatCoordinateInput(selectedObjectDebugInfo.currentPosition[1]),
        z: current?.z ?? formatCoordinateInput(selectedObjectDebugInfo.currentPosition[2]),
        [axis]: value,
      }));

      const parsed = Number(value);
      if (!Number.isFinite(parsed)) {
        return;
      }

      const axisIndex = DEBUG_AXES.find((entry) => entry.axis === axis)?.index;
      if (axisIndex === undefined) {
        return;
      }

      const nextPosition = [...selectedObjectDebugInfo.currentPosition] as Vector3Tuple;
      nextPosition[axisIndex] = parsed;
      updateSelectedObjectPosition(selectedObjectDebugInfo.id, nextPosition);
    },
    [selectedObjectDebugInfo, updateSelectedObjectPosition],
  );

  const resetPositionDraft = useCallback(() => {
    if (!selectedObjectDebugInfo) {
      setPositionDraft(null);
      return;
    }

    setPositionDraft({
      x: formatCoordinateInput(selectedObjectDebugInfo.currentPosition[0]),
      y: formatCoordinateInput(selectedObjectDebugInfo.currentPosition[1]),
      z: formatCoordinateInput(selectedObjectDebugInfo.currentPosition[2]),
    });
  }, [selectedObjectDebugInfo]);

  const handlePositionDraftBlur = useCallback(
    (axis: DebugAxis) => {
      if (!positionDraft) {
        return;
      }

      const value = Number(positionDraft[axis]);
      if (Number.isFinite(value)) {
        return;
      }

      resetPositionDraft();
    },
    [positionDraft, resetPositionDraft],
  );

  const handleResetSelectedObjectPosition = useCallback(() => {
    if (!selectedObjectDebugInfo) {
      return;
    }
    updateSelectedObjectPosition(selectedObjectDebugInfo.id, selectedObjectDebugInfo.originalPosition);
  }, [selectedObjectDebugInfo, updateSelectedObjectPosition]);

  const handleResetAllObjectPositions = useCallback(() => {
    setObjectPositionOverrides({});
  }, []);

  const updateSelectedObjectQuaternion = useCallback(
    (objectId: string, nextQuaternion: QuaternionTuple) => {
      const sourceObject = renderableDebugObjectMap.get(objectId);
      if (!sourceObject) {
        return;
      }

      const normalized = normalizeQuaternionTuple(nextQuaternion);
      if (!normalized) {
        return;
      }

      setObjectQuaternionOverrides((current) => {
        if (quaternionsEqual(sourceObject.originalQuaternion, normalized)) {
          if (!(objectId in current)) {
            return current;
          }

          const next = { ...current };
          delete next[objectId];
          return next;
        }

        return {
          ...current,
          [objectId]: normalized,
        };
      });
    },
    [renderableDebugObjectMap],
  );

  const handleQuaternionDraftChange = useCallback(
    (axis: QuaternionAxis, value: string) => {
      if (!selectedObjectDebugInfo) {
        return;
      }

      rotationInputSourceRef.current = "quaternion";
      const nextDraft = {
        ...(quaternionDraft ?? quaternionDraftFromValue(selectedObjectDebugInfo.currentQuaternion)),
        [axis]: value,
      };
      setQuaternionDraft(nextDraft);

      const parsed = parseQuaternionDraft(nextDraft);
      if (!parsed) {
        return;
      }

      updateSelectedObjectQuaternion(selectedObjectDebugInfo.id, parsed);
    },
    [quaternionDraft, selectedObjectDebugInfo, updateSelectedObjectQuaternion],
  );

  const handleEulerDraftChange = useCallback(
    (axis: DebugAxis, value: string) => {
      if (!selectedObjectDebugInfo) {
        return;
      }

      rotationInputSourceRef.current = "euler";
      const nextDraft = {
        ...(eulerDraft ?? eulerDraftFromValue(quaternionToEulerDeg(selectedObjectDebugInfo.currentQuaternion))),
        [axis]: value,
      };
      setEulerDraft(nextDraft);

      const parsed = parseEulerDraft(nextDraft);
      if (!parsed) {
        return;
      }

      const nextQuaternion = eulerDegToQuaternion(parsed);
      setQuaternionDraft(quaternionDraftFromValue(nextQuaternion));
      updateSelectedObjectQuaternion(
        selectedObjectDebugInfo.id,
        nextQuaternion,
      );
    },
    [eulerDraft, selectedObjectDebugInfo, updateSelectedObjectQuaternion],
  );

  const resetQuaternionDraft = useCallback(() => {
    if (!selectedObjectDebugInfo) {
      setQuaternionDraft(null);
      return;
    }

    setQuaternionDraft(quaternionDraftFromValue(selectedObjectDebugInfo.currentQuaternion));
  }, [selectedObjectDebugInfo]);

  const resetEulerDraft = useCallback(() => {
    if (!selectedObjectDebugInfo) {
      setEulerDraft(null);
      return;
    }

    setEulerDraft(eulerDraftFromValue(quaternionToEulerDeg(selectedObjectDebugInfo.currentQuaternion)));
  }, [selectedObjectDebugInfo]);

  const handleEulerDraftBlur = useCallback(() => {
    if (!eulerDraft) {
      return;
    }

    const parsed = parseEulerDraft(eulerDraft);
    if (parsed) {
      setEulerDraft(eulerDraftFromValue(parsed));
      return;
    }

    resetEulerDraft();
  }, [eulerDraft, resetEulerDraft]);

  const handleQuaternionDraftBlur = useCallback(() => {
    if (!quaternionDraft) {
      return;
    }

    const parsed = parseQuaternionDraft(quaternionDraft);
    if (parsed) {
      setQuaternionDraft(quaternionDraftFromValue(parsed));
      return;
    }

    resetQuaternionDraft();
  }, [quaternionDraft, resetQuaternionDraft]);

  const handleResetSelectedObjectRotation = useCallback(() => {
    if (!selectedObjectDebugInfo) {
      return;
    }
    updateSelectedObjectQuaternion(
      selectedObjectDebugInfo.id,
      selectedObjectDebugInfo.originalQuaternion,
    );
  }, [selectedObjectDebugInfo, updateSelectedObjectQuaternion]);

  const handleResetAllObjectRotations = useCallback(() => {
    setObjectQuaternionOverrides({});
  }, []);

  function handleSceneStep(direction: -1 | 1) {
    if (!selectedDatasetIndex?.scenes.length || selectedSceneIndex < 0) {
      return;
    }

    const nextScene = selectedDatasetIndex.scenes[selectedSceneIndex + direction];
    if (!nextScene) {
      return;
    }

    setSelectedSceneUid(nextScene.scene_uid);
  }

  async function handleCopySceneName() {
    if (!selectedSceneLabel || selectedSceneLabel === "Choose a scene") {
      setCopyState("failed");
      return;
    }

    try {
      const copied = await copyText(selectedSceneLabel);
      setCopyState(copied ? "copied" : "failed");
    } catch {
      setCopyState("failed");
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <Layers3 size={18} />
          </div>
          <div>
            <h1>SceneViewer</h1>
            <p>Preview HSM, SAGE, SceneSmith, SceneWeaver, HSSD, and 3D-FRONT scenes with repo-local renderable assets.</p>
          </div>
        </div>

        <div className="toolbar">
          <div className="select-shell">
            <span>Dataset</span>
            <SearchableSelect
              ariaLabel="Dataset"
              value={selectedDataset}
              onChange={handleDatasetChange}
              disabled={loading || !(catalog?.datasets.length ?? 0)}
              options={datasetSelectOptions}
              placeholder="Choose a dataset"
              emptyMessage="No datasets found"
            />
          </div>

          <div className="select-shell select-shell-scene">
            <span>Scene</span>
            <div className="scene-select-row">
              <button
                type="button"
                className="scene-switch-button"
                onClick={() => handleSceneStep(-1)}
                disabled={loading || !hasPreviousScene}
                aria-label="Previous scene"
              >
                <ChevronLeft size={16} />
              </button>
              <SearchableSelect
                ariaLabel="Scene"
                value={selectedSceneUid}
                onChange={setSelectedSceneUid}
                disabled={loading || !(selectedDatasetIndex?.scenes.length ?? 0)}
                options={sceneSelectOptions}
                placeholder="Choose a scene"
                emptyMessage="No scenes found"
              />
              <button
                type="button"
                className="scene-switch-button"
                onClick={() => handleSceneStep(1)}
                disabled={loading || !hasNextScene}
                aria-label="Next scene"
              >
                <ChevronRight size={16} />
              </button>
            </div>
          </div>

          <label className="control-shell control-shell-range">
            <span>Wall Opacity</span>
            <div className="range-control">
              <input
                type="range"
                min={0}
                max={100}
                step={5}
                value={Math.round(wallOpacity * 100)}
                onInput={(event) => handleWallOpacityChange(event.currentTarget.value)}
                onChange={(event) => handleWallOpacityChange(event.currentTarget.value)}
              />
              <strong>{Math.round(wallOpacity * 100)}%</strong>
            </div>
          </label>

          <div className="select-shell">
            <span>Wall View</span>
            <SearchableSelect
              ariaLabel="Wall view"
              value={wallDisplayMode}
              onChange={(nextValue) => setWallDisplayMode(nextValue as WallDisplayMode)}
              options={WALL_DISPLAY_MODE_OPTIONS}
              placeholder="Choose wall view"
            />
          </div>

          <label className="control-shell control-shell-toggle">
            <span>Object Labels</span>
            <div className="toggle-control">
              <Tag size={14} />
              <input
                type="checkbox"
                checked={showObjectLabels}
                onChange={(event) => setShowObjectLabels(event.target.checked)}
              />
              <strong>{showObjectLabels ? "On" : "Off"}</strong>
            </div>
          </label>
        </div>

        {selectedRenderScene ? (
          <ScenePreviewProgressIndicator
            progress={topbarPreviewProgress}
            className="topbar-progress"
          />
        ) : null}
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <main className="workspace">
        <section className="preview-panel">
          <ScenePreviewCanvas
            scene={selectedScene}
            renderScene={selectedRenderScene}
            wallOpacity={wallOpacity}
            wallDisplayMode={wallDisplayMode}
            showObjectLabels={showObjectLabels}
            selectedObjectId={selectedObjectId}
            selectedObjectDebugInfo={selectedObjectDebugInfo}
            objectPositionOverrides={objectPositionOverrides}
            objectQuaternionOverrides={objectQuaternionOverrides}
            onSelectedObjectChange={setSelectedObjectId}
            onPointerDebugChange={handlePointerDebugChange}
            onProgressChange={handlePreviewProgressChange}
          />
        </section>

        <aside className="info-panel">
          <div className="info-panel-header">
            <div className="scene-identity">
              <span>Scene Name</span>
              <strong>{selectedSceneLabel}</strong>
            </div>
            <button type="button" className="scene-copy-button" onClick={handleCopySceneName}>
              {copyState === "copied" ? <Check size={15} /> : <Copy size={15} />}
              <span>
                {copyState === "copied"
                  ? "Copied"
                  : copyState === "failed"
                    ? "Retry"
                    : "Copy"}
              </span>
            </button>
          </div>

          <section className="info-card">
            <div className="section-title">
              <ScrollText size={16} />
              <h3>Prompt / Description</h3>
            </div>
            <p className="long-copy">
              {selectedScene?.description || "This scene does not include a richer prompt yet."}
            </p>
          </section>

          <section className="info-card">
            <div className="section-title">
              <Database size={16} />
              <h3>Overview</h3>
            </div>
            <div className="metric-grid">
              {Object.entries(selectedScene?.stats ?? {}).map(([key, value]) => (
                <div key={key} className="metric-card">
                  <span>{key.replaceAll("_", " ")}</span>
                  <strong>{value}</strong>
                </div>
              ))}
            </div>
          </section>

          <section className="info-card">
            <div className="section-title">
              <Tag size={16} />
              <h3>Debug Coordinates</h3>
            </div>

            <div className="debug-coordinate-grid">
              <div className="debug-coordinate-card">
                <span>Canvas</span>
                <strong>
                  {pointerDebug ? `${pointerDebug.canvas[0]}, ${pointerDebug.canvas[1]}` : "--"}
                </strong>
                <p>Mouse position inside the preview canvas</p>
              </div>
              <div className="debug-coordinate-card">
                <span>World</span>
                <strong>{pointerDebug?.world ? formatVector(pointerDebug.world) : "--"}</strong>
                <p>Ground-plane coordinates under the cursor</p>
              </div>
            </div>

            {selectedObjectDebugInfo ? (
              <div className="debug-editor">
                <div className="debug-object-summary">
                  <div>
                    <span>Selected</span>
                    <strong>{selectedObjectDebugInfo.label}</strong>
                    <p>{selectedObjectDebugInfo.id}</p>
                  </div>
                  <div className="debug-pill">
                    {selectedObjectDebugInfo.hasOverride || selectedObjectDebugInfo.hasRotationOverride
                      ? "Simulated override"
                      : "Original render position"}
                  </div>
                </div>

                <div className="debug-coordinate-list">
                  <div className="debug-coordinate-row">
                    <span>Original</span>
                    <code>{formatVector(selectedObjectDebugInfo.originalPosition)}</code>
                  </div>
                  <div className="debug-coordinate-row">
                    <span>Current</span>
                    <code>{formatVector(selectedObjectDebugInfo.currentPosition)}</code>
                  </div>
                  <div className="debug-coordinate-row">
                    <span>Delta</span>
                    <code>{selectedObjectDelta ? formatVector(selectedObjectDelta) : "--"}</code>
                  </div>
                </div>

                <div className="debug-axis-editor">
                  {DEBUG_AXES.map(({ axis }) => (
                    <label key={axis} className="debug-axis-field">
                      <span>{axis.toUpperCase()}</span>
                      <input
                        type="number"
                        inputMode="decimal"
                        step="0.05"
                        value={positionDraft?.[axis] ?? ""}
                        onChange={(event) => handlePositionDraftChange(axis, event.target.value)}
                        onBlur={() => handlePositionDraftBlur(axis)}
                      />
                    </label>
                  ))}
                </div>

                <div className="debug-rotation-section">
                  <div className="debug-coordinate-list">
                    <div className="debug-coordinate-row">
                      <span>Euler {EULER_ORDER} (Resolved)</span>
                      <code>{formatVector(quaternionToEulerDeg(selectedObjectDebugInfo.currentQuaternion))}</code>
                    </div>
                    <div className="debug-coordinate-row">
                      <span>rotation_xyzw (Original)</span>
                      <code>{formatQuaternion(selectedObjectDebugInfo.originalQuaternion)}</code>
                    </div>
                    <div className="debug-coordinate-row">
                      <span>rotation_xyzw (Current)</span>
                      <code>{formatQuaternion(selectedObjectDebugInfo.currentQuaternion)}</code>
                    </div>
                    <div className="debug-coordinate-row">
                      <span>Rotation Y (Derived)</span>
                      <code>{formatCoordinate(selectedObjectDebugInfo.originalRotationYDeg)}°</code>
                    </div>
                    <div className="debug-coordinate-row">
                      <span>Rotation Y (Current)</span>
                      <code>{formatCoordinate(selectedObjectDebugInfo.currentRotationYDeg)}°</code>
                    </div>
                  </div>

                  <div className="debug-axis-editor">
                    {DEBUG_AXES.map(({ axis }) => (
                      <label key={`euler-${axis}`} className="debug-axis-field">
                        <span>{`Euler ${axis.toUpperCase()}`}</span>
                        <input
                          type="number"
                          inputMode="decimal"
                          step="1"
                          value={eulerDraft?.[axis] ?? ""}
                          onChange={(event) => handleEulerDraftChange(axis, event.target.value)}
                          onBlur={handleEulerDraftBlur}
                        />
                      </label>
                    ))}
                  </div>

                  <div className="debug-axis-editor debug-axis-editor-quaternion">
                    {QUATERNION_AXES.map((axis) => (
                      <label key={axis} className="debug-axis-field">
                        <span>{axis.toUpperCase()}</span>
                        <input
                          type="number"
                          inputMode="decimal"
                          step="0.01"
                          value={quaternionDraft?.[axis] ?? ""}
                          onChange={(event) =>
                            handleQuaternionDraftChange(axis, event.target.value)
                          }
                          onBlur={handleQuaternionDraftBlur}
                        />
                      </label>
                    ))}
                  </div>
                </div>

                <div className="debug-actions">
                  <button
                    type="button"
                    className="debug-action-button"
                    onClick={handleResetSelectedObjectPosition}
                    disabled={!selectedObjectDebugInfo.hasOverride}
                  >
                    <RotateCcw size={14} />
                    <span>Reset Position</span>
                  </button>
                  <button
                    type="button"
                    className="debug-action-button"
                    onClick={handleResetSelectedObjectRotation}
                    disabled={!selectedObjectDebugInfo.hasRotationOverride}
                  >
                    <RotateCcw size={14} />
                    <span>Reset Rotation</span>
                  </button>
                  <button
                    type="button"
                    className="debug-action-button debug-action-button-secondary"
                    onClick={() => { handleResetAllObjectPositions(); handleResetAllObjectRotations(); }}
                    disabled={!hasPositionOverrides && !hasRotationOverrides}
                  >
                    <RotateCcw size={14} />
                    <span>Reset All</span>
                  </button>
                </div>

                <p className="debug-hint">
                  这些改动只在当前网页会话里生效，用于人类调试，不会写回场景文件。支持修改
                  <code>position (x/y/z)</code>、<code>Euler {EULER_ORDER}</code> 和 <code>rotation_xyzw</code>。
                </p>
              </div>
            ) : (
              <p className="long-copy">
                在左侧 3D 预览里点击任意物体后，这里会显示它的当前坐标和旋转，并允许你临时模拟修改
                <code>position (x/y/z)</code>、<code>Euler {EULER_ORDER}</code> 和 <code>rotation_xyzw</code>。
              </p>
            )}
          </section>

          <section className="info-card">
            <div className="section-title">
              <Layers3 size={16} />
              <h3>Rooms</h3>
            </div>
            <div className="list-stack">
              {(selectedScene?.normalized.rooms ?? []).map((room) => (
                <article key={room.id} className="list-row">
                  <div>
                    <strong>{room.id}</strong>
                    <p>{selectedScene ? roomSubtitle(room, selectedScene.dataset) : ""}</p>
                  </div>
                </article>
              ))}
            </div>
          </section>

          <section className="info-card">
            <div className="section-title">
              <Boxes size={16} />
              <h3>Object Finder</h3>
            </div>
            <div className="object-finder">
              <label className="object-finder-search">
                <span>Search by type, label, or id</span>
                <input
                  type="search"
                  value={objectFinderQuery}
                  onChange={(event) => setObjectFinderQuery(event.target.value)}
                  placeholder="chair / bowl / object id"
                />
              </label>

              {objectFinderGroups.length > 0 ? (
                <div className="list-stack compact">
                  {objectFinderGroups.map((group) => {
                    const isExpanded =
                      objectFinderQuery.trim().length > 0 ||
                      group.items.length === 1 ||
                      Boolean(expandedObjectTypes[group.type]);

                    return (
                      <div key={group.type} className="object-type-group">
                        <button
                          type="button"
                          className={`type-row type-row-button ${isExpanded ? "is-expanded" : ""}`}
                          onClick={() => {
                            if (group.items.length === 1) {
                              setSelectedObjectId(group.items[0].id);
                              return;
                            }
                            handleObjectTypeToggle(group.type);
                          }}
                        >
                          <span>{group.type}</span>
                          <strong>{group.items.length}</strong>
                        </button>

                        {isExpanded ? (
                          <div className="object-instance-list">
                            {group.items.map((item, index) => {
                              const fallbackLabel =
                                group.items.length > 1 ? `${group.type} #${index + 1}` : group.type;
                              const displayLabel = item.label === group.type ? fallbackLabel : item.label;
                              return (
                                <button
                                  key={item.id}
                                  type="button"
                                  className={`object-instance-row ${
                                    selectedObjectId === item.id ? "is-selected" : ""
                                  }`}
                                  onClick={() => setSelectedObjectId(item.id)}
                                >
                                  <span>{displayLabel}</span>
                                  <strong>{item.id}</strong>
                                </button>
                              );
                            })}
                          </div>
                        ) : null}
                      </div>
                    );
                  })}
                </div>
              ) : (
                <p className="long-copy">
                  {objectFinderQuery.trim()
                    ? "No matching renderable objects."
                    : "This scene does not expose selectable renderable objects yet."}
                </p>
              )}
            </div>
          </section>

          <section className="info-card">
            <div className="section-title">
              <ImageIcon size={16} />
              <h3>Preview Images</h3>
            </div>
            <div className="image-grid">
              {previewImages.slice(0, 6).map((url) => (
                <a key={url} href={url} target="_blank" rel="noreferrer" className="image-thumb">
                  <img src={url} alt="" loading="lazy" />
                </a>
              ))}
            </div>
          </section>

          <section className="info-card">
            <div className="section-title">
              <Database size={16} />
              <h3>Source Assets</h3>
            </div>
            <div className="list-stack compact">
              {assetEntries.map(([key, value]) => (
                <div key={key} className="type-row asset-row">
                  <span>{key}</span>
                  <strong>{String(value)}</strong>
                </div>
              ))}
            </div>
          </section>
        </aside>
      </main>
    </div>
  );
}
