export const REPO_ASSET_PREFIX = "/repo-assets";

const LEGACY_HSM_HSSD_SEGMENT = "/hssd-models/";

function encodePathSegments(relativePath: string): string {
  return relativePath
    .replace(/^\/+/, "")
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
}

export function normalizeRepoAssetPath(path: string | null | undefined): string | null {
  if (!path) {
    return null;
  }

  const normalized = path.trim().replace(/\\/g, "/");
  if (!normalized) {
    return null;
  }

  if (
    normalized.startsWith("assets/") ||
    normalized.startsWith("./assets/") ||
    normalized.startsWith("../assets/")
  ) {
    return normalized.replace(/^\.\//, "");
  }

  const legacyHsmIndex = normalized.lastIndexOf(LEGACY_HSM_HSSD_SEGMENT);
  if (legacyHsmIndex >= 0) {
    const suffix = normalized.slice(legacyHsmIndex + LEGACY_HSM_HSSD_SEGMENT.length);
    return `assets/hsm/hssd-models/${suffix}`.replace(/\/{2,}/g, "/");
  }

  const embeddedAssetsIndex = normalized.lastIndexOf("/assets/");
  if (embeddedAssetsIndex >= 0) {
    return normalized.slice(embeddedAssetsIndex + 1);
  }

  return normalized.replace(/^\/+/, "");
}

export function toRepoAssetUrl(relativePath: string | null | undefined): string | null {
  const normalizedPath = normalizeRepoAssetPath(relativePath);
  if (!normalizedPath) {
    return null;
  }
  return `${REPO_ASSET_PREFIX}/${encodePathSegments(normalizedPath)}`;
}

export async function fetchRepoJson<T>(relativePath: string): Promise<T> {
  const url = toRepoAssetUrl(relativePath);
  if (!url) {
    throw new Error(`Cannot resolve repo asset URL for ${relativePath}`);
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${relativePath}: ${response.status}`);
  }

  return (await response.json()) as T;
}

export async function fetchRepoText(relativePath: string): Promise<string> {
  const url = toRepoAssetUrl(relativePath);
  if (!url) {
    throw new Error(`Cannot resolve repo asset URL for ${relativePath}`);
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${relativePath}: ${response.status}`);
  }

  return await response.text();
}
