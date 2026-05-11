export const REPO_ASSET_PREFIX = "/repo-assets";

function encodePathSegments(relativePath: string): string {
  return relativePath
    .replace(/^\/+/, "")
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
}

export function toRepoAssetUrl(relativePath: string | null | undefined): string | null {
  if (!relativePath) {
    return null;
  }
  return `${REPO_ASSET_PREFIX}/${encodePathSegments(relativePath)}`;
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
