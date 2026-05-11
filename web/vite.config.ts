import type { ServerResponse } from "node:http";
import { createReadStream, existsSync, statSync } from "node:fs";
import path from "node:path";
import type { Connect, PreviewServer, ViteDevServer } from "vite";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { lookup as lookupMime } from "mime-types";

const REPO_ROOT = path.resolve(__dirname, "..");
const REPO_ASSET_PREFIX = "/repo-assets/";

function serveRepoAsset(
  req: Connect.IncomingMessage,
  res: ServerResponse,
  next: Connect.NextFunction,
) {
  const requestUrl = req.url?.split("?")[0] ?? "/";
  if (!requestUrl.startsWith(REPO_ASSET_PREFIX)) {
    next();
    return;
  }

  const relativePath = decodeURIComponent(requestUrl.slice(REPO_ASSET_PREFIX.length));
  const filePath = path.resolve(REPO_ROOT, relativePath);
  const repoRootWithSeparator = `${REPO_ROOT}${path.sep}`;
  if (filePath !== REPO_ROOT && !filePath.startsWith(repoRootWithSeparator)) {
    res.statusCode = 403;
    res.end("Forbidden");
    return;
  }

  if (!existsSync(filePath) || statSync(filePath).isDirectory()) {
    res.statusCode = 404;
    res.end("Not Found");
    return;
  }

  const mimeType = lookupMime(filePath);
  if (mimeType) {
    res.setHeader("Content-Type", mimeType);
  }
  res.setHeader("Cache-Control", "no-cache");
  createReadStream(filePath).pipe(res);
}

function repoAssetsPlugin() {
  return {
    name: "sceneviewer-repo-assets",
    configureServer(server: ViteDevServer) {
      server.middlewares.use(serveRepoAsset);
    },
    configurePreviewServer(server: PreviewServer) {
      server.middlewares.use(serveRepoAsset);
    },
  };
}

export default defineConfig({
  plugins: [react(), repoAssetsPlugin()],
  server: {
    port: 5174,
    fs: {
      allow: [REPO_ROOT],
    },
  },
  preview: {
    port: 4174,
  },
});
