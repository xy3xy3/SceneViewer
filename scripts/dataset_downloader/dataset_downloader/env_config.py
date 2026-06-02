"""Load optional configuration from ``.env`` and ``config.yml`` at the repo root.

Resolution order (later values override earlier ones):
1. Hard-coded defaults in ``config.py``
2. ``config.yml`` at the repo root (YAML mapping)
3. ``.env`` at the repo root (``KEY=VALUE`` lines, ``#`` comments)
4. Real environment variables (``os.environ``)
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from .config import REPO_ROOT

_ENV_FILE = REPO_ROOT / ".env"
_CONFIG_YML = REPO_ROOT / "config.yml"
_CONFIG_YAML = REPO_ROOT / "config.yaml"

_loaded = False


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse a simple ``.env`` file (no shell expansion, no multiline)."""
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            result[key] = value
    return result


def _load_config_yml() -> dict[str, str]:
    """Load a flat key-value mapping from ``config.yml`` / ``config.yaml``."""
    path = _CONFIG_YML if _CONFIG_YML.exists() else _CONFIG_YAML
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if v is not None}


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    # 1. config.yml (lowest priority among overrides)
    for key, value in _load_config_yml().items():
        os.environ.setdefault(key, value)
    # 2. .env file
    for key, value in _parse_env_file(_ENV_FILE).items():
        os.environ.setdefault(key, value)
    # Real os.environ already has highest priority.


def get(key: str, default: str | None = None) -> str | None:
    """Return a config value by key, loading files on first call."""
    _ensure_loaded()
    return os.environ.get(key, default)


def get_path(key: str, default: Path | None = None) -> Path | None:
    """Return a config value as an absolute ``Path``, or *default*."""
    raw = get(key)
    if raw is None:
        return default
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()
