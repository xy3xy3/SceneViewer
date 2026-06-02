from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ASSETS_ROOT = REPO_ROOT / "assets"
PREPROCESSED_ROOT = ASSETS_ROOT / "preprocessed"
RENDERABLE_ROOT = ASSETS_ROOT / "renderable"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    repo_id: str
    archive_suffix: str
    archive_prefix: str | None
    destination_root: Path
    default_subsets: tuple[str, ...] = ()
    recommended_subsets: tuple[str, ...] = ()
    license_note: str | None = None
    supports_remote_download: bool = True


DATASETS: dict[str, DatasetSpec] = {
    "hssd": DatasetSpec(
        key="hssd",
        repo_id="local/HSSD",
        archive_suffix="",
        archive_prefix=None,
        destination_root=ASSETS_ROOT / "hssd",
        license_note=(
            "HSSD stage GLBs are imported from a local checkout. Use "
            "`dataset-downloader import-hssd-local` before preprocessing."
        ),
        supports_remote_download=False,
    ),
    "hsm": DatasetSpec(
        key="hsm",
        repo_id="3dlg-hcvc/hsm",
        archive_suffix=".json",
        archive_prefix="generated_scenes/",
        destination_root=ASSETS_ROOT / "hsm",
        license_note=(
            "HSM scene JSONs are gated on Hugging Face, and renderable previews also require "
            "local HSSD model GLBs. Set SCENEVIEWER_HSSD_ROOT in .env or config.yml to point "
            "to an existing HSSD checkout, or place them under assets/hsm/hssd-models/."
        ),
    ),
    "sage": DatasetSpec(
        key="sage",
        repo_id="nvidia/SAGE-10k",
        archive_suffix=".zip",
        archive_prefix="scenes/",
        destination_root=ASSETS_ROOT / "sage",
    ),
    "scenesmith": DatasetSpec(
        key="scenesmith",
        repo_id="nepfaff/scenesmith-example-scenes",
        archive_suffix=".tar",
        archive_prefix=None,
        destination_root=ASSETS_ROOT / "scenesmith",
        default_subsets=("Room", "House"),
        recommended_subsets=("Room", "House"),
        license_note=(
            "The upstream dataset's NotGenerated subset uses HSSD assets under "
            "CC BY-NC 4.0 and is best kept opt-in."
        ),
    ),
    "sceneweaver": DatasetSpec(
        key="sceneweaver",
        repo_id="local/SceneWeaver",
        archive_suffix="",
        archive_prefix=None,
        destination_root=ASSETS_ROOT / "sceneweaver",
        license_note=(
            "SceneWeaver uses local experiment outputs. Import them with "
            "`dataset-downloader import-sceneweaver-local` before preprocessing."
        ),
        supports_remote_download=False,
    ),
    "3dfront": DatasetSpec(
        key="3dfront",
        repo_id="manual/3D-FRONT",
        archive_suffix=".zip",
        archive_prefix=None,
        destination_root=ASSETS_ROOT / "3dfront",
        license_note=(
            "3D-FRONT is a manual-download dataset. Place 3D-FRONT.zip, "
            "3D-FUTURE-model.zip, and 3D-FRONT-texture.zip in assets/ before preprocessing."
        ),
        supports_remote_download=False,
    ),
}


SCENESMITH_ALL_SUBSETS = (
    "Room",
    "House",
    "NoCritic",
    "NotGenerated",
    "NoAssetValidation",
    "NoSpecializedTools",
    "NoObserveScene",
    "NoAgentMemory",
)
