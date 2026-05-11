from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
ASSETS_ROOT = REPO_ROOT / "assets"
PREPROCESSED_ROOT = ASSETS_ROOT / "preprocessed"


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


DATASETS: dict[str, DatasetSpec] = {
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
