"""Task 1 settings: loads task1.yaml and defines names/paths shared by all Task 1 code."""
from pathlib import Path

from common.io import load_json, load_yaml
from common.paths import REPO_ROOT, storage_dir

TASK = "task1"
TASK_DIR = REPO_ROOT / "task1"
CONFIG_PATH = TASK_DIR / "configs" / "task1.yaml"
SPLIT_PATH = TASK_DIR / "data" / "stl10_split_seed6304.json"   # committed image identifiers
RESULTS = TASK_DIR / "results"                                  # committed JSON/CSV results

BACKBONES = ["resnet50", "vit_b16", "clip_vitb32"]
HEAD_OF = {"resnet50": "resnet50", "vit_b16": "vit_b16", "clip_vitb32": "clip_linear"}
MODELS = ["resnet50", "vit_b16", "clip_linear", "clip_zeroshot"]   # the 4 evaluated predictors


def cfg() -> dict:
    return load_yaml(CONFIG_PATH)


def split() -> dict:
    return load_json(SPLIT_PATH)


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# Large artefacts -> Google Drive (PA1_STORAGE), never committed.
def image_dir() -> Path:
    return _mk(storage_dir("cache", TASK) / "images")


def feature_dir(backbone: str) -> Path:
    return _mk(storage_dir("cache", TASK) / "features" / backbone)


def ckpt_dir() -> Path:
    return storage_dir("checkpoints", TASK)


# ---- canonical condition names (Part 2) ----
def color_condition(c: dict) -> str:
    cc = c["color"]
    return f"test_hue{int(cc['hue_degrees'])}" if cc["additional"] == "hue_rotation" else "test_palette"


def patch_condition(c: dict) -> str:
    return f"test_patch{c['patch_shuffle']['grid']}x{c['patch_shuffle']['grid']}"


def shift_condition(delta: int, direction: str) -> str:
    return f"test_shift{delta}_{direction}"


CUE_CANDIDATES = "cue_candidates"      # all generated conflicts (image store)
CUE_ACCEPTED = "test_cue_conflict"     # the accepted subset used for evaluation
CUE_META = TASK_DIR / "data" / "cue_conflict_meta.json"       # content/style of each candidate
CUE_REVIEW = TASK_DIR / "data" / "cue_conflict_review.json"   # rejection rule + rejected rows
