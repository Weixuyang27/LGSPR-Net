"""Shared dataset definitions for the four comparison cases."""
from pathlib import Path

from . import data as hsi


CASE_NAMES = ("case1", "case2", "case3", "case4")
TRANSFER_NAMES = {
    "case1": "pu_to_pc",
    "case2": "ip_to_sa",
    "case3": "ch_to_pc",
    "case4": "ch_to_sa",
}


def transfer_name(case):
    try:
        return TRANSFER_NAMES[case]
    except KeyError as exc:
        raise ValueError(f"Unknown case: {case}") from exc


def case_spec(case):
    specs = {
        "case1": {
            "source": "Pavia University", "target": "Pavia Centre",
            "target_classes": 9, "episode_n_way": 9, "query": 19,
        },
        "case2": {
            "source": "Indian Pines", "target": "Salinas",
            "target_classes": 16, "episode_n_way": 8, "query": 4,
        },
        "case3": {
            "source": "Chikusei", "target": "Pavia Centre",
            "target_classes": 9, "episode_n_way": 9, "query": 19,
        },
        "case4": {
            "source": "Chikusei", "target": "Salinas",
            "target_classes": 16, "episode_n_way": 8, "query": 4,
        },
    }
    try:
        return specs[case]
    except KeyError as exc:
        raise ValueError(f"Unknown case: {case}") from exc


def _find_chikusei_files(data_root):
    directory = Path(data_root) / "Chikusei_MATLAB"
    image = directory / "HyperspecVNIR_Chikusei_20140729.mat"
    gt = directory / "HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat"
    mat_files = list(directory.glob("*.mat")) if directory.is_dir() else []
    if not image.exists():
        candidates = [p for p in mat_files if "gt" not in p.stem.lower() and "ground" not in p.stem.lower()]
        if len(candidates) == 1:
            image = candidates[0]
    if not gt.exists():
        candidates = [p for p in mat_files if "gt" in p.stem.lower() or "ground" in p.stem.lower()]
        if len(candidates) == 1:
            gt = candidates[0]
    for path in (image, gt):
        if not path.exists():
            raise FileNotFoundError(f"Missing required Chikusei file: {path}")
    return image, gt


def load_target(data_root, case, n_bands):
    """Load the target scene, using full-span resampling for cross-sensor cases."""
    root = Path(data_root)
    if case in ("case1", "case3"):
        cube, gt = hsi.load_pavia_centre(
            root / "paviaC/Pavia.mat", root / "paviaC/Pavia_gt.mat",
            None if case == "case3" else n_bands,
        )
    elif case in ("case2", "case4"):
        cube, gt = hsi.load_salinas(
            root / "salinas/Salinas_corrected.mat", root / "salinas/Salinas_gt.mat",
            None if case == "case4" else n_bands,
        )
    else:
        raise ValueError(f"Unknown case: {case}")
    if case in ("case3", "case4"):
        cube = hsi.resample_bands(cube, n_bands)
    expected = case_spec(case)["target_classes"]
    if int(gt.max()) != expected:
        raise ValueError(f"Expected {expected} target classes, got {int(gt.max())}")
    return cube, gt, expected


def load_case(data_root, case, n_bands):
    """Load source and target for a comparison method at a shared band width."""
    root = Path(data_root)
    if case == "case1":
        source_cube, source_gt = hsi.load_pavia_university(
            root / "paviaU/PaviaU.mat", root / "paviaU/PaviaU_gt.mat", n_bands)
    elif case == "case2":
        source_cube, source_gt = hsi.load_indian_pines(
            root / "indianPines/Indian_pines_corrected.mat",
            root / "indianPines/Indian_pines_gt.mat", n_bands)
    elif case in ("case3", "case4"):
        image, gt = _find_chikusei_files(root)
        source_cube, source_gt = hsi.load_chikusei(image, gt, n_bands)
    else:
        raise ValueError(f"Unknown case: {case}")
    target_cube, target_gt, classes = load_target(root, case, n_bands)
    return source_cube, source_gt, target_cube, target_gt, classes
