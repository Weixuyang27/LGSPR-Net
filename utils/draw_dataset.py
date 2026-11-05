"""Paper-style dataset overviews and classification-map comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
import scipy.io as sio


ROOT = Path(__file__).resolve().parents[1]


def _configure_times_new_roman():
    for font_path in (Path("/mnt/c/Windows/Fonts/times.ttf"), Path("C:/Windows/Fonts/times.ttf")):
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            break
    plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix"})


_configure_times_new_roman()

ORIGINAL_CLASS_COLORS = np.asarray([
    [216, 27, 96], [30, 136, 229], [255, 193, 7], [0, 137, 123],
    [126, 87, 194], [67, 160, 71], [251, 140, 0], [84, 110, 122],
    [229, 57, 53], [0, 188, 212], [156, 204, 101], [171, 71, 188],
    [255, 112, 67], [38, 198, 218], [205, 220, 57], [121, 85, 72],
], dtype=np.float32) / 255.0

PALETTE_ORDERS = {
    9: [1, 7, 4, 0, 8, 6, 2, 5, 3],
    16: [1, 7, 13, 9, 3, 5, 10, 14, 2, 6, 15, 12, 8, 0, 11, 4],
}

DATASETS = {
    "case1": {
        "name": "Pavia Centre",
        "cube": "paviaC/Pavia.mat",
        "gt": "paviaC/Pavia_gt.mat",
        "cube_key": "pavia",
        "gt_key": "pavia_gt",
        "rgb_bands": (56, 33, 13),
        "rgb_gains": (1.0, 1.0, 1.0),
        "gamma": 0.9,
        "palette_order": [1, 7, 4, 0, 8, 6, 5, 2, 3],
        "classes": [
            "Water", "Trees", "Asphalt", "Self-blocking bricks", "Bitumen",
            "Tiles", "Shadows", "Meadows", "Bare soil",
        ],
    },
    "case2": {
        "name": "Salinas",
        "cube": "salinas/Salinas_corrected.mat",
        "gt": "salinas/Salinas_gt.mat",
        "cube_key": "salinas_corrected",
        "gt_key": "salinas_gt",
        "rgb_bands": (57, 27, 17),
        "rgb_gains": (0.92, 0.40, 0.32),
        "gamma": 1.05,
        "classes": [
            "Broccoli green weeds 1", "Broccoli green weeds 2", "Fallow",
            "Fallow rough plow", "Fallow smooth", "Stubble", "Celery",
            "Grapes untrained", "Soil vineyard develop",
            "Corn senesced green weeds", "Lettuce romaine 4 wk",
            "Lettuce romaine 5 wk", "Lettuce romaine 6 wk",
            "Lettuce romaine 7 wk", "Vineyard untrained",
            "Vineyard vertical trellis",
        ],
    },
    "pu": {
        "name": "Pavia University",
        "cube": "paviaU/PaviaU.mat",
        "gt": "paviaU/PaviaU_gt.mat",
        "cube_key": "paviaU",
        "gt_key": "paviaU_gt",
        "rgb_bands": (60, 30, 2),
        "rgb_gains": (1.0, 1.0, 1.0),
        "gamma": 1.10,
        "saturation": 0.48,
        "brightness": 0.68,
        "classes": [
            "Asphalt", "Meadows", "Gravel", "Trees", "Painted metal sheets",
            "Bare soil", "Bitumen", "Self-blocking bricks", "Shadows",
        ],
    },
    "ip": {
        "name": "Indian Pines",
        "cube": "indianPines/Indian_pines_corrected.mat",
        "gt": "indianPines/Indian_pines_gt.mat",
        "cube_key": "indian_pines_corrected",
        "gt_key": "indian_pines_gt",
        "rgb_bands": (50, 27, 17),
        "rgb_gains": (0.92, 0.45, 0.36),
        "gamma": 1.05,
        "classes": [
            "Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture",
            "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats",
            "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat",
            "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers",
        ],
    },
    "chikusei": {
        "name": "Chikusei",
        "cube": "Chikusei_MATLAB/HyperspecVNIR_Chikusei_20140729.mat",
        "gt": "Chikusei_MATLAB/HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat",
        "cube_key": "chikusei",
        "rgb_bands": (57, 37, 22),
        "rgb_gains": (1.0, 0.92, 0.90),
        "gamma": 1.0,
        "saturation": 0.72,
        "brightness": 0.82,
        "embedded_metadata": True,
    },
}

METHODS = (
    ("SSRN", "results/comparisons/{transfer}/ssrn/seed{seed}/predictions.npz", "results/ssrn_comparison_{case}/ssrn_b50_lr002_v1/seed{seed}/predictions.npz"),
    ("RN-FSC", "results/comparisons/{transfer}/rnfsc/seed{seed}/predictions.npz", "results/rnfsc_comparison_{case}/rnfsc_project5_v1/seed{seed}/predictions.npz"),
    ("DFSL+NN", "results/comparisons/{transfer}/dfsl/seed{seed}/predictions_nn.npz", "results/dfsl_comparison_{case}/dfsl_project5_v1/seed{seed}/predictions_nn.npz"),
    ("DFSL+SVM", "results/comparisons/{transfer}/dfsl/seed{seed}/predictions_svm.npz", "results/dfsl_comparison_{case}/dfsl_project5_v1/seed{seed}/predictions_svm.npz"),
    ("Gia-CFSL", "results/comparisons/{transfer}/giacfsl/seed{seed}/predictions.npz", "results/giacfsl_comparison_{case}/giacfsl_project5_v1/seed{seed}/predictions.npz"),
    ("DCFSL", "results/comparisons/{transfer}/dcfsl/seed{seed}/predictions.npz", "results/dcfsl_comparison_{case}/dcfsl_project5_v1/seed{seed}/predictions.npz"),
    ("CMFSL", "results/comparisons/{transfer}/cmfsl/seed{seed}/predictions.npz", "results/cmfsl_comparison_{case}/cmfsl_project5_v1/seed{seed}/predictions.npz"),
)

CROSS_SENSOR_METHODS = (
    ("SSRN", "results/comparisons/{transfer}/ssrn/seed{seed}/predictions.npz", "results/ssrn_comparison_{case}/cross_sensor_cmp_v1/seed{seed}/predictions.npz"),
    ("RN-FSC", "results/comparisons/{transfer}/rnfsc/seed{seed}/predictions.npz", "results/rnfsc_comparison_{case}/cross_sensor_cmp_v1/seed{seed}/predictions.npz"),
    ("DFSL+NN", "results/comparisons/{transfer}/dfsl/seed{seed}/predictions_nn.npz", "results/dfsl_comparison_{case}/cross_sensor_cmp_v1/seed{seed}/predictions_nn.npz"),
    ("DFSL+SVM", "results/comparisons/{transfer}/dfsl/seed{seed}/predictions_svm.npz", "results/dfsl_comparison_{case}/cross_sensor_cmp_v1/seed{seed}/predictions_svm.npz"),
    ("Gia-CFSL", "results/comparisons/{transfer}/giacfsl/seed{seed}/predictions.npz", "results/giacfsl_comparison_{case}/cross_sensor_cmp_v1/seed{seed}/predictions.npz"),
    ("DCFSL", "results/comparisons/{transfer}/dcfsl/seed{seed}/predictions.npz", "results/dcfsl_comparison_{case}/cross_sensor_cmp_v1/seed{seed}/predictions.npz"),
    ("CMFSL", "results/comparisons/{transfer}/cmfsl/seed{seed}/predictions.npz", "results/cmfsl_comparison_{case}/cross_sensor_cmp_v1/seed{seed}/predictions.npz"),
)


def _load_mat(path, key):
    content = sio.loadmat(path)
    if key not in content:
        available = [name for name in content if not name.startswith("__")]
        raise KeyError(f"Missing key {key!r} in {path}; available={available}")
    return np.asarray(content[key])


def _load_chikusei_display(image_path, gt_path, bands):
    import h5py

    content = sio.loadmat(gt_path)
    root = content["GT"]
    gt = np.asarray(root["gt"][0, 0], dtype=np.int64)
    colors = np.asarray(root["classcolor"][0, 0], dtype=np.float32) / 255.0
    names = [str(np.asarray(item).reshape(-1)[0])
             for item in root["classnames"][0, 0].reshape(-1)]
    indices = np.asarray(bands, dtype=np.int64) - 1
    with h5py.File(image_path, "r") as handle:
        dataset = handle["chikusei"]
        if dataset.shape[0] != 128 or dataset.shape[1:] != gt.shape[::-1]:
            raise ValueError(f"Unexpected Chikusei cube/GT shapes: {dataset.shape} and {gt.shape}")
        cube = np.stack(
            [np.asarray(dataset[int(index)], dtype=np.float32).T for index in indices], axis=2)
    return cube, gt, names, colors


def false_color(cube, bands, lower=2.0, upper=98.0, gamma=0.9,
                channel_gains=(1.0, 1.0, 1.0), saturation=1.0, brightness=1.0):
    indices = np.asarray(bands, dtype=np.int64) - 1
    if len(indices) != 3 or indices.min() < 0 or indices.max() >= cube.shape[2]:
        raise ValueError(f"RGB bands {tuple(bands)} are invalid for a {cube.shape[2]}-band cube")
    rgb = cube[:, :, indices].astype(np.float32, copy=True)
    for channel in range(3):
        low, high = np.percentile(rgb[:, :, channel], (lower, upper))
        denominator = max(float(high - low), np.finfo(np.float32).eps)
        rgb[:, :, channel] = np.clip((rgb[:, :, channel] - low) / denominator, 0, 1)
    gains = np.asarray(channel_gains, dtype=np.float32).reshape(1, 1, 3)
    rgb = np.power(np.clip(rgb * gains, 0, 1), gamma)
    luminance = np.sum(
        rgb * np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32),
        axis=2, keepdims=True,
    )
    rgb = luminance + float(saturation) * (rgb - luminance)
    return np.clip(rgb * float(brightness), 0, 1)


def _foreground_colors(class_count, palette_order=None):
    if class_count not in PALETTE_ORDERS:
        raise ValueError(f"No class-color order is defined for {class_count} classes")
    order = palette_order if palette_order is not None else PALETTE_ORDERS[class_count]
    return ORIGINAL_CLASS_COLORS[order]


def palette_for_case(case_name):
    dataset_key = "case1" if case_name in {"case1", "case3"} else "case2"
    config = DATASETS[dataset_key]
    foreground = _foreground_colors(len(config["classes"]), config.get("palette_order"))
    return np.rint(np.vstack((np.zeros((1, 3)), foreground)) * 255.0).astype(np.uint8)


HSI_COLORS = palette_for_case("case2")


def save_label_map(label_map, path, class_count, palette=HSI_COLORS):
    palette = np.asarray(palette, dtype=np.uint8)
    if class_count + 1 > len(palette):
        raise ValueError(f"palette supports at most {len(palette) - 1} classes")
    labels = np.asarray(label_map)
    if labels.ndim != 2:
        raise ValueError(f"expected a 2-D label map, got shape {labels.shape}")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output, palette[np.clip(labels, 0, class_count)])


def draw_overview(cube, gt, class_names, rgb_bands, output, dpi,
                  rgb_gains=(1.0, 1.0, 1.0), gamma=0.9,
                  saturation=1.0, brightness=1.0, palette_order=None,
                  class_colors=None, additional_outputs=()):
    if cube.shape[:2] != gt.shape:
        raise ValueError(f"Cube/GT shape mismatch: {cube.shape[:2]} vs {gt.shape}")
    classes = len(class_names)
    rgb = false_color(
        cube, rgb_bands, gamma=gamma, channel_gains=rgb_gains,
        saturation=saturation, brightness=brightness,
    )
    foreground = (_foreground_colors(classes, palette_order) if class_colors is None
                  else np.asarray(class_colors, dtype=np.float32))
    if foreground.shape != (classes, 3):
        raise ValueError(f"class colors must have shape {(classes, 3)}, got {foreground.shape}")
    colors = np.vstack(([0.0, 0.0, 0.0], foreground))
    cmap = ListedColormap(colors, name="hsi_classes")
    norm = BoundaryNorm(np.arange(-0.5, classes + 1.5), cmap.N)

    aspect = cube.shape[1] / cube.shape[0]
    panel_width = max(3.0, 5.4 * aspect)
    fig = plt.figure(figsize=(2 * panel_width + 4.2, 6.0), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, 0.72), wspace=0.04)
    ax_rgb, ax_gt, ax_legend = [fig.add_subplot(grid[0, index]) for index in range(3)]
    ax_rgb.imshow(rgb, interpolation="nearest")
    ax_gt.imshow(gt, cmap=cmap, norm=norm, interpolation="nearest")
    for axis in (ax_rgb, ax_gt):
        axis.set_xticks([]); axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.8); spine.set_color("0.25")
    ax_rgb.text(0.5, -0.045, "(a)", transform=ax_rgb.transAxes,
                ha="center", va="top", fontsize=18)
    ax_gt.text(0.5, -0.045, "(b)", transform=ax_gt.transAxes,
               ha="center", va="top", fontsize=18)

    ax_legend.axis("off")
    y_positions = np.linspace(0.945, 0.055, classes)
    row_step = 0.89 / max(classes - 1, 1)
    if classes <= 10:
        swatch_width, swatch_height, text_x, legend_fontsize = 0.235, row_step * 0.62, 0.285, 13
    else:
        swatch_width, swatch_height, text_x, legend_fontsize = 0.16, row_step * 0.78, 0.205, 9
    for index, (class_name, y_pos) in enumerate(zip(class_names, y_positions), start=1):
        ax_legend.add_patch(Rectangle(
            (0.02, y_pos - swatch_height / 2), swatch_width, swatch_height,
            transform=ax_legend.transAxes, facecolor=colors[index],
            edgecolor="0.45", linewidth=0.45, clip_on=False,
        ))
        ax_legend.text(text_x, y_pos, class_name, transform=ax_legend.transAxes,
                       ha="left", va="center", fontsize=legend_fontsize)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    for extra_output in additional_outputs:
        extra_output = Path(extra_output)
        extra_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(extra_output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _class_colormap(class_count, palette_order=None):
    colors = np.vstack((np.zeros((1, 3), dtype=np.float32),
                        _foreground_colors(class_count, palette_order)))
    cmap = ListedColormap(colors, name=f"hsi_{class_count}_classes")
    norm = BoundaryNorm(np.arange(-0.5, class_count + 1.5), cmap.N)
    return cmap, norm


def _restore_saved_prediction(path, gt, class_count):
    with np.load(path, allow_pickle=False) as saved:
        required = {"flat_indices", "truth", "prediction", "image_shape"}
        missing = required.difference(saved.files)
        if missing:
            raise KeyError(f"{path} is missing arrays: {sorted(missing)}")
        shape = tuple(int(value) for value in saved["image_shape"].reshape(-1))
        indices = saved["flat_indices"].astype(np.int64, copy=False).reshape(-1)
        truth = saved["truth"].astype(np.int64, copy=False).reshape(-1)
        prediction = saved["prediction"].astype(np.int64, copy=False).reshape(-1)
    if shape != gt.shape or not np.array_equal(gt.reshape(-1)[indices], truth):
        raise ValueError(f"{path}: prediction metadata does not match the ground truth")
    if len(prediction) and (prediction.min() < 1 or prediction.max() > class_count):
        raise ValueError(f"{path}: prediction labels must be in 1..{class_count}")
    label_map = np.zeros(gt.shape, dtype=np.uint16)
    label_map.reshape(-1)[indices] = prediction
    label_map[(gt > 0) & (label_map == 0)] = gt[(gt > 0) & (label_map == 0)]
    return label_map


def _load_ours(path, gt, class_count):
    if path.suffix.lower() == ".npz":
        saved = np.load(path, allow_pickle=False)
        required = {"flat_test_indices", "y_true", "y_pred"}
        if not required.issubset(saved.files):
            raise ValueError(f"{path}: missing fields {sorted(required - set(saved.files))}")
        indices = np.asarray(saved["flat_test_indices"], dtype=np.int64).reshape(-1)
        truth = np.asarray(saved["y_true"], dtype=np.int64).reshape(-1)
        prediction = np.asarray(saved["y_pred"], dtype=np.int64).reshape(-1)
        if not (len(indices) == len(truth) == len(prediction)):
            raise ValueError(f"{path}: prediction arrays have inconsistent lengths")
        if len(indices) and (indices.min() < 0 or indices.max() >= gt.size):
            raise ValueError(f"{path}: test indices are outside the target image")
        if not np.array_equal(gt.reshape(-1)[indices], truth + 1):
            raise ValueError(f"{path}: prediction labels do not match the ground truth")
        if len(prediction) and (prediction.min() < 0 or prediction.max() >= class_count):
            raise ValueError(f"{path}: predicted labels must be in 0..{class_count - 1}")
        label_map = np.zeros(gt.shape, dtype=np.uint16)
        label_map.reshape(-1)[indices] = prediction + 1
        label_map[(gt > 0) & (label_map == 0)] = gt[(gt > 0) & (label_map == 0)]
        return label_map
    label_map = np.asarray(np.load(path, allow_pickle=False))
    if label_map.shape != gt.shape or label_map.min() < 0 or label_map.max() > class_count:
        raise ValueError(f"{path}: invalid LGSPR-Net prediction map")
    return label_map.astype(np.uint16, copy=False)


def _panel_label(index, name):
    letter = chr(ord("a") + index) if index < 26 else str(index + 1)
    return f"({letter}) {name}"


def draw_comparison(case, seed, project_root, data_root, out_dir, ours_path,
                    dpi, excluded_methods=()):
    transfer_name = {"case1": "pu_to_pc", "case2": "ip_to_sa",
                     "case3": "ch_to_pc", "case4": "ch_to_sa"}[case]
    dataset_key = "case1" if case == "case3" else "case2" if case == "case4" else case
    config = DATASETS[dataset_key]
    cube = _load_mat(data_root / config["cube"], config["cube_key"])
    gt = _load_mat(data_root / config["gt"], config["gt_key"]).astype(np.uint16)
    class_count = len(config["classes"])
    rgb = false_color(
        cube, config["rgb_bands"], gamma=config.get("gamma", 0.9),
        channel_gains=config.get("rgb_gains", (1.0, 1.0, 1.0)),
        saturation=config.get("saturation", 1.0), brightness=config.get("brightness", 1.0),
    )
    panels = [("False-Color Image", rgb, True), ("Ground truth", gt, False)]
    method_specs = CROSS_SENSOR_METHODS if case in ("case3", "case4") else METHODS
    for name, *templates in method_specs:
        if name.casefold() in excluded_methods:
            continue
        candidates = [project_root / template.format(
            case=case, transfer=transfer_name, seed=seed) for template in templates]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            print(f"[skip] {transfer_name} {name}: {candidates[0]}")
            continue
        panels.append((name, _restore_saved_prediction(path, gt, class_count), False))
        print(f"[load] {transfer_name} {name}: {path}")
    if ours_path is not None and ours_path.exists():
        panels.append(("Ours", _load_ours(ours_path, gt, class_count), False))
        print(f"[load] {transfer_name} Ours: {ours_path}")

    columns = len(panels)
    scene_aspect = gt.shape[1] / gt.shape[0]
    figure_width = 7.16
    figure_height = (figure_width / columns) / scene_aspect + 0.24
    fig, axes = plt.subplots(1, columns, figsize=(figure_width, figure_height), squeeze=False)
    cmap, norm = _class_colormap(class_count, config.get("palette_order"))
    for index, (axis, (name, image, is_rgb)) in enumerate(zip(axes.flat, panels)):
        axis.imshow(image, interpolation="nearest" if is_rgb else "nearest",
                    **({} if is_rgb else {"cmap": cmap, "norm": norm}))
        axis.set_xticks([]); axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.text(0.5, -0.035, _panel_label(index, name), transform=axis.transAxes,
                  ha="center", va="top", fontsize=5.4)
    fig.subplots_adjust(left=0.004, right=0.996, top=0.995, bottom=0.09, wspace=0.035)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Preserve native scene detail in rasterized PDF panels.
    panel_width = figure_width * 0.992 / columns
    panel_height = figure_height * (0.995 - 0.09)
    native_dpi = int(np.ceil(max(gt.shape[1] / panel_width, gt.shape[0] / panel_height)))
    pdf_dpi = max(dpi, native_dpi)
    for suffix, kwargs in ((".png", {"dpi": dpi}), (".pdf", {"dpi": pdf_dpi})):
        path = out_dir / f"{transfer_name}_comparison_seed{seed}{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.015, facecolor="white", **kwargs)
        print(f"[saved] {path}")
    plt.close(fig)


def _dataset_cases(selection):
    names = {"pc": "case1", "salinas": "case2", "pu": "pu", "ip": "ip",
             "chikusei": "chikusei"}
    if selection == "all":
        return tuple(names.values())
    return (names[selection],)


def _comparison_cases(selection):
    transfers = {"pu_to_pc": "case1", "ip_to_sa": "case2",
                 "ch_to_pc": "case3", "ch_to_sa": "case4"}
    if selection == "same_sensor":
        return (transfers["pu_to_pc"], transfers["ip_to_sa"])
    if selection == "cross_sensor":
        return (transfers["ch_to_pc"], transfers["ch_to_sa"])
    if selection == "all":
        return tuple(transfers.values())
    return (transfers[selection],)


def _check_dataset_files(data_root, cases):
    missing = []
    for case in cases:
        config = DATASETS["case1" if case == "case3" else "case2" if case == "case4" else case]
        for field in ("cube", "gt"):
            path = data_root / config[field]
            if not path.is_file() and path not in missing:
                missing.append(path)
    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(
            "Missing dataset files:\n"
            f"{lines}\n"
            "Place the public datasets under --data_root using the directory "
            "layout documented in README.md, or pass --data_root to an existing complete dataset directory."
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    dataset = subparsers.add_parser("dataset")
    dataset.add_argument("--dataset", choices=("pc", "salinas", "pu", "ip", "chikusei", "all"), default="all")
    dataset.add_argument("--data_root", type=Path, default=ROOT / "Datasets")
    dataset.add_argument("--out_dir", type=Path, default=ROOT / "paper" / "figures")
    dataset.add_argument("--rgb_bands", type=int, nargs=3)
    dataset.add_argument("--dpi", type=int, default=400)

    comparison = subparsers.add_parser("comparison")
    comparison.add_argument(
        "--transfer",
        choices=("pu_to_pc", "ip_to_sa", "ch_to_pc", "ch_to_sa",
                 "same_sensor", "cross_sensor", "all"),
        default="all",
    )
    comparison.add_argument("--seed", type=int, default=1337)
    comparison.add_argument("--project_root", type=Path, default=ROOT)
    comparison.add_argument("--data_root", type=Path, default=ROOT / "Datasets")
    comparison.add_argument("--out_dir", type=Path, default=ROOT / "paper" / "figures")
    transfer_cases = {"pu_to_pc": "case1", "ip_to_sa": "case2",
                      "ch_to_pc": "case3", "ch_to_sa": "case4"}
    for transfer, case in transfer_cases.items():
        comparison.add_argument(
            f"--ours_{transfer}", type=Path,
            default=ROOT / "classification_results" / case / f"{case}_prediction_map.npy",
        )
    comparison.add_argument("--no_ours", action="store_true")
    comparison.add_argument("--exclude", nargs="*", default=(), metavar="METHOD")
    comparison.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    if args.command == "dataset" and args.rgb_bands and args.dataset == "all":
        parser.error("--rgb_bands requires one individual dataset")
    return args


def main():
    args = parse_args()
    if args.command == "dataset":
        cases = _dataset_cases(args.dataset)
        _check_dataset_files(args.data_root, cases)
        stems = {"case1": "dataset_pc", "case2": "dataset_salinas", "pu": "dataset_pu",
                 "ip": "dataset_ip", "chikusei": "dataset_chikusei"}
        for case in cases:
            config = DATASETS[case]
            bands = tuple(args.rgb_bands) if args.rgb_bands else config["rgb_bands"]
            if config.get("embedded_metadata"):
                cube, gt, names, colors = _load_chikusei_display(
                    args.data_root / config["cube"], args.data_root / config["gt"], bands)
                display_bands = (1, 2, 3)
            else:
                cube = _load_mat(args.data_root / config["cube"], config["cube_key"])
                gt = _load_mat(args.data_root / config["gt"], config["gt_key"]).astype(np.int64)
                names, colors, display_bands = config["classes"], None, bands
            stem = args.out_dir / stems[case]
            draw_overview(
                cube, gt, names, display_bands, stem.with_suffix(".png"), args.dpi,
                rgb_gains=config["rgb_gains"], gamma=config["gamma"],
                saturation=config.get("saturation", 1.0),
                brightness=config.get("brightness", 1.0),
                palette_order=config.get("palette_order"), class_colors=colors,
                additional_outputs=(stem.with_suffix(".pdf"),),
            )
            print(f"[saved] {stem}.png/.pdf | RGB bands={bands}")
        return

    cases = _comparison_cases(args.transfer)
    _check_dataset_files(args.data_root, cases)
    excluded = {name.casefold() for name in args.exclude}
    case_transfers = {"case1": "pu_to_pc", "case2": "ip_to_sa",
                      "case3": "ch_to_pc", "case4": "ch_to_sa"}
    for case in cases:
        ours = None if args.no_ours else getattr(args, f"ours_{case_transfers[case]}").resolve()
        draw_comparison(case, args.seed, args.project_root.resolve(), args.data_root.resolve(),
                        args.out_dir.resolve(), ours, args.dpi, excluded)


if __name__ == "__main__":
    main()
