"""Generate the sensitivity figures reported in the paper."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
CASES = (
    ("case1", "PU to PC", "PU to PC"),
    ("case2", "IP to SA", "IP to SA"),
    ("case3", "CH to PC", "CH to PC"),
    ("case4", "CH to SA", "CH to SA"),
)
TRANSFERS = {"pu_to_pc": "case1", "ip_to_sa": "case2",
             "ch_to_pc": "case3", "ch_to_sa": "case4"}
TRANSFER_NAMES = {case: transfer for transfer, case in TRANSFERS.items()}
CANONICAL = {
    "case1": ROOT / "results/factorial_ablation_case1/case1_fsr_abl8_v1_seed1337.json",
    "case2": ROOT / "results/factorial_ablation_case2/case2_fsr_abl8_v1_seed1337.json",
    "case3": ROOT / "results/factorial_ablation_case3/case3_fsr_cross_sensor_v1_seed1337.json",
    "case4": ROOT / "results/factorial_ablation_case4/case4_fsr_cross_sensor_v1_seed1337.json",
}
PATCH_CSV = {
    case: ROOT / f"results/patch_size_{case}/summary_patch_v1_seed1337.csv"
    for case, _, _ in CASES
}
PATCH_CSV["case1"] = ROOT / "results/patch_size_case1/summary_patch_v1_seed1337_partial.csv"
LAMBDA_CSV = {
    "case1": ROOT / "results/conditional_domain_sensitivity_case1/summary_condlam_v1_seed1337.csv",
    "case2": ROOT / "results/conditional_domain_sensitivity_case2/summary_condlam_v1_seed1337.csv",
    "case3": ROOT / "results/conditional_domain_sensitivity_case3/summary_condlam_cross_v1_seed1337.csv",
    "case4": ROOT / "results/conditional_domain_sensitivity_case4/summary_condlam_cross_v1_seed1337.csv",
}
AMPLITUDE_CSV = {
    "case1": ROOT / "results/perturbation_amplitude_sensitivity_case1/summary_amp_v1_seed1337.csv",
    "case2": ROOT / "results/perturbation_amplitude_sensitivity_case2/summary_amp_v1_seed1337.csv",
    "case3": ROOT / "results/perturbation_amplitude_sensitivity_case3/summary_amp_cross_v1_seed1337.csv",
    "case4": ROOT / "results/perturbation_amplitude_sensitivity_case4/summary_amp_cross_v1_seed1337.csv",
}
METRICS = (
    ("oa", "OA", "#28599A", "o"),
    ("aa", "AA", "#C56527", "s"),
    ("kappa", r"$\kappa \times 100$", "#72539A", "^"),
)
CASE_TITLES = {case: title for case, title, _ in CASES}
EXPECTED_PARAMETERS = {
    "patch": (5.0, 7.0, 9.0, 11.0, 13.0),
    "lambda": (0.0, 0.001, 0.005, 0.01, 0.05, 0.1),
    "amplitude": (0.25, 0.5, 1.0, 1.5, 2.0),
}


def read_canonical(case: str) -> dict[str, float]:
    with CANONICAL[case].open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return {"oa": float(raw["oa"]), "aa": float(raw["aa"]),
            "kappa": 100.0 * float(raw["kappa"])}


def read_sweep(path: Path, parameter: str) -> list[dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return [{
        "parameter": float(row[parameter]),
        "oa": float(row["oa"]),
        "aa": float(row["aa"]),
        "kappa": 100.0 * float(row["kappa"]),
        "status": row.get("status", "complete"),
    } for row in rows]


def canonicalize(rows: list[dict[str, float | str]], value: float, case: str) -> None:
    canonical = read_canonical(case)
    matches = [row for row in rows if math.isclose(
        float(row["parameter"]), value, rel_tol=0.0, abs_tol=1e-12)]
    if len(matches) != 1:
        raise ValueError(f"{case}: expected one adopted row for {value}")
    matches[0].update(canonical)


def save(fig: plt.Figure, stem: str, output_dir: Path = OUT) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".png", {"dpi": 600}), (".pdf", {})):
        path = output_dir / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.045,
                    facecolor="white", **kwargs)
        print(f"[saved] {path}")
    plt.close(fig)


def _amplitude_multiplier(record: dict) -> float:
    ratios = (
        float(record["perturb_gain"]) / 0.05,
        float(record["perturb_spectral_tilt"]) / 0.05,
        float(record["perturb_offset"]) / 0.02,
    )
    if max(ratios) - min(ratios) > 1e-6:
        raise ValueError("the three SPS amplitudes must use one common multiplier")
    return float(np.mean(ratios))


def read_json_sweep(input_dir: Path, kind: str, case: str) -> list[dict[str, float]]:
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(
            f"--input_dir must be a directory containing multiple training-result JSON files; "
            f"received file: {input_dir}"
        )
    grouped: dict[float, list[dict]] = {}
    matched_files = []
    for path in sorted(input_dir.rglob("*.json")):
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        if not isinstance(record, dict):
            continue
        record_case = record.get("case")
        if record_case is None:
            record_case = next((name for name, _, _ in CASES if name in path.name), None)
        if record_case != case or not all(key in record for key in ("oa", "aa", "kappa")):
            continue
        try:
            if kind == "amplitude":
                parameter = _amplitude_multiplier(record)
            elif kind == "lambda":
                parameter = float(record["lambda_cd"] if "lambda_cd" in record
                                  else record["lambda_conditional_domain"])
            else:
                parameter = float(record["patch_size"])
        except KeyError:
            continue
        parameter = round(parameter, 12)
        grouped.setdefault(parameter, []).append(record)
        matched_files.append(path)
    if len(grouped) < 2:
        raise ValueError(
            f"{input_dir} contains {len(grouped)} distinct {kind} setting(s) for "
            f"{TRANSFER_NAMES[case]}; "
            "at least two are required"
        )
    rows = []
    for parameter, records in sorted(grouped.items()):
        kappa = np.asarray([float(item["kappa"]) for item in records], dtype=float)
        if np.max(np.abs(kappa)) <= 1.5:
            kappa *= 100.0
        rows.append({
            "parameter": parameter,
            "oa": float(np.mean([float(item["oa"]) for item in records])),
            "aa": float(np.mean([float(item["aa"]) for item in records])),
            "kappa": float(np.mean(kappa)),
        })
    print(f"[input] {len(matched_files)} JSON files, {len(rows)} settings for "
          f"{TRANSFER_NAMES[case]}")
    return rows


def validate_paper_grid(rows: list[dict[str, float]], kind: str, case: str) -> None:
    actual = tuple(float(row["parameter"]) for row in rows)
    expected = EXPECTED_PARAMETERS[kind]
    if len(actual) != len(expected) or any(
        not math.isclose(value, target, rel_tol=0.0, abs_tol=1e-10)
        for value, target in zip(actual, expected)
    ):
        raise ValueError(
            f"{TRANSFER_NAMES[case]} {kind} settings are {actual}; expected {expected}"
        )


def _metric_limits(rows: list[dict[str, float]]) -> tuple[float, float]:
    values = [float(row[key]) for row in rows for key, *_ in METRICS]
    low = max(0.0, math.floor(min(values) - 2.0))
    high = min(100.0, math.ceil(max(values) + 2.0))
    if high - low < 8.0:
        center = 0.5 * (high + low)
        low, high = max(0.0, center - 4.0), min(100.0, center + 4.0)
    return low, high


def plot_json_sweep(rows: list[dict[str, float]], kind: str, case: str,
                    output_dir: Path) -> None:
    parameters = [float(row["parameter"]) for row in rows]
    title = CASE_TITLES[case]
    low, high = _metric_limits(rows)

    if kind == "amplitude":
        angles = np.linspace(0, 2 * np.pi, len(rows), endpoint=False)
        closed_angles = np.r_[angles, angles[0]]
        fig, ax = plt.subplots(figsize=(3.55, 3.25), subplot_kw={"projection": "polar"})
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        for key, label, color, marker in METRICS:
            values = np.asarray([float(row[key]) for row in rows])
            ax.plot(closed_angles, np.r_[values, values[0]], color=color,
                    marker=marker, linewidth=1.1, markersize=3.0, label=label)
        ax.set_xticks(angles)
        ax.set_xticklabels([f"{value:g}" for value in parameters])
        ax.set_ylim(low, high)
        ax.set_title(title, pad=10)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=False)
        stem = f"perturbation_amplitude_sensitivity_{TRANSFER_NAMES[case]}"
    elif kind == "patch":
        fig, ax = plt.subplots(figsize=(3.55, 2.75))
        x = np.arange(len(rows))
        width = 0.24
        for offset, (key, label, color, _) in zip((-width, 0.0, width), METRICS):
            ax.bar(x + offset, [float(row[key]) for row in rows], width=width,
                   label=label, color=color, edgecolor="white", linewidth=0.45)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{value:g}" for value in parameters])
        ax.set_xlabel(r"Patch size $p$")
        ax.set_ylabel("Metric value")
        ax.set_ylim(low, high)
        ax.set_title(title)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.38, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(ncol=3, frameon=False)
        stem = f"patch_size_sensitivity_{TRANSFER_NAMES[case]}"
    else:
        fig, ax = plt.subplots(figsize=(3.55, 2.75))
        x = np.arange(len(rows))
        for key, label, color, marker in METRICS:
            ax.plot(x, [float(row[key]) for row in rows], color=color,
                    marker=marker, label=label, linewidth=1.1, markersize=3.2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{value:g}" for value in parameters])
        ax.set_xlabel(r"$\lambda_{cd}$")
        ax.set_ylabel("Metric value")
        ax.set_ylim(low, high)
        ax.set_title(title)
        ax.grid(axis="y", color="#D0D0D0", linestyle="--", linewidth=0.45)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(ncol=3, frameon=False)
        stem = f"conditional_domain_weight_sensitivity_{TRANSFER_NAMES[case]}"
    fig.tight_layout()
    save(fig, stem, output_dir)


def plot_patch_bars(rows_by_case: dict[str, list[dict[str, float | str]]],
                    output_dir: Path = OUT) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(3.55, 4.55), sharey=True)
    width = 0.23
    offsets = (-width, 0.0, width)
    x = np.arange(5)
    for panel, (case, title, _) in enumerate(CASES):
        ax = axes.flat[panel]
        rows = rows_by_case[case]
        ax.axvspan(1.53, 2.47, color="#F5C04A", alpha=0.13, zorder=0)
        for offset, (key, label, color, _) in zip(offsets, METRICS):
            values = [float(row[key]) for row in rows]
            bars = ax.bar(x + offset, values, width=width, label=label,
                          color=color, edgecolor="white", linewidth=0.45, zorder=3)
            bars[2].set_edgecolor("#1E1E1E")
            bars[2].set_linewidth(0.9)
        ax.set_title(f"({chr(97 + panel)}) {title}", pad=2.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["5", "7", "9", "11", "13"])
        ax.set_xlabel(r"Patch size $p$")
        ax.set_ylim(84, 100)
        ax.set_yticks([84, 88, 92, 96, 100])
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.38, alpha=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="in", length=2.0, width=0.55)
    axes[0, 0].set_ylabel("Metric value")
    axes[1, 0].set_ylabel("Metric value")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.53, 1.005), handlelength=1.2, columnspacing=0.8)
    fig.subplots_adjust(left=0.14, right=0.995, bottom=0.08, top=0.92,
                        wspace=0.12, hspace=0.43)
    save(fig, "patch_size_sensitivity", output_dir)


def plot_lambda_lines(rows_by_case: dict[str, list[dict[str, float | str]]],
                      output_dir: Path = OUT) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(3.55, 4.55), sharex=True, sharey=True)
    x = np.arange(6)
    tick_labels = ["0", ".001", ".005", ".01", ".05", ".1"]
    for panel, (case, title, _) in enumerate(CASES):
        ax = axes.flat[panel]
        rows = rows_by_case[case]
        ax.axvspan(2.73, 3.27, color="#7A7A7A", alpha=0.09,
                   linewidth=0, zorder=0)
        ax.axvline(3, color="#666666", linestyle="--",
                   linewidth=0.85, alpha=0.8, zorder=1)
        for key, label, color, marker in METRICS:
            ax.plot(x, [float(row[key]) for row in rows], color=color,
                    marker=marker, label=label, linewidth=1.1, markersize=3.1,
                    markeredgewidth=0.45, markeredgecolor="white", zorder=3)
        ax.set_title(f"({chr(97 + panel)}) {title}", pad=2.5)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel(r"$\lambda_{cd}$")
        ax.set_xlim(-0.35, 5.35)
        ax.set_ylim(88, 99)
        ax.set_yticks([88, 90, 92, 94, 96, 98])
        ax.grid(axis="y", color="#D0D0D0", linestyle="--",
                linewidth=0.45, alpha=0.75)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="in", length=2.0, width=0.55)
    axes[0, 0].set_ylabel("Metric value")
    axes[1, 0].set_ylabel("Metric value")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.53, 1.005), handlelength=1.3,
               columnspacing=0.8)
    fig.subplots_adjust(left=0.14, right=0.995, bottom=0.08, top=0.92,
                        wspace=0.12, hspace=0.43)
    save(fig, "conditional_domain_weight_sensitivity", output_dir)


def plot_amplitude_polar(rows_by_case: dict[str, list[dict[str, float | str]]],
                         output_dir: Path = OUT) -> None:
    labels = ["0.25", "0.5", "1.0", "1.5", "2.0"]
    short_titles = ["PU to PC", "IP to SA", "CH to PC", "CH to SA"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    fig, axes = plt.subplots(2, 2, figsize=(3.65, 4.55),
                             subplot_kw={"projection": "polar"})
    for panel, (case, title, _) in enumerate(CASES):
        ax = axes.flat[panel]
        rows = rows_by_case[case]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.axvline(angles[2], color="#D39A00", linewidth=4.0,
                   alpha=0.18, zorder=0)
        for key, label, color, marker in METRICS:
            values = np.array([float(row[key]) for row in rows])
            closed_values = np.r_[values, values[0]]
            ax.plot(closed_angles, closed_values, color=color, linewidth=1.1,
                    marker=marker, markersize=2.6, markeredgecolor="white",
                    markeredgewidth=0.35, label=label, zorder=3)
            ax.fill(closed_angles, closed_values, color=color, alpha=0.035)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels)
        ax.set_ylim(84, 100)
        ax.set_yticks([88, 92, 96, 100])
        ax.set_yticklabels(["88", "92", "96", "100"] if panel == 0 else [])
        ax.set_rlabel_position(78)
        ax.grid(color="#CFCFCF", linewidth=0.45)
        ax.spines["polar"].set_color("#8A8A8A")
        ax.spines["polar"].set_linewidth(0.55)
        ax.set_title(f"({chr(97 + panel)}) {short_titles[panel]}", pad=7)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.52, 0.01),
               handlelength=1.3, columnspacing=0.8)
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.085, top=0.97,
                        wspace=0.46, hspace=0.34)
    save(fig, "perturbation_amplitude_sensitivity", output_dir)


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 6.7,
        "axes.titlesize": 7.1,
        "axes.labelsize": 6.9,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 5.9,
        "legend.fontsize": 6.4,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def parse_args():
    parser = argparse.ArgumentParser(description="Plot LGSPR-Net sensitivity experiments")
    parser.add_argument("--type", choices=("patch", "lambda", "amplitude"))
    parser.add_argument("--transfer", choices=("all",) + tuple(TRANSFERS))
    parser.add_argument("--input_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, default=OUT)
    args = parser.parse_args()
    direct = (args.type, args.transfer, args.input_dir)
    if any(value is not None for value in direct) and not all(value is not None for value in direct):
        parser.error("--type, --transfer, and --input_dir must be provided together")
    return args


def main() -> None:
    args = parse_args()
    configure_style()
    if args.type:
        if args.transfer == "all":
            rows_by_case = {
                case: read_json_sweep(args.input_dir, args.type, case)
                for case, _, _ in CASES
            }
            for case, rows in rows_by_case.items():
                validate_paper_grid(rows, args.type, case)
            plotter = {
                "patch": plot_patch_bars,
                "lambda": plot_lambda_lines,
                "amplitude": plot_amplitude_polar,
            }[args.type]
            plotter(rows_by_case, args.output_dir)
            return
        case = TRANSFERS[args.transfer]
        rows = read_json_sweep(args.input_dir, args.type, case)
        plot_json_sweep(rows, args.type, case, args.output_dir)
        return
    patch = {case: read_sweep(path, "patch_size") for case, path in PATCH_CSV.items()}
    conditional = {case: read_sweep(path, "lambda_conditional_domain")
                   for case, path in LAMBDA_CSV.items()}
    amplitude = {case: read_sweep(path, "perturbation_amplitude_multiplier")
                 for case, path in AMPLITUDE_CSV.items()}
    for case, _, _ in CASES:
        canonicalize(patch[case], 9.0, case)
        canonicalize(conditional[case], 0.01, case)
        canonicalize(amplitude[case], 1.0, case)
    plot_patch_bars(patch)
    plot_lambda_lines(conditional)
    plot_amplitude_polar(amplitude)


if __name__ == "__main__":
    main()
