"""Shared target-map inference and visualization utilities for LGSPR-Net."""
import json
import os

import numpy as np
import scipy.io as sio
import torch
from sklearn import metrics

from . import data as ssd
from .draw_dataset import palette_for_case, save_label_map
from .metrics import average_accuracy
from .splits import load_or_create_split
from models.lgspr_net import LGSPRNet, prototype_logits


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_target_loaders(
    cube,
    ground_truth,
    n_way,
    labeled_per_class,
    seed,
    batch_size,
    max_test_per_class=0,
    half_width=4,
    target_split=None,
):
    """Reproduce the training target split, but extract all patches lazily."""
    np.random.seed(seed)
    cube_pad = np.pad(
        np.asarray(cube, dtype=np.float32),
        ((half_width, half_width), (half_width, half_width), (0, 0)),
        mode="constant",
    )
    gt = np.asarray(ground_truth, dtype=np.int64)
    gt_pad = np.pad(gt, ((half_width, half_width), (half_width, half_width)), mode="constant")
    rows, columns = np.nonzero(gt)
    original_indices = np.ravel_multi_index((rows, columns), gt.shape)
    if target_split:
        support_flat, test_flat = load_or_create_split(
            target_split,
            gt,
            shots=labeled_per_class,
            seed=seed,
            max_test_per_class=max_test_per_class,
        )
        support_indices = np.searchsorted(original_indices, support_flat).tolist()
        test_indices = np.searchsorted(original_indices, test_flat).tolist()
    else:
        labels = gt[rows, columns]
        support_indices, test_indices = [], []
        for cls in range(n_way):
            class_indices = np.flatnonzero(labels == cls + 1).tolist()
            np.random.shuffle(class_indices)
            if len(class_indices) < labeled_per_class:
                raise ValueError(
                    f"class {cls} has {len(class_indices)} labeled pixels, "
                    f"fewer than labeled_per_class={labeled_per_class}"
                )
            support_indices.extend(class_indices[:labeled_per_class])
            class_test = class_indices[labeled_per_class:]
            if max_test_per_class > 0:
                class_test = class_test[:max_test_per_class]
            test_indices.extend(class_test)
        np.random.shuffle(test_indices)
    rows = rows + half_width
    columns = columns + half_width

    support_dataset = ssd.LazyHSIPatchDataset(
        cube_pad, gt_pad, rows, columns, support_indices, half_width
    )
    test_dataset = ssd.LazyHSIPatchDataset(
        cube_pad, gt_pad, rows, columns, test_indices, half_width
    )
    support_loader = torch.utils.data.DataLoader(
        support_dataset,
        batch_size=max(len(support_dataset), 1),
        shuffle=False,
        num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    print(
        f"[data] target split: support={len(support_indices)}, test={len(test_indices)}, "
        f"seed={seed} split={target_split or 'generated in memory'}",
        flush=True,
    )
    return support_loader, test_loader, rows, columns, support_indices, test_indices


def _predict(encoder, support_loader, test_loader, device, n_way, temperature,
             map_target_input=False):
    encoder.eval()
    support_x, support_y = next(iter(support_loader))
    support_x = support_x.float().to(device)
    support_y = support_y.long().to(device)
    if map_target_input:
        support_x = encoder.map_spectral_input(support_x, "target")
    predictions, labels = [], []
    with torch.no_grad():
        zc_support = encoder(support_x)
        for query_x, query_y in test_loader:
            query_x = query_x.float().to(device)
            if map_target_input:
                query_x = encoder.map_spectral_input(query_x, "target")
            zc_query = encoder(query_x)
            logits = prototype_logits(
                zc_query, zc_support, support_y, n_way, temperature
            )
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            labels.append(query_y.numpy())
    return np.concatenate(predictions).astype(np.int64), np.concatenate(labels).astype(np.int64)


def run_target_map(
    *,
    case_name,
    cube,
    ground_truth,
    checkpoint,
    out_dir,
    gpu,
    seed,
    n_bands,
    n_way,
    labeled_per_class,
    batch_size,
    proto_temperature,
    category_dim,
    response_dim,
    max_test_per_class=0,
    patch_size=9,
    target_split=None,
):
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    ckpt = _torch_load(checkpoint, device)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(
            "checkpoint is not in the current LGSPR-Net format: missing 'model'"
        )
    checkpoint_case = ckpt.get("case")
    if checkpoint_case is not None and checkpoint_case != case_name:
        raise ValueError(
            f"checkpoint belongs to {checkpoint_case}, but evaluation requested {case_name}"
        )
    n_bands = int(ckpt.get("n_bands", n_bands))
    n_way = int(ckpt.get("n_way", n_way))
    labeled_per_class = int(ckpt.get("labeled_per_class", labeled_per_class))
    patch_size = int(ckpt.get("patch_size", patch_size))
    proto_temperature = float(ckpt.get("proto_temperature", proto_temperature))
    split_seed = int(ckpt.get("split_seed", seed))
    if target_split is None:
        saved_split = ckpt.get("target_split")
        if saved_split and os.path.exists(saved_split):
            target_split = saved_split
    palette = palette_for_case(case_name)
    if n_way + 1 > len(palette):
        raise ValueError(f"palette supports at most {len(palette) - 1} classes")

    if patch_size < 3 or patch_size % 2 == 0:
        raise ValueError("patch_size must be an odd integer >= 3")
    half_width = patch_size // 2
    support_loader, test_loader, rows, columns, support_indices, test_indices = build_target_loaders(
        cube,
        ground_truth,
        n_way,
        labeled_per_class,
        split_seed,
        batch_size,
        max_test_per_class=max_test_per_class,
        half_width=half_width,
        target_split=target_split,
    )
    encoder_state = ckpt["model"]
    category_dim = int(ckpt.get("category_dim", category_dim))
    response_dim = int(ckpt.get("response_dim", response_dim))
    use_lgsse = bool(ckpt.get("lgsse", True))
    encoder = LGSPRNet(
        n_bands=n_bands,
        category_dim=category_dim,
        response_dim=response_dim,
        use_lgsse=use_lgsse,
    )
    source_input_bands = int(ckpt.get("source_input_bands", n_bands)) if isinstance(ckpt, dict) else n_bands
    target_input_bands = int(ckpt.get("target_input_bands", n_bands)) if isinstance(ckpt, dict) else n_bands
    map_target_input = target_input_bands != n_bands or any(
        key.startswith("target_mapping.") for key in encoder_state
    )
    if map_target_input:
        if cube.shape[2] != target_input_bands:
            raise ValueError(
                f"target cube has {cube.shape[2]} bands, but checkpoint expects "
                f"{target_input_bands} before spectral mapping"
            )
        encoder.configure_spectral_mappings(source_input_bands, target_input_bands)
    encoder = encoder.to(device)
    training_only_prefixes = (
        "perturbation_predictor.", "pre_norm_response.", "relation_module.",
    )
    inference_state = {
        key: value for key, value in encoder_state.items()
        if not key.startswith(training_only_prefixes)
    }
    missing, unexpected = encoder.load_state_dict(inference_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint does not match LGSPR-Net inference model: "
            f"missing={missing}, unexpected={unexpected}"
        )
    print(
        f"[predict] checkpoint={checkpoint} device={device} "
        "inference=category embedding and support-derived prototypes",
        flush=True,
    )

    prediction, labels = _predict(
        encoder, support_loader, test_loader, device, n_way, proto_temperature,
        map_target_input=map_target_input,
    )
    if len(prediction) != len(test_indices):
        raise RuntimeError(
            f"prediction count mismatch: predictions={len(prediction)}, test={len(test_indices)}"
        )

    gt_map = np.asarray(ground_truth, dtype=np.uint16)
    prediction_map = np.zeros_like(gt_map, dtype=np.uint16)
    for sample_index in support_indices:
        r = int(rows[sample_index]) - half_width
        c = int(columns[sample_index]) - half_width
        prediction_map[r, c] = gt_map[r, c]
    for sample_index, pred in zip(test_indices, prediction):
        r = int(rows[sample_index]) - half_width
        c = int(columns[sample_index]) - half_width
        prediction_map[r, c] = int(pred) + 1

    oa = 100.0 * metrics.accuracy_score(labels, prediction)
    cm = metrics.confusion_matrix(labels, prediction, labels=list(range(n_way)))
    each_acc, aa = average_accuracy(cm)
    kappa = metrics.cohen_kappa_score(labels, prediction)
    print(f"[predict] OA={oa:.2f}% AA={aa * 100.0:.2f}% Kappa={kappa:.4f}", flush=True)
    for i, value in enumerate(each_acc):
        print(f"class {i}: {value * 100.0:.2f}%", flush=True)

    prefix = case_name.lower()
    paths = {
        "prediction_png": os.path.join(out_dir, f"{prefix}_prediction_map.png"),
        "ground_truth_png": os.path.join(out_dir, f"{prefix}_ground_truth.png"),
        "prediction_npy": os.path.join(out_dir, f"{prefix}_prediction_map.npy"),
        "prediction_mat": os.path.join(out_dir, f"{prefix}_prediction_map.mat"),
        "metrics_json": os.path.join(out_dir, f"{prefix}_map_metrics.json"),
    }
    save_label_map(prediction_map, paths["prediction_png"], n_way, palette)
    save_label_map(gt_map, paths["ground_truth_png"], n_way, palette)
    np.save(paths["prediction_npy"], prediction_map)
    sio.savemat(
        paths["prediction_mat"],
        {
            "prediction_map": prediction_map,
            "ground_truth": gt_map,
            "oa": oa,
            "aa": aa * 100.0,
            "kappa": kappa,
            "each_acc": each_acc * 100.0,
            "confusion_matrix": cm,
        },
    )
    checkpoint_episode = int(ckpt.get("episode", -1)) if isinstance(ckpt, dict) else -1
    with open(paths["metrics_json"], "w", encoding="utf-8") as handle:
        json.dump(
            {
                "case": case_name,
                "checkpoint": checkpoint,
                "checkpoint_episode": checkpoint_episode,
                "seed": int(ckpt.get("seed", seed)),
                "split_seed": split_seed,
                "target_split": target_split,
                "support_per_class": int(labeled_per_class),
                "test_samples": int(len(labels)),
                "oa": float(oa),
                "aa": float(aa * 100.0),
                "kappa": float(kappa),
                "each_acc": [float(v * 100.0) for v in each_acc],
                "inference_feature": "z_cls_only",
                "prototype_metric": "episode_scale_normalized_euclidean",
            },
            handle,
            indent=2,
        )
    for path in paths.values():
        print(f"[save] {path}", flush=True)
    return paths
