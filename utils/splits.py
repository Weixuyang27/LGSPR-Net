"""Portable original-image flat pixel indices; no dependency on torch."""
import hashlib
import json
from pathlib import Path

import numpy as np


def gt_digest(gt):
    return hashlib.sha256(np.asarray(gt, dtype="<i8").tobytes()).hexdigest()


def load_or_create_split(path, gt, shots=5, seed=1337, max_test_per_class=0):
    gt = np.asarray(gt)
    if gt.ndim != 2 or not np.issubdtype(gt.dtype, np.integer) or gt.min() < 0:
        raise ValueError("Ground truth must be a nonnegative integer 2D array")
    classes = np.unique(gt[gt > 0])
    if not np.array_equal(classes, np.arange(1, len(classes) + 1)) or len(classes) < 2:
        raise ValueError("Expected contiguous labels 1..C and background 0")
    if shots < 1 or max_test_per_class < 0:
        raise ValueError("shots must be positive; max_test_per_class must be nonnegative")
    metadata = dict(version=1, shape=list(gt.shape), gt_sha256=gt_digest(gt),
                    shots=int(shots), seed=int(seed), max_test_per_class=int(max_test_per_class))
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if json.loads(str(saved["metadata"])) != metadata:
                raise ValueError(f"Split metadata does not match this experiment: {path}")
            train, test = saved["train"].copy(), saved["test"].copy()
    else:
        rng = np.random.RandomState(seed)
        trains, tests = [], []
        for label in classes:
            indices = np.flatnonzero(gt.ravel() == label)
            if len(indices) <= shots:
                raise ValueError(f"Class {label} needs > {shots} pixels")
            rng.shuffle(indices)
            trains.append(indices[:shots])
            remaining = indices[shots:]
            tests.append(remaining[:max_test_per_class] if max_test_per_class else remaining)
        train, test = np.concatenate(trains), np.concatenate(tests)
        rng.shuffle(test)
    for indices in (train, test):
        if indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) == 0:
            raise ValueError("Invalid split index array")
        if indices.min() < 0 or indices.max() >= gt.size or len(np.unique(indices)) != len(indices):
            raise ValueError("Duplicate or out-of-range split indices")
        if np.any(gt.ravel()[indices] == 0):
            raise ValueError("Split includes background pixels")
    if np.intersect1d(train, test).size:
        raise ValueError("Training and test pixels overlap")
    for label in classes:
        if np.sum(gt.ravel()[train] == label) != shots:
            raise ValueError("Incorrect training label budget")
        expected = int(np.sum(gt == label)) - shots
        expected = min(expected, max_test_per_class) if max_test_per_class else expected
        if np.sum(gt.ravel()[test] == label) != expected:
            raise ValueError("Incorrect test coverage")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(stream, train=train, test=test, metadata=json.dumps(metadata))
    return train, test


def fixed_target_pack(cube, gt, split_path, shots, seed, max_test_per_class=0,
                      half_width=4):
    """Match the target-pack interface with configurable odd patch size."""
    import torch
    from .data import LazyHSIPatchDataset

    train, test = load_or_create_split(split_path, gt, shots, seed, max_test_per_class)
    half_width = int(half_width)
    if half_width < 1:
        raise ValueError("half_width must be positive")
    padded_cube = np.pad(
        cube,
        ((half_width, half_width), (half_width, half_width), (0, 0)),
        mode="constant",
    )
    padded_gt = np.pad(gt, half_width, mode="constant")
    # Split indices refer only to labeled pixels in the original image.
    row, col = np.nonzero(gt)
    original = np.ravel_multi_index((row, col), gt.shape)
    train_idx, test_idx = np.searchsorted(original, train), np.searchsorted(original, test)
    row, col = row + half_width, col + half_width
    train_data = LazyHSIPatchDataset(
        padded_cube, padded_gt, row, col, train_idx, half_width
    )
    test_data = LazyHSIPatchDataset(
        padded_cube, padded_gt, row, col, test_idx, half_width
    )
    return (torch.utils.data.DataLoader(train_data, batch_size=len(train_data)),
            torch.utils.data.DataLoader(test_data, batch_size=100), {}, None,
            padded_gt, np.concatenate((train_idx, test_idx)), row, col, len(train_data))
