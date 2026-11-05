"""Dataset loading and episodic patch extraction for LGSPR-Net."""
import math
import random
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
from .episodic import (
    ArraySplitDataset,
    ClassBalancedSampler,
    EpisodicDataset,
    radiation_noise,
    zero_pad_cube,
)


class LazyHSIPatchDataset(torch.utils.data.Dataset):
    """Extract channel-first HSI patches on demand instead of materializing them all."""

    def __init__(self, padded_cube, padded_gt, rows, columns, sample_indices, half_width):
        self.padded_cube = padded_cube
        self.padded_gt = padded_gt
        self.rows = rows
        self.columns = columns
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        self.half_width = int(half_width)

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, item):
        sample_index = int(self.sample_indices[item])
        row = int(self.rows[sample_index])
        column = int(self.columns[sample_index])
        h = self.half_width
        patch = self.padded_cube[
            row - h : row + h + 1,
            column - h : column + h + 1,
            :,
        ]
        patch = np.ascontiguousarray(patch.transpose(2, 0, 1), dtype=np.float32)
        label = int(self.padded_gt[row, column]) - 1
        return torch.from_numpy(patch), label


def align_bands(cube, n_bands):
    """Truncate or pad spectral dimension to n_bands."""
    if n_bands is None or n_bands <= 0:
        return cube
    s = cube.shape[2]
    if s == n_bands:
        return cube
    if s > n_bands:
        return cube[:, :, :n_bands]
    pad = n_bands - s
    return np.pad(cube, ((0, 0), (0, 0), (0, pad)), mode="edge")


def resample_bands(cube, n_bands):
    """Uniformly resample the full spectral span to n_bands."""
    if n_bands is None or n_bands <= 0 or cube.shape[2] == n_bands:
        return cube
    indices = np.linspace(0, cube.shape[2] - 1, int(n_bands)).round().astype(np.int64)
    return np.ascontiguousarray(cube[:, :, indices])


def load_pavia_university(image_file, label_file, n_bands=102):
    try:
        image_data = sio.loadmat(image_file)
        label_data = sio.loadmat(label_file)
    except Exception as exc:
        raise RuntimeError(f"Failed to read Pavia University mat files: {exc}") from exc
    data_all = image_data["paviaU"]
    label = label_data["paviaU_gt"]
    data = data_all.reshape(-1, data_all.shape[2])
    data_scaler = preprocessing.scale(data)
    cube = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])
    cube = align_bands(cube, n_bands)
    return cube, label


def load_pavia_centre(image_file, label_file, n_bands=102):
    try:
        image_data = sio.loadmat(image_file)
        label_data = sio.loadmat(label_file)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read Pavia Centre mat files: {exc}. "
            f"Please verify file integrity or re-download '{image_file}'."
        ) from exc
    data_all = image_data["pavia"]
    label = label_data["pavia_gt"]
    data = data_all.reshape(-1, data_all.shape[2])
    data_scaler = preprocessing.scale(data)
    cube = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])
    cube = align_bands(cube, n_bands)
    return cube, label


def _first_existing_key(mat_dict, candidates, file_path):
    for key in candidates:
        if key in mat_dict:
            return key
    data_keys = [k for k in mat_dict.keys() if not k.startswith("__")]
    raise KeyError(f"None of keys {candidates} found in {file_path}. Available keys: {data_keys}")


def load_hsi_mat(image_file, label_file, image_keys, label_keys, n_bands=None):
    """Load a generic HSI .mat cube/gt pair and standardize spectral values."""
    try:
        image_data = sio.loadmat(image_file)
        label_data = sio.loadmat(label_file)
    except Exception as exc:
        raise RuntimeError(f"Failed to read HSI mat files: {exc}") from exc
    image_key = _first_existing_key(image_data, image_keys, image_file)
    label_key = _first_existing_key(label_data, label_keys, label_file)
    data_all = image_data[image_key]
    label = label_data[label_key]
    data = data_all.reshape(-1, data_all.shape[2])
    data_scaler = preprocessing.scale(data)
    cube = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])
    if n_bands is not None and n_bands > 0:
        cube = align_bands(cube, n_bands)
    return cube, label


def parse_label_map(map_text):
    """Parse mappings such as '2,3,4:0;5,6,7:1' into 1-based-to-0-based labels."""
    out = {}
    if not map_text:
        return out
    for group in map_text.split(";"):
        group = group.strip()
        if not group:
            continue
        left, right = group.split(":")
        dst = int(right.strip())
        for item in left.split(","):
            src = int(item.strip())
            out[src] = dst
    return out


def remap_groundtruth(gt, label_map):
    """Keep mapped labels only; output labels are 1..N and background is 0."""
    mapped = np.zeros_like(gt, dtype=np.int64)
    for src_label, dst_zero_based in label_map.items():
        mapped[gt == int(src_label)] = int(dst_zero_based) + 1
    return mapped


def load_indian_pines(image_file, label_file, n_bands=200):
    return load_hsi_mat(
        image_file,
        label_file,
        image_keys=["indian_pines_corrected", "indian_pines", "Indian_pines_corrected"],
        label_keys=["indian_pines_gt", "Indian_pines_gt"],
        n_bands=n_bands,
    )


def load_salinas(image_file, label_file, n_bands=200):
    return load_hsi_mat(
        image_file,
        label_file,
        image_keys=["salinas_corrected", "salinas", "Salinas_corrected"],
        label_keys=["salinas_gt", "Salinas_gt"],
        n_bands=n_bands,
    )


def _unwrap_mat_value(value):
    """Unwrap singleton MATLAB cell/struct containers returned by loaders."""
    value = np.asarray(value)
    while value.dtype == object and value.size == 1:
        value = np.asarray(value.reshape(-1)[0])
    return np.squeeze(value)


def _orient_chikusei_cube(cube, label):
    """Return a Chikusei cube in row/column/band order."""
    if cube.ndim != 3 or label.ndim != 2:
        raise ValueError(
            f"Expected a 3-D cube and 2-D label map, got {cube.shape} and {label.shape}"
        )
    for axes in ((0, 1, 2), (1, 0, 2), (2, 1, 0), (1, 2, 0), (2, 0, 1), (0, 2, 1)):
        candidate = np.transpose(cube, axes)
        if candidate.shape[:2] == label.shape:
            return candidate
    raise ValueError(
        f"Cannot align Chikusei cube shape {cube.shape} with label shape {label.shape}"
    )


def load_chikusei(image_file, label_file, n_bands=100):
    """Load Chikusei without materializing its 6-GB float64 cube in RAM."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "Chikusei loading requires h5py; install it with 'pip install h5py'."
        ) from exc

    try:
        label_data = sio.loadmat(label_file)
    except Exception as exc:
        raise RuntimeError(f"Failed to read Chikusei ground truth: {exc}") from exc
    label_key = _first_existing_key(
        label_data, ["GT", "gt", "chikusei_gt", "Chikusei_gt"], label_file
    )
    label_root = label_data[label_key]
    if label_root.dtype.names and "gt" in label_root.dtype.names:
        label = np.asarray(label_root["gt"][0, 0])
    else:
        label = _unwrap_mat_value(label_root)
    label = np.squeeze(label).astype(np.int64, copy=False)

    try:
        with h5py.File(image_file, "r") as handle:
            image_key = next(
                (key for key in ("chikusei", "Chikusei", "HyperspecVNIR_Chikusei_20140729")
                 if key in handle),
                None,
            )
            if image_key is None:
                raise KeyError(f"No Chikusei cube found; available keys: {list(handle.keys())}")
            dataset = handle[image_key]
            if dataset.ndim != 3:
                raise ValueError(f"Expected a 3-D Chikusei cube, got {dataset.shape}")

            band_axis = None
            transpose_spatial = False
            for axis in range(3):
                spatial_shape = tuple(dataset.shape[i] for i in range(3) if i != axis)
                if spatial_shape == label.shape:
                    band_axis, transpose_spatial = axis, False
                    break
                if spatial_shape == label.shape[::-1]:
                    band_axis, transpose_spatial = axis, True
                    break
            if band_axis is None:
                raise ValueError(
                    f"Cannot align cube shape {dataset.shape} with label shape {label.shape}"
                )

            native_bands = dataset.shape[band_axis]
            selected = np.linspace(0, native_bands - 1, int(n_bands)).round().astype(np.int64)
            cube = np.empty((*label.shape, len(selected)), dtype=np.float32)
            if band_axis == 0 and transpose_spatial:
                # Row blocks avoid repeatedly decompressing the chunked cube.
                row_block = 8
                for start in range(0, label.shape[0], row_block):
                    stop = min(start + row_block, label.shape[0])
                    block = np.asarray(dataset[:, :, start:stop], dtype=np.float32)
                    cube[start:stop] = block[selected].transpose(2, 1, 0)
                    if start == 0 or stop == label.shape[0] or stop % 256 == 0:
                        print(
                            f"[data] Chikusei spectral loading: {stop}/{label.shape[0]} rows",
                            flush=True,
                        )
            else:
                for output_band, source_band in enumerate(selected):
                    index = [slice(None)] * 3
                    index[band_axis] = int(source_band)
                    band = np.asarray(dataset[tuple(index)], dtype=np.float32)
                    if transpose_spatial:
                        band = band.T
                    cube[:, :, output_band] = band

            for output_band in range(cube.shape[2]):
                band = cube[:, :, output_band]
                mean = float(band.mean(dtype=np.float64))
                std = float(band.std(dtype=np.float64))
                band -= mean
                band /= std if std >= 1e-6 else 1.0
    except Exception as exc:
        raise RuntimeError(f"Failed to read Chikusei image: {exc}") from exc
    return cube, label


def build_metatrain_dict(cube, gt, max_per_class=200, half_width=4, seed=None):
    """Labeled pixels -> per-class list of patches (n_bands, 2*half_width+1, 2*half_width+1)."""
    g = gt.copy()
    g[g == 0] = 0
    h = half_width
    rows, cols = np.nonzero(g)
    h_max, w_max = cube.shape[0], cube.shape[1]
    coordinates = {}
    for idx in range(len(rows)):
        r, c = int(rows[idx]), int(cols[idx])
        lab = int(g[r, c])
        if lab <= 0:
            continue
        if r - h < 0 or r + h >= h_max or c - h < 0 or c + h >= w_max:
            continue
        cls = lab - 1
        coordinates.setdefault(cls, []).append((r, c))
    out = {}
    rng = np.random.default_rng(seed)
    for k, coords in coordinates.items():
        if not coords:
            continue
        if len(coords) > max_per_class:
            # Random selection avoids sampling one contiguous image region.
            chosen = rng.choice(len(coords), size=max_per_class, replace=False)
            coords = [coords[int(i)] for i in chosen]
        patches = []
        for r, c in coords:
            patch = cube[r - h : r + h + 1, c - h : c + h + 1, :]
            if patch.shape[:2] != (2 * h + 1, 2 * h + 1):
                continue
            patches.append(np.ascontiguousarray(patch.transpose(2, 0, 1), dtype=np.float32))
        if patches:
            out[k] = patches
    return out


def get_train_test_loader_pc(
    Data_Band_Scaler,
    GroundTruth,
    shot_num_per_class,
    test_lsample_num_per_class,
    half_width=4,
    max_test_per_class=0,
    progress_interval=5000,
    build_da=True,
):
    """Build the Pavia Centre support/test loaders used in Case 1."""
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape
    print(f"[data] target cube shape={Data_Band_Scaler.shape}, building padded view", flush=True)
    gt1 = GroundTruth.reshape(Data_Band_Scaler.shape[0], Data_Band_Scaler.shape[1])
    HalfWidth = half_width
    data = np.pad(
        np.asarray(Data_Band_Scaler, dtype=np.float32),
        ((HalfWidth, HalfWidth), (HalfWidth, HalfWidth), (0, 0)),
        mode="constant",
    )
    G = np.pad(gt1, ((HalfWidth, HalfWidth), (HalfWidth, HalfWidth)), mode="constant")
    Row, Column = np.nonzero(G)
    print(f"[data] target labeled pixels={len(Row)}, classes={int(np.max(G))}", flush=True)

    train, test, da_train = {}, {}, {}
    m = int(np.max(G))
    nlabeled = test_lsample_num_per_class
    for i in range(m):
        indices = [j for j in range(len(Row)) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        nb_val = shot_num_per_class
        train[i] = indices[:nb_val]
        da_train[i] = []
        if build_da:
            for _ in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
                da_train[i] += indices[:nb_val]
        test_indices_i = indices[nb_val:]
        if max_test_per_class and max_test_per_class > 0:
            test_indices_i = test_indices_i[:max_test_per_class]
        test[i] = test_indices_i
        print(
            f"[data] class {i}: support={len(train[i])}, test={len(test[i])}, da={len(da_train[i])}",
            flush=True,
        )

    train_indices, test_indices, da_train_indices = [], [], []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)
    print(f"[data] patch counts: train={nTrain}, test={nTest}, da_train={da_nTrain}", flush=True)

    RandPerm = np.array(train_indices + test_indices)
    print("[data] using lazy target train/test patch extraction", flush=True)
    train_dataset = LazyHSIPatchDataset(data, G, Row, Column, train_indices, HalfWidth)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=max(len(train_indices), 1), shuffle=False, num_workers=0
    )
    test_dataset = LazyHSIPatchDataset(data, G, Row, Column, test_indices, HalfWidth)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)

    imdb_da_train = {}
    imdb_da_train["data"] = np.zeros(
        [2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain], dtype=np.float32
    )
    imdb_da_train["Labels"] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train["set"] = np.ones([da_nTrain]).astype(np.int64)
    da_RandPerm = np.array(da_train_indices)
    if build_da:
        print("[data] building target DA patch tensor", flush=True)
    else:
        print("[data] skipping repeated target DA tensor; using real support patches", flush=True)
    for iSample in range(da_nTrain):
        imdb_da_train["data"][:, :, :, iSample] = radiation_noise(
            data[
                Row[da_RandPerm[iSample]] - HalfWidth : Row[da_RandPerm[iSample]] + HalfWidth + 1,
                Column[da_RandPerm[iSample]]
                - HalfWidth : Column[da_RandPerm[iSample]]
                + HalfWidth
                + 1,
                :,
            ]
        )
        imdb_da_train["Labels"][iSample] = G[
            Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]
        ].astype(np.int64)
        if progress_interval and progress_interval > 0 and (
            (iSample + 1) % progress_interval == 0 or iSample + 1 == da_nTrain
        ):
            print(f"[data] DA patches {iSample + 1}/{da_nTrain}", flush=True)
    imdb_da_train["Labels"] = imdb_da_train["Labels"] - 1

    return train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain


def get_train_test_loader_generic(
    Data_Band_Scaler,
    GroundTruth,
    shot_num_per_class,
    test_lsample_num_per_class,
    half_width=4,
    max_test_per_class=0,
    progress_interval=5000,
    build_da=True,
):
    """Build support/test loaders for any mapped target ground truth with labels 1..N."""
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape
    print(f"[data] target cube shape={Data_Band_Scaler.shape}, building padded view", flush=True)
    data_band_scaler = zero_pad_cube(Data_Band_Scaler)
    gt1 = GroundTruth.reshape(Data_Band_Scaler.shape[0], Data_Band_Scaler.shape[1])
    groundtruth = zero_pad_cube(gt1)

    HalfWidth = half_width
    G = groundtruth[
        nRow - HalfWidth : 2 * nRow + HalfWidth,
        nColumn - HalfWidth : 2 * nColumn + HalfWidth,
    ]
    data = data_band_scaler[
        nRow - HalfWidth : 2 * nRow + HalfWidth,
        nColumn - HalfWidth : 2 * nColumn + HalfWidth,
        :,
    ]
    Row, Column = np.nonzero(G)
    print(f"[data] target labeled pixels={len(Row)}, classes={int(np.max(G))}", flush=True)

    train, test, da_train = {}, {}, {}
    m = int(np.max(G))
    nlabeled = test_lsample_num_per_class
    for i in range(m):
        indices = [j for j in range(len(Row)) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        nb_val = min(shot_num_per_class, len(indices))
        train[i] = indices[:nb_val]
        da_train[i] = []
        if build_da and nb_val > 0:
            for _ in range(math.ceil((200 - nlabeled) / max(nlabeled, 1)) + 1):
                da_train[i] += indices[:nb_val]
        test_indices_i = indices[nb_val:]
        if max_test_per_class and max_test_per_class > 0:
            test_indices_i = test_indices_i[:max_test_per_class]
        test[i] = test_indices_i
        print(
            f"[data] class {i}: support={len(train[i])}, test={len(test[i])}, da={len(da_train[i])}",
            flush=True,
        )

    train_indices, test_indices, da_train_indices = [], [], []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)
    print(f"[data] patch counts: train={nTrain}, test={nTest}, da_train={da_nTrain}", flush=True)

    RandPerm = np.array(train_indices + test_indices)
    # Lazy extraction avoids materializing the full patch tensor.
    print("[data] using lazy target train/test patch extraction", flush=True)
    train_dataset = LazyHSIPatchDataset(
        data, G, Row, Column, train_indices, HalfWidth
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=max(len(train_indices), 1), shuffle=False, num_workers=0
    )
    test_dataset = LazyHSIPatchDataset(
        data, G, Row, Column, test_indices, HalfWidth
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)

    imdb_da_train = {}
    imdb_da_train["data"] = np.zeros(
        [2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain], dtype=np.float32
    )
    imdb_da_train["Labels"] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train["set"] = np.ones([da_nTrain]).astype(np.int64)
    da_RandPerm = np.array(da_train_indices)
    if build_da:
        print("[data] building target DA patch tensor", flush=True)
    else:
        print("[data] skipping repeated target DA tensor; using real support patches", flush=True)
    for iSample in range(da_nTrain):
        imdb_da_train["data"][:, :, :, iSample] = radiation_noise(
            data[
                Row[da_RandPerm[iSample]] - HalfWidth : Row[da_RandPerm[iSample]] + HalfWidth + 1,
                Column[da_RandPerm[iSample]]
                - HalfWidth : Column[da_RandPerm[iSample]]
                + HalfWidth
                + 1,
                :,
            ]
        )
        imdb_da_train["Labels"][iSample] = G[
            Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]
        ].astype(np.int64)
        if progress_interval and progress_interval > 0 and (
            (iSample + 1) % progress_interval == 0 or iSample + 1 == da_nTrain
        ):
            print(f"[data] DA patches {iSample + 1}/{da_nTrain}", flush=True)
    imdb_da_train["Labels"] = imdb_da_train["Labels"] - 1

    return train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain


def get_target_pack(
    Data_Band_Scaler,
    GroundTruth,
    class_num,
    shot_num_per_class,
    test_lsample_num_per_class,
    max_test_per_class=0,
    progress_interval=5000,
    build_da=True,
    target_split=None,
    split_seed=1337,
    half_width=4,
):
    if target_split:
        if build_da:
            raise ValueError("Fixed splits support the real-support path (build_da=False)")
        if int(np.max(GroundTruth)) != class_num:
            raise ValueError("class_num does not match target ground truth")
        from .splits import fixed_target_pack
        return fixed_target_pack(
            Data_Band_Scaler, GroundTruth, target_split,
            shot_num_per_class, split_seed, max_test_per_class,
            half_width=half_width,
        )
    train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain = get_train_test_loader_pc(
        Data_Band_Scaler,
        GroundTruth,
        shot_num_per_class=shot_num_per_class,
        test_lsample_num_per_class=test_lsample_num_per_class,
        half_width=half_width,
        max_test_per_class=max_test_per_class,
        progress_interval=progress_interval,
        build_da=build_da,
    )
    target_da_datas = np.transpose(imdb_da_train["data"], (3, 2, 0, 1))
    target_da_labels = imdb_da_train["Labels"]
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        target_da_train_set.setdefault(class_, []).append(path)
    target_loader = None
    if build_da:
        target_dataset = ArraySplitDataset(imdb_da_train, train=True)
        target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=128, shuffle=True, num_workers=0)
    return (
        train_loader,
        test_loader,
        target_da_train_set,
        target_loader,
        G,
        RandPerm,
        Row,
        Column,
        nTrain,
    )


def get_target_pack_generic(
    Data_Band_Scaler,
    GroundTruth,
    class_num,
    shot_num_per_class,
    test_lsample_num_per_class,
    max_test_per_class=0,
    progress_interval=5000,
    build_da=True,
    target_split=None,
    split_seed=1337,
    half_width=4,
):
    if target_split:
        if build_da:
            raise ValueError("Fixed splits support the real-support path (build_da=False)")
        if int(np.max(GroundTruth)) != class_num:
            raise ValueError("class_num does not match target ground truth")
        from .splits import fixed_target_pack
        return fixed_target_pack(
            Data_Band_Scaler, GroundTruth, target_split,
            shot_num_per_class, split_seed, max_test_per_class,
            half_width=half_width,
        )
    train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain = get_train_test_loader_generic(
        Data_Band_Scaler,
        GroundTruth,
        shot_num_per_class=shot_num_per_class,
        test_lsample_num_per_class=test_lsample_num_per_class,
        half_width=half_width,
        max_test_per_class=max_test_per_class,
        progress_interval=progress_interval,
        build_da=build_da,
    )
    if int(np.max(G)) != class_num:
        raise RuntimeError(f"Mapped target has {int(np.max(G))} classes, but class_num={class_num}")
    target_da_datas = np.transpose(imdb_da_train["data"], (3, 2, 0, 1))
    target_da_labels = imdb_da_train["Labels"]
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        target_da_train_set.setdefault(class_, []).append(path)
    target_loader = None
    if build_da:
        target_dataset = ArraySplitDataset(imdb_da_train, train=True)
        target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=128, shuffle=True, num_workers=0)
    return (
        train_loader,
        test_loader,
        target_da_train_set,
        target_loader,
        G,
        RandPerm,
        Row,
        Column,
        nTrain,
    )


class TaskMatchedClasses(object):
    """Few-shot task with episode-local labels; separate tasks need not share semantics."""

    def __init__(self, data, class_list, shot_num, query_num):
        self.data = data
        self.class_list = list(class_list)
        self.num_classes = len(self.class_list)
        self.support_num = shot_num
        self.query_num = query_num
        labels_map = dict(zip(self.class_list, range(self.num_classes)))
        self.support_datas, self.query_datas = [], []
        self.support_labels, self.query_labels = [], []
        for c in self.class_list:
            temp = self.data[c]
            samples = random.sample(temp, len(temp))
            random.shuffle(samples)
            self.support_datas += samples[:shot_num]
            self.query_datas += samples[shot_num : shot_num + query_num]
            self.support_labels += [labels_map[c]] * shot_num
            self.query_labels += [labels_map[c]] * query_num


def get_loader_for_task(task, num_per_class, split="train", shuffle=False):
    dataset = EpisodicDataset(task, split=split)
    if split == "train":
        sampler = ClassBalancedSampler(
            num_per_class, task.num_classes, task.support_num, shuffle=shuffle
        )
    else:
        sampler = ClassBalancedSampler(
            num_per_class, task.num_classes, task.query_num, shuffle=shuffle
        )
    return torch.utils.data.DataLoader(
        dataset, batch_size=num_per_class * task.num_classes, sampler=sampler
    )


def virtual_domain_shift_numpy(patch_chw, strength=1.0, gaussian_std=0.01):
    """Spectral scaling + noise on (C,H,W) numpy float with controllable intensity."""
    x = np.transpose(patch_chw, (1, 2, 0))
    x = radiation_noise(x)
    x = x.astype(np.float32)
    ch = x.shape[2]
    scale = np.random.normal(loc=1.0, scale=0.10 * strength, size=(1, 1, ch)).astype(np.float32)
    shift = np.random.normal(loc=0.0, scale=0.05 * strength, size=(1, 1, ch)).astype(np.float32)
    x = x * scale + shift
    if gaussian_std > 0:
        x = x + np.random.normal(0.0, gaussian_std * strength, size=x.shape).astype(np.float32)
    return np.transpose(x, (2, 0, 1)).astype(np.float32)
