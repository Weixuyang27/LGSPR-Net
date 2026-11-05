"""Train DFSL once and evaluate its NN and SVM target classifiers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from utils import data as hsi
from models.dfsl import DFSLFeatureEncoder, initialize_official, prototype_logits
from utils.splits import load_or_create_split
from utils.cases import CASE_NAMES, case_spec, load_case as load_shared_case, transfer_name


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "ac459bf75485dacec0b6f94aa80f8e505872547c"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASE_NAMES, required=True)
    parser.add_argument("--data_root", type=Path, default=ROOT / "Datasets")
    parser.add_argument("--target_split", type=Path)
    parser.add_argument("--split_seed", type=int, default=1337)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--source_sample_seed", type=int, default=0)
    parser.add_argument("--labeled_per_class", type=int, default=5)
    parser.add_argument("--source_max_per_class", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--episode_n_way", type=int, default=0)
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--query", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--max_test_per_class", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()
    args.episode_n_way = args.episode_n_way or case_spec(args.case)["episode_n_way"]
    args.query = args.query or case_spec(args.case)["query"]
    positive = (args.labeled_per_class, args.source_max_per_class, args.episodes,
                args.episode_n_way, args.shot, args.query, args.eval_batch_size,
                args.log_interval, args.threads)
    if min(positive) < 1 or args.lr <= 0 or args.max_test_per_class < 0:
        parser.error("Invalid training option")
    suffix = f"{args.case}_{args.labeled_per_class}shot_split{args.split_seed}"
    if args.max_test_per_class: suffix += f"_DEBUG{args.max_test_per_class}"
    args.target_split = args.target_split or ROOT / "splits" / (suffix + ".npz")
    run_name = f"seed{args.seed}" + (f"_debug{args.max_test_per_class}" if args.max_test_per_class else "")
    args.output_dir = args.output_dir or ROOT / "results/comparisons" / transfer_name(args.case) / "dfsl" / run_name
    return args


def load_case(args):
    # The released DFSL network is structurally fixed to 100 bands.
    return load_shared_case(args.data_root, args.case, 100)


class PatchStore:
    half = 4

    def __init__(self, cube, gt):
        self.gt = np.asarray(gt, dtype=np.int64)
        self.shape = self.gt.shape
        self.cube = np.pad(np.asarray(cube, dtype=np.float32),
                           ((self.half, self.half), (self.half, self.half), (0, 0)),
                           mode="reflect")

    def patch(self, flat_index):
        row, col = np.unravel_index(int(flat_index), self.shape)
        row += self.half; col += self.half
        value = self.cube[row-self.half:row+self.half+1,
                          col-self.half:col+self.half+1]
        return np.ascontiguousarray(value.transpose(2, 0, 1), dtype=np.float32)

    def label(self, flat_index):
        return int(self.gt.ravel()[int(flat_index)]) - 1


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, store, indices):
        self.store = store; self.indices = np.asarray(indices, dtype=np.int64)
    def __len__(self): return len(self.indices)
    def __getitem__(self, item):
        index = int(self.indices[item])
        return torch.from_numpy(self.store.patch(index)), self.store.label(index)


def source_pool(store, maximum, seed):
    rng, result = np.random.RandomState(seed), {}
    for label in np.unique(store.gt[store.gt > 0]):
        indices = np.flatnonzero(store.gt.ravel() == label)
        if len(indices) > maximum:
            indices = rng.choice(indices, maximum, replace=False)
        result[int(label) - 1] = indices.tolist()
    return result


def sample_episode(store, pool, eligible, n_way, shot, query):
    sx, sy, qx, qy = [], [], [], []
    for local, class_id in enumerate(random.sample(eligible, n_way)):
        selected = random.sample(pool[class_id], shot + query)
        sx.extend(store.patch(i) for i in selected[:shot])
        qx.extend(store.patch(i) for i in selected[shot:])
        sy.extend([local] * shot); qy.extend([local] * query)
    return (torch.from_numpy(np.stack(sx)), torch.tensor(sy),
            torch.from_numpy(np.stack(qx)), torch.tensor(qy))


@torch.no_grad()
def extract(encoder, loader, device):
    encoder.eval(); features, labels = [], []
    for x, y in loader:
        features.append(encoder(x.to(device)).cpu().numpy()); labels.append(y.numpy())
    return np.concatenate(features), np.concatenate(labels)


def metric_record(truth, prediction, classes):
    matrix = confusion_matrix(truth, prediction, labels=np.arange(classes))
    per_class = 100.0 * matrix.diagonal() / matrix.sum(1)
    return {"oa": float(100.0 * np.mean(truth == prediction)),
            "aa": float(per_class.mean()),
            "kappa": float(cohen_kappa_score(truth, prediction)),
            "each_class_accuracy": per_class.tolist(),
            "confusion_matrix": matrix.tolist()}


def svm_cv_folds(labels):
    """Retain four-fold CV where feasible, using training labels only."""
    _, counts = np.unique(labels, return_counts=True)
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("DFSL+SVM cross-validation needs >=2 labels per class")
    return min(4, int(counts.min()))


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists; choose a new --output_dir: {args.output_dir}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    source_cube, source_gt, target_cube, target_gt, classes = load_case(args)
    support_indices, test_indices = load_or_create_split(
        args.target_split, target_gt, args.labeled_per_class,
        args.split_seed, args.max_test_per_class)
    source_store, target_store = PatchStore(source_cube, source_gt), PatchStore(target_cube, target_gt)
    del source_cube, target_cube
    pool = source_pool(source_store, args.source_max_per_class, args.source_sample_seed)
    eligible = sorted(k for k, values in pool.items()
                      if len(values) >= args.shot + args.query)
    if len(eligible) < args.episode_n_way:
        raise ValueError("Not enough source classes for the requested episode")
    support_loader = torch.utils.data.DataLoader(
        PatchDataset(target_store, support_indices), batch_size=len(support_indices), shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        PatchDataset(target_store, test_indices), batch_size=args.eval_batch_size, shuffle=False)

    encoder = DFSLFeatureEncoder().to(device); encoder.apply(initialize_official)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(method="DFSL", classifiers=["1-NN", "RBF-SVM"],
                  architecture="official_residual_3dcnn_160d",
                  source_dataset=case_spec(args.case)["source"],
                  target_dataset=case_spec(args.case)["target"],
                  protocol="source_episodic_meta_train_then_fixed_target_classifier",
                  target_classes=classes, input_bands=100, patch_size=9,
                  normalization="whole_image_bandwise_scale_then_first_100_bands",
                  padding="reflect", checkpoint_selection="fixed_final_episode",
                  target_augmentation=False, target_train_count=len(support_indices),
                  test_count=len(test_indices),
                  source_samples_per_class={str(k): len(v) for k, v in pool.items()},
                  split_sha256=hashlib.sha256(args.target_split.read_bytes()).hexdigest(),
                  svm_kernel="rbf", svm_cv=svm_cv_folds(target_gt.ravel()[support_indices]),
                  svm_C_grid=[2.0 ** p for p in range(-2, 9)],
                  svm_gamma_grid=[2.0 ** p for p in range(-2, 9)],
                  feature_scaling="support_fitted_global_minmax",
                  deterministic_algorithms="warn_only_maxpool3d_cuda",
                  cudnn_benchmark=False, cudnn_deterministic=True,
                  torch_version=str(torch.__version__), python_version=sys.version,
                  python_executable=sys.executable, cuda_version=torch.version.cuda,
                  cudnn_version=torch.backends.cudnn.version(),
                  gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  implementation_source="gokling1219/DFSL", upstream_commit=UPSTREAM_COMMIT,
                  adaptations=("project source-target pairs; shared fixed target splits; "
                               "5000 project episodes; source-only meta-training; final-only evaluation"))
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[DFSL] {args.case} | {args.episode_n_way}-way {args.shot}-shot/"
          f"{args.query}-query | classifiers=NN,SVM | device={device}", flush=True)
    print(f"[split] target train={len(support_indices)} test={len(test_indices)} "
          f"file={args.target_split}", flush=True)

    encoder.train(); history, correct, total, start = [], 0, 0, time.perf_counter()
    for episode in range(1, args.episodes + 1):
        sx, sy, qx, qy = sample_episode(
            source_store, pool, eligible, args.episode_n_way, args.shot, args.query)
        sy, qy = sy.to(device), qy.to(device)
        support = encoder(sx.to(device)); query = encoder(qx.to(device))
        logits = prototype_logits(query, support, args.episode_n_way, args.shot)
        loss = torch.nn.functional.cross_entropy(logits, qy)
        if not torch.isfinite(loss): raise RuntimeError(f"Nonfinite loss at episode {episode}")
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        correct += int((logits.argmax(1) == qy).sum()); total += len(qy)
        if episode == 1 or episode % args.log_interval == 0 or episode == args.episodes:
            row = {"episode": episode, "loss": float(loss.item()),
                   "accuracy": correct / total}
            history.append(row)
            print(f"ep {episode} loss={row['loss']:.6f} acc={row['accuracy']:.4f}", flush=True)
    train_seconds = time.perf_counter() - start
    torch.save({"encoder": encoder.state_dict(), "config": config,
                "episode": args.episodes}, args.output_dir / "final.pt")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    support_feature, support_label = extract(encoder, support_loader, device)
    test_feature, truth = extract(encoder, test_loader, device)
    minimum, maximum = support_feature.min(), support_feature.max()
    scale = max(float(maximum - minimum), np.finfo(np.float32).eps)
    support_feature = (support_feature - minimum) / scale
    test_feature = (test_feature - minimum) / scale
    # DFSL+NN: the paper's nearest-neighbor classifier over the five supports/class.
    nn_parts = []
    for start_index in range(0, len(test_feature), 4096):
        batch = test_feature[start_index:start_index + 4096]
        distance = ((batch[:, None, :] - support_feature[None, :, :]) ** 2).sum(2)
        nn_parts.append(support_label[distance.argmin(1)])
    nn_prediction = np.concatenate(nn_parts)
    # Preserve the released C/gamma grid; cap folds by the training label budget.
    grid = {"C": [2.0 ** p for p in range(-2, 9)],
            "gamma": [2.0 ** p for p in range(-2, 9)]}
    svm = GridSearchCV(SVC(kernel="rbf"), grid, cv=svm_cv_folds(support_label), refit=True,
                       n_jobs=args.threads)
    svm.fit(support_feature, support_label)
    svm_prediction = svm.predict(test_feature)
    common = {"config": config, "train_seconds": train_seconds,
              "selected_episode": args.episodes, "selected_checkpoint": "final.pt"}
    nn_metrics = {**metric_record(truth, nn_prediction, classes), **common,
                  "classifier": "1-NN"}
    svm_metrics = {**metric_record(truth, svm_prediction, classes), **common,
                   "classifier": "RBF-SVM", "svm_best_params": svm.best_params_,
                   "svm_best_cv_score": float(svm.best_score_)}
    (args.output_dir / "metrics_nn.json").write_text(json.dumps(nn_metrics, indent=2), encoding="utf-8")
    (args.output_dir / "metrics_svm.json").write_text(json.dumps(svm_metrics, indent=2), encoding="utf-8")
    np.savez_compressed(args.output_dir / "predictions_nn.npz", flat_indices=test_indices,
                        truth=truth + 1, prediction=nn_prediction + 1, image_shape=target_gt.shape)
    np.savez_compressed(args.output_dir / "predictions_svm.npz", flat_indices=test_indices,
                        truth=truth + 1, prediction=svm_prediction + 1, image_shape=target_gt.shape)
    print(f"[DFSL+NN] OA={nn_metrics['oa']:.2f}% AA={nn_metrics['aa']:.2f}% "
          f"Kappa={nn_metrics['kappa']:.4f}", flush=True)
    print(f"[DFSL+SVM] OA={svm_metrics['oa']:.2f}% AA={svm_metrics['aa']:.2f}% "
          f"Kappa={svm_metrics['kappa']:.4f} best={svm.best_params_}", flush=True)
    print(f"[saved] {args.output_dir}", flush=True)


if __name__ == "__main__": main()
