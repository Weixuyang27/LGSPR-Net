"""Train RN-FSC with source meta-training and target 5-shot fine-tuning."""
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

from utils import data as hsi
from models.rnfsc import (
    RNFeatureEncoder,
    RelationNetwork,
    initialize_official,
    relation_mse,
    relation_scores,
)
from utils.splits import load_or_create_split
from utils.cases import CASE_NAMES, case_spec, load_case as load_shared_case, transfer_name


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "e971e999a003be67f82d5165bc84f5845d3f399e"


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
    parser.add_argument("--source_episodes", type=int, default=5000)
    parser.add_argument("--finetune_episodes", type=int, default=1000)
    parser.add_argument("--source_n_way", type=int, default=0,
                        help="0 selects 9 for case1/3 and 8 for case2/4")
    parser.add_argument("--source_query", type=int, default=0,
                        help="0 selects 19 for case1/3 and 4 for case2/4")
    parser.add_argument("--target_n_way", type=int, default=0,
                        help="0 uses all target classes")
    parser.add_argument("--target_query", type=int, default=4)
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--source_lr_step", type=int, default=5000)
    parser.add_argument("--finetune_lr_step", type=int, default=700)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--max_test_per_class", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()
    if args.source_n_way == 0:
        args.source_n_way = case_spec(args.case)["episode_n_way"]
    if args.source_query == 0:
        args.source_query = case_spec(args.case)["query"]
    target_classes = case_spec(args.case)["target_classes"]
    if args.target_n_way == 0:
        args.target_n_way = target_classes
    positive = (args.labeled_per_class, args.source_max_per_class,
                args.source_episodes, args.finetune_episodes, args.source_n_way,
                args.source_query, args.target_n_way, args.target_query, args.shot,
                args.source_lr_step, args.finetune_lr_step, args.eval_batch_size,
                args.log_interval, args.threads)
    if min(positive) < 1 or args.lr <= 0 or not 0 < args.lr_gamma <= 1:
        parser.error("Invalid training option")
    if args.shot + args.target_query > args.labeled_per_class:
        parser.error("RN-FSC target fine-tuning uses only the real target support set")
    if args.target_n_way > target_classes:
        parser.error("target_n_way exceeds the target class count")
    suffix = f"{args.case}_{args.labeled_per_class}shot_split{args.split_seed}"
    if args.max_test_per_class:
        suffix += f"_DEBUG{args.max_test_per_class}"
    args.target_split = args.target_split or ROOT / "splits" / (suffix + ".npz")
    run_name = f"seed{args.seed}" + (f"_debug{args.max_test_per_class}" if args.max_test_per_class else "")
    args.output_dir = args.output_dir or ROOT / "results/comparisons" / transfer_name(args.case) / "rnfsc" / run_name
    return args


def load_case(args):
    # RN-FSC is defined for 100 spectral bands. We retain the project's
    # whole-image bandwise standardization, then keep the first 100 bands.
    return load_shared_case(args.data_root, args.case, 100)


class PatchStore:
    """Reflect-padded 28x28 patch extraction used by RN-FSC."""
    half = 14

    def __init__(self, cube, gt):
        self.gt = np.asarray(gt, dtype=np.int64)
        self.shape = self.gt.shape
        self.cube = np.pad(np.asarray(cube, dtype=np.float32),
                           ((self.half, self.half), (self.half, self.half), (0, 0)),
                           mode="reflect")

    def patch(self, flat_index):
        row, column = np.unravel_index(int(flat_index), self.shape)
        row += self.half
        column += self.half
        value = self.cube[row - self.half:row + self.half,
                          column - self.half:column + self.half, :]
        return np.ascontiguousarray(value.transpose(2, 0, 1), dtype=np.float32)

    def label(self, flat_index):
        return int(self.gt.ravel()[int(flat_index)]) - 1


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, store, indices):
        self.store = store
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        return torch.from_numpy(self.store.patch(index)), self.store.label(index)


class EpisodicPool:
    def __init__(self, store, indices_by_class):
        self.store = store
        self.indices = {int(key): [int(x) for x in values]
                        for key, values in indices_by_class.items()}

    @property
    def classes(self):
        return sorted(self.indices)

    def count(self, class_id):
        return len(self.indices[class_id])

    def sample(self, class_id, count):
        selected = random.sample(self.indices[class_id], count)
        return [self.store.patch(index) for index in selected]


def build_source_pool(store, max_per_class, seed):
    rng = np.random.RandomState(seed)
    values = {}
    for label in np.unique(store.gt[store.gt > 0]):
        indices = np.flatnonzero(store.gt.ravel() == label)
        if len(indices) > max_per_class:
            indices = rng.choice(indices, max_per_class, replace=False)
        values[int(label) - 1] = indices.tolist()
    return EpisodicPool(store, values)


def build_target_pool(store, support_indices, classes):
    labels = store.gt.ravel()[support_indices] - 1
    values = {class_id: support_indices[labels == class_id].tolist()
              for class_id in range(classes)}
    return EpisodicPool(store, values)


def sample_episode(pool, eligible, n_way, shot, query):
    chosen_classes = random.sample(eligible, n_way)
    support, queries, support_labels, query_labels = [], [], [], []
    for local_label, class_id in enumerate(chosen_classes):
        patches = pool.sample(class_id, shot + query)
        support.extend(patches[:shot])
        queries.extend(patches[shot:])
        support_labels.extend([local_label] * shot)
        query_labels.extend([local_label] * query)
    return (torch.from_numpy(np.stack(support)),
            torch.tensor(support_labels, dtype=torch.long),
            torch.from_numpy(np.stack(queries)),
            torch.tensor(query_labels, dtype=torch.long))


def train_stage(name, encoder, relation, pool, n_way, shot, query, episodes,
                lr, step_size, gamma, log_interval, device):
    eligible = sorted(class_id for class_id in pool.classes
                      if pool.count(class_id) >= shot + query)
    if len(eligible) < n_way:
        raise ValueError(f"{name}: only {len(eligible)} classes have at least "
                         f"{shot + query} samples; requested {n_way}-way")
    encoder.train()
    relation.train()
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    relation_optimizer = torch.optim.Adam(relation.parameters(), lr=lr)
    encoder_scheduler = torch.optim.lr_scheduler.StepLR(
        encoder_optimizer, step_size=step_size, gamma=gamma)
    relation_scheduler = torch.optim.lr_scheduler.StepLR(
        relation_optimizer, step_size=step_size, gamma=gamma)
    history, correct, total = [], 0, 0
    start = time.perf_counter()
    for episode in range(1, episodes + 1):
        support, _, queries, labels = sample_episode(
            pool, eligible, n_way, shot, query)
        support_features = encoder(support.to(device))
        query_features = encoder(queries.to(device))
        labels = labels.to(device)
        scores = relation_scores(relation, support_features, query_features, n_way, shot)
        loss = relation_mse(scores, labels, n_way)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name}: nonfinite loss at episode {episode}")
        encoder_optimizer.zero_grad(set_to_none=True)
        relation_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(relation.parameters(), 0.5)
        encoder_optimizer.step()
        relation_optimizer.step()
        encoder_scheduler.step()
        relation_scheduler.step()
        correct += int((scores.argmax(1) == labels).sum())
        total += len(labels)
        if episode == 1 or episode % log_interval == 0 or episode == episodes:
            row = {"stage": name, "episode": episode, "loss": float(loss.item()),
                   "accuracy": correct / total,
                   "lr": encoder_optimizer.param_groups[0]["lr"]}
            history.append(row)
            print(f"[{name}] ep {episode} loss={row['loss']:.6f} "
                  f"acc={row['accuracy']:.4f} lr={row['lr']:.6g}", flush=True)
    return history, time.perf_counter() - start


@torch.no_grad()
def evaluate(encoder, relation, support_loader, test_loader, device, classes):
    encoder.eval()
    relation.eval()
    support_x, support_y = next(iter(support_loader))
    order = torch.argsort(support_y)
    support_x, support_y = support_x[order].to(device), support_y[order]
    expected = torch.arange(classes).repeat_interleave(len(support_y) // classes)
    if not torch.equal(support_y, expected):
        raise ValueError("Target support labels are not balanced/contiguous")
    support_features = encoder(support_x)
    shot = len(support_y) // classes
    truth, prediction = [], []
    for x, y in test_loader:
        query_features = encoder(x.to(device))
        scores = relation_scores(relation, support_features, query_features, classes, shot)
        prediction.append(scores.argmax(1).cpu().numpy())
        truth.append(y.numpy())
    truth, prediction = np.concatenate(truth), np.concatenate(prediction)
    matrix = confusion_matrix(truth, prediction, labels=np.arange(classes))
    per_class = 100.0 * matrix.diagonal() / matrix.sum(1)
    return {
        "oa": float(100.0 * np.mean(truth == prediction)),
        "aa": float(per_class.mean()),
        "kappa": float(cohen_kappa_score(truth, prediction)),
        "each_class_accuracy": per_class.tolist(),
        "confusion_matrix": matrix.tolist(),
    }, truth, prediction


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists; choose a new --output_dir: {args.output_dir}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
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
    source_pool = build_source_pool(
        source_store, args.source_max_per_class, args.source_sample_seed)
    target_pool = build_target_pool(target_store, support_indices, classes)
    support_loader = torch.utils.data.DataLoader(
        PatchDataset(target_store, support_indices), batch_size=len(support_indices), shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        PatchDataset(target_store, test_indices), batch_size=args.eval_batch_size, shuffle=False)

    encoder, relation = RNFeatureEncoder(), RelationNetwork()
    encoder.apply(initialize_official)
    relation.apply(initialize_official)
    encoder, relation = encoder.to(device), relation.to(device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(
        method="RN-FSC", architecture="official_deep_3dcnn_relation_network",
        source_dataset=case_spec(args.case)["source"],
        target_dataset=case_spec(args.case)["target"],
        protocol="source_meta_train_then_target_5shot_finetune",
        target_classes=classes, input_bands=100, patch_size=28,
        normalization="whole_image_bandwise_scale_then_first_100_bands",
        padding="reflect", checkpoint_selection="fixed_final_finetune_episode",
        target_augmentation=False, source_classes_total=len(source_pool.classes),
        source_samples_per_class={str(k): source_pool.count(k) for k in source_pool.classes},
        target_train_count=len(support_indices), test_count=len(test_indices),
        split_sha256=hashlib.sha256(args.target_split.read_bytes()).hexdigest(),
        deterministic_algorithms="warn_only_maxpool3d_cuda",
        cudnn_benchmark=False, cudnn_deterministic=True,
        torch_version=str(torch.__version__), python_version=sys.version,
        python_executable=sys.executable, cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        implementation_source="gokling1219/RN-FSC", upstream_commit=UPSTREAM_COMMIT,
        released_code_note=("the public source pretraining script uses 8-16-32-64 channels, "
                            "but the released fine-tuning/test scripts expect 16-32-64-128 and "
                            "load a missing newmodel2 checkpoint; this adapter consistently uses "
                            "the published fine-tuning/test architecture in both stages"),
        adaptations=("project source-target pairs; shared fixed target splits; project bandwise "
                     "normalization; source 5000 episodes; final-only evaluation"),
    )
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[RN-FSC] {args.case} | source {args.source_n_way}-way "
          f"{args.shot}-shot/{args.source_query}-query -> target {args.target_n_way}-way "
          f"{args.shot}-shot/{args.target_query}-query | device={device}", flush=True)
    print(f"[split] target train={len(support_indices)} test={len(test_indices)} "
          f"file={args.target_split}", flush=True)
    print(f"[input] 100 bands | 28x28 reflect-padded patches", flush=True)

    source_history, source_seconds = train_stage(
        "source", encoder, relation, source_pool, args.source_n_way, args.shot,
        args.source_query, args.source_episodes, args.lr, args.source_lr_step,
        args.lr_gamma, args.log_interval, device)
    torch.save({"encoder": encoder.state_dict(), "relation": relation.state_dict(),
                "config": config, "episode": args.source_episodes},
               args.output_dir / "source_pretrained.pt")
    target_history, target_seconds = train_stage(
        "target", encoder, relation, target_pool, args.target_n_way, args.shot,
        args.target_query, args.finetune_episodes, args.lr, args.finetune_lr_step,
        args.lr_gamma, args.log_interval, device)
    torch.save({"encoder": encoder.state_dict(), "relation": relation.state_dict(),
                "config": config, "episode": args.finetune_episodes},
               args.output_dir / "final.pt")
    (args.output_dir / "history.json").write_text(
        json.dumps(source_history + target_history, indent=2), encoding="utf-8")
    metrics, truth, prediction = evaluate(
        encoder, relation, support_loader, test_loader, device, classes)
    metrics.update(config=config, source_train_seconds=source_seconds,
                   target_finetune_seconds=target_seconds,
                   selected_episode=args.finetune_episodes,
                   selected_checkpoint="final.pt")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(args.output_dir / "predictions.npz", flat_indices=test_indices,
                        truth=truth + 1, prediction=prediction + 1,
                        image_shape=target_gt.shape)
    print(f"[final] OA={metrics['oa']:.2f}% AA={metrics['aa']:.2f}% "
          f"Kappa={metrics['kappa']:.4f}", flush=True)
    print(f"[saved] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
