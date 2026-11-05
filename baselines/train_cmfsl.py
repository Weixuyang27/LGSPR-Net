"""Train CMFSL under the project's shared five-shot comparison protocol."""
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
from models.cmfsl import (CMFSLNetwork, build_ccm_statistics, ccm_logits,
                              initialize_official)
from utils.splits import fixed_target_pack
from utils.cases import CASE_NAMES, case_spec, load_case as load_shared_case, transfer_name


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "62090e351208904c339ad45d9daaea3054535737"


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
    parser.add_argument("--target_aug_per_class", type=int, default=200)
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
    positive = (args.labeled_per_class, args.source_max_per_class,
                args.target_aug_per_class, args.episodes, args.episode_n_way,
                args.shot, args.query, args.eval_batch_size, args.log_interval,
                args.threads)
    if min(positive) < 1 or args.lr <= 0 or args.max_test_per_class < 0:
        parser.error("Invalid training option")
    suffix = f"{args.case}_{args.labeled_per_class}shot_split{args.split_seed}"
    if args.max_test_per_class: suffix += f"_DEBUG{args.max_test_per_class}"
    args.target_split = args.target_split or ROOT / "splits" / (suffix + ".npz")
    run_name = f"seed{args.seed}" + (f"_debug{args.max_test_per_class}" if args.max_test_per_class else "")
    args.output_dir = args.output_dir or ROOT / "results/comparisons" / transfer_name(args.case) / "cmfsl" / run_name
    return args


def load_case(args):
    # The LXConvNet reshape fixes its spectral input at exactly 100 bands.
    return load_shared_case(args.data_root, args.case, 100)


def augment_target_support(support_x, support_y, classes, per_class, rng):
    pools = {}
    for class_id in range(classes):
        originals = support_x[support_y == class_id].cpu().numpy()
        generated = []
        for index in range(per_class):
            patch = originals[index % len(originals)].astype(np.float32, copy=True)
            alpha = rng.uniform(0.9, 1.1)
            noise = rng.normal(0.0, 1.0, patch.shape).astype(np.float32)
            generated.append((alpha * patch + noise / 25.0).astype(np.float32))
        pools[class_id] = generated
    return pools


def sample_episode(pool, eligible, n_way, shot, query):
    support, support_y, queries, query_y = [], [], [], []
    for local, class_id in enumerate(random.sample(eligible, n_way)):
        selected = random.sample(pool[class_id], shot + query)
        support.extend(selected[:shot]); queries.extend(selected[shot:])
        support_y.extend([local] * shot); query_y.extend([local] * query)
    return (torch.from_numpy(np.stack(support).astype(np.float32)),
            torch.tensor(support_y, dtype=torch.long),
            torch.from_numpy(np.stack(queries).astype(np.float32)),
            torch.tensor(query_y, dtype=torch.long))


@torch.no_grad()
def evaluate(model, support_loader, test_loader, device, classes):
    model.eval()
    support_x, support_y = next(iter(support_loader))
    support_feature = model(support_x.to(device), "target")
    means, precision = build_ccm_statistics(support_feature, support_y.to(device))
    truth, prediction = [], []
    for x, y in test_loader:
        logits = ccm_logits(model(x.to(device), "target"), means, precision)
        prediction.append(logits.argmax(1).cpu().numpy()); truth.append(y.numpy())
    truth, prediction = np.concatenate(truth), np.concatenate(prediction)
    matrix = confusion_matrix(truth, prediction, labels=np.arange(classes))
    per_class = 100.0 * matrix.diagonal() / matrix.sum(1)
    return {"oa": float(100.0 * np.mean(truth == prediction)),
            "aa": float(per_class.mean()),
            "kappa": float(cohen_kappa_score(truth, prediction)),
            "each_class_accuracy": per_class.tolist(),
            "confusion_matrix": matrix.tolist()}, truth, prediction


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
    source_pool = hsi.build_metatrain_dict(
        source_cube, source_gt, max_per_class=args.source_max_per_class,
        half_width=4, seed=args.source_sample_seed)
    support_loader, raw_test_loader, *_ = fixed_target_pack(
        target_cube, target_gt, args.target_split, args.labeled_per_class,
        args.split_seed, args.max_test_per_class)
    del source_cube, target_cube
    support_x, support_y = next(iter(support_loader))
    target_pool = augment_target_support(
        support_x, support_y, classes, args.target_aug_per_class,
        np.random.RandomState(args.seed))
    required = args.shot + args.query
    source_eligible = sorted(k for k, v in source_pool.items() if len(v) >= required)
    target_eligible = sorted(k for k, v in target_pool.items() if len(v) >= required)
    if min(len(source_eligible), len(target_eligible)) < args.episode_n_way:
        raise ValueError("Not enough eligible classes for the requested episode")
    test_loader = torch.utils.data.DataLoader(
        raw_test_loader.dataset, batch_size=args.eval_batch_size, shuffle=False)

    model = CMFSLNetwork().to(device); model.apply(initialize_official); model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(method="CMFSL", architecture="official_SPRM_LXConvNet_CCM",
                  source_dataset=case_spec(args.case)["source"],
                  target_dataset=case_spec(args.case)["target"],
                  protocol="alternating_source_target_collaborative_meta_training",
                  target_classes=classes, input_bands=100, feature_dim=128,
                  patch_size=9, normalization="whole_image_bandwise_scale_then_first_100_bands",
                  padding="zero", checkpoint_selection="fixed_final_episode",
                  target_augmentation="official_radiation_noise_alpha_0.9_1.1_beta_1_over_25",
                  target_train_count=len(support_loader.dataset), test_count=len(test_loader.dataset),
                  source_samples_per_class={str(k): len(v) for k, v in source_pool.items()},
                  split_sha256=hashlib.sha256(args.target_split.read_bytes()).hexdigest(),
                  deterministic_algorithms="warn_only_pool3d_cuda",
                  cudnn_benchmark=False, cudnn_deterministic=True,
                  torch_version=str(torch.__version__), python_version=sys.version,
                  python_executable=sys.executable, cuda_version=torch.version.cuda,
                  cudnn_version=torch.backends.cudnn.version(),
                  gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  implementation_source="B-Xi/TIP_2022_CMFSL", upstream_commit=UPSTREAM_COMMIT,
                  adaptations=("project source-target pairs; shared fixed target splits; "
                               "case2 8-way/4-query; 5000 episodes; final-only evaluation"))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[CMFSL] {args.case} | alternating source/target | {args.episode_n_way}-way "
          f"{args.shot}-shot/{args.query}-query | device={device}", flush=True)
    print(f"[split] target train={len(support_loader.dataset)} test={len(test_loader.dataset)} "
          f"file={args.target_split}", flush=True)

    history, correct, total, start = [], 0, 0, time.perf_counter()
    for episode in range(1, args.episodes + 1):
        domain = "source" if episode % 2 == 1 else "target"
        pool = source_pool if domain == "source" else target_pool
        eligible = source_eligible if domain == "source" else target_eligible
        sx, sy, qx, qy = sample_episode(
            pool, eligible, args.episode_n_way, args.shot, args.query)
        sy, qy = sy.to(device), qy.to(device)
        support = model(sx.to(device), domain); query = model(qx.to(device), domain)
        means, precision = build_ccm_statistics(support, sy)
        logits = ccm_logits(query, means, precision)
        loss = torch.nn.functional.cross_entropy(logits, qy)
        if not torch.isfinite(loss): raise RuntimeError(f"Nonfinite loss at episode {episode}")
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        correct += int((logits.argmax(1) == qy).sum()); total += len(qy)
        if episode == 1 or episode % args.log_interval == 0 or episode == args.episodes:
            row = {"episode": episode, "domain": domain, "loss": float(loss.item()),
                   "accuracy": correct / total}
            history.append(row)
            print(f"ep {episode} domain={domain} loss={row['loss']:.6f} "
                  f"acc={row['accuracy']:.4f}", flush=True)

    train_seconds = time.perf_counter() - start
    torch.save({"model": model.state_dict(), "config": config,
                "episode": args.episodes}, args.output_dir / "final.pt")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    metrics, truth, prediction = evaluate(model, support_loader, test_loader, device, classes)
    metrics.update(config=config, train_seconds=train_seconds,
                   selected_episode=args.episodes, selected_checkpoint="final.pt")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with np.load(args.target_split, allow_pickle=False) as saved:
        test_indices = saved["test"]
    np.savez_compressed(args.output_dir / "predictions.npz", flat_indices=test_indices,
                        truth=truth + 1, prediction=prediction + 1,
                        image_shape=target_gt.shape)
    print(f"[final] OA={metrics['oa']:.2f}% AA={metrics['aa']:.2f}% "
          f"Kappa={metrics['kappa']:.4f}", flush=True)
    print(f"[saved] {args.output_dir}", flush=True)


if __name__ == "__main__": main()
