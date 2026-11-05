"""Run author's SSRN as a target-only few-label supervised baseline."""
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
from sklearn.metrics import confusion_matrix, cohen_kappa_score

from utils import data
from utils.splits import fixed_target_pack
from models.ssrn import SSRN
from utils.cases import CASE_NAMES, case_spec, load_target, transfer_name

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=CASE_NAMES, required=True)
    p.add_argument("--data_root", type=Path, default=ROOT / "Datasets")
    p.add_argument("--target_split", type=Path)
    p.add_argument("--split_seed", type=int, default=1337)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--labeled_per_class", type=int, default=5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--eval_interval", type=int, default=10)
    p.add_argument("--checkpoint_selection", choices=("best_target_oa", "fixed_final_epoch"),
                   default="fixed_final_epoch")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max_test_per_class", type=int, default=0, help="0 = full test; positive = debug only")
    p.add_argument("--output_dir", type=Path)
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.threads,
           args.labeled_per_class, args.eval_interval) < 1:
        p.error("epochs, batch sizes, threads and labeled_per_class must be positive")
    if args.lr <= 0 or args.weight_decay < 0 or args.max_test_per_class < 0:
        p.error("Invalid optimizer or test limit")
    suffix = f"{args.case}_{args.labeled_per_class}shot_split{args.split_seed}"
    if args.max_test_per_class:
        suffix += f"_DEBUG{args.max_test_per_class}"
    args.target_split = args.target_split or ROOT / "splits" / (suffix + ".npz")
    run_name = f"seed{args.seed}" + (f"_debug{args.max_test_per_class}" if args.max_test_per_class else "")
    args.output_dir = args.output_dir or ROOT / "results/comparisons" / transfer_name(args.case) / "ssrn" / run_name
    return args


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    truth, pred = [], []
    for x, y in loader:
        pred.append(model(x.to(device)).argmax(1).cpu().numpy())
        truth.append(y.numpy())
    truth, pred = np.concatenate(truth), np.concatenate(pred)
    cm = confusion_matrix(truth, pred, labels=np.arange(n_classes))
    each = 100.0 * cm.diagonal() / cm.sum(axis=1)
    return dict(oa=float(100 * np.mean(truth == pred)), aa=float(each.mean()),
                kappa=float(cohen_kappa_score(truth, pred)),
                each_class_accuracy=each.tolist(), confusion_matrix=cm.tolist()), truth, pred


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists; choose a new --output_dir: {args.output_dir}")
    # Match the deterministic protocol used by the current main experiments.
    # Setting CUBLAS_WORKSPACE_CONFIG before the first CUDA operation makes a
    # seed identify a reproducible run instead of only a reproducible split.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    bands = 100 if args.case in ("case3", "case4") else (102 if args.case == "case1" else 200)
    classes = case_spec(args.case)["target_classes"]
    print(f"[SSRN] {args.case}, device={device}, target-only, {args.labeled_per_class} labels/class", flush=True)
    cube, gt, _ = load_target(args.data_root, args.case, bands)
    pack = fixed_target_pack(cube, gt, args.target_split, args.labeled_per_class,
                             args.split_seed, args.max_test_per_class)
    del cube
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(pack[0].dataset, batch_size=args.batch_size,
                                               shuffle=True, generator=generator)
    test_loader = torch.utils.data.DataLoader(pack[1].dataset, batch_size=args.eval_batch_size)
    model = SSRN(bands, classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(n_bands=bands, n_classes=classes, patch_size=9, device=str(device),
                  torch_version=str(torch.__version__), protocol="target_only_supervised",
                  normalization="whole_image_bandwise_scale",
                  padding="zero", source_training=False, augmentation=False,
                  deterministic_algorithms=True,
                  implementation_source="zilongzhong/SSTN:NetworksBlocks.py",
                  upstream_commit="1b377d9c1157d37c17c03ca17aaa3b655378d547",
                  python_version=sys.version,
                  python_executable=sys.executable,
                  cuda_version=torch.version.cuda,
                  cudnn_version=torch.backends.cudnn.version(),
                  gpu_name=(torch.cuda.get_device_name(device) if device.type == "cuda" else None),
                  split_sha256=hashlib.sha256(args.target_split.read_bytes()).hexdigest(),
                  debug=bool(args.max_test_per_class), train_count=len(train_loader.dataset),
                  test_count=len(test_loader.dataset))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[split] train={config['train_count']} test={config['test_count']} file={args.target_split}", flush=True)
    start = time.perf_counter()
    history = []
    best_oa, best_epoch = float("-inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        row = dict(epoch=epoch, loss=total_loss / len(train_loader.dataset))
        # The paper5_v1 protocol uses the fixed final epoch, so it does not
        # repeatedly inspect the target test set during training. Periodic
        # evaluation is retained only for the explicitly requested legacy
        # best-target-OA mode.
        should_evaluate = (
            args.checkpoint_selection == "best_target_oa"
            and (epoch % args.eval_interval == 0 or epoch == args.epochs)
        )
        if should_evaluate:
            epoch_metrics, _, _ = evaluate(model, test_loader, device, classes)
            row.update(eval_oa=epoch_metrics["oa"], eval_aa=epoch_metrics["aa"],
                       eval_kappa=epoch_metrics["kappa"])
            if epoch_metrics["oa"] > best_oa:
                best_oa, best_epoch = epoch_metrics["oa"], epoch
                torch.save(dict(model=model.state_dict(), config=config, epoch=epoch),
                           args.output_dir / "best.pt")
                print(f"[eval] epoch={epoch} OA={best_oa:.2f}% new best", flush=True)
            else:
                print(f"[eval] epoch={epoch} OA={epoch_metrics['oa']:.2f}%", flush=True)
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"[train] epoch={epoch}/{args.epochs} loss={row['loss']:.6f}", flush=True)
    train_seconds = time.perf_counter() - start
    torch.save(dict(model=model.state_dict(), config=config, epoch=args.epochs), args.output_dir / "final.pt")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    selected_epoch = args.epochs
    selected_checkpoint = args.output_dir / "final.pt"
    if args.checkpoint_selection == "best_target_oa":
        selected_epoch = best_epoch
        selected_checkpoint = args.output_dir / "best.pt"
        selected = torch.load(selected_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(selected["model"])
    metrics, truth, pred = evaluate(model, test_loader, device, classes)
    metrics.update(config=config, train_seconds=train_seconds,
                   selected_epoch=selected_epoch,
                   selected_checkpoint=selected_checkpoint.name)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with np.load(args.target_split, allow_pickle=False) as saved:
        test_indices = saved["test"]
    # Predictions use original dataset labels 1..C; confusion matrix rows/columns follow 1..C.
    np.savez_compressed(args.output_dir / "predictions.npz", flat_indices=test_indices,
                        truth=truth + 1, prediction=pred + 1, image_shape=gt.shape)
    label = "final" if args.checkpoint_selection == "fixed_final_epoch" else "selected"
    print(f"[{label}] epoch={selected_epoch} OA={metrics['oa']:.2f}% "
          f"AA={metrics['aa']:.2f}% Kappa={metrics['kappa']:.4f}", flush=True)
    print(f"[saved] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
