"""Train the official-style DCFSL baseline under this project's shared protocol."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from utils import data as hsi
from models.dcfsl import (
    DCFSLNetwork,
    DomainClassifier,
    RandomizedMultilinearMap,
    euclidean_logits,
    grl_coefficient,
    initialize_official,
)
from utils.splits import fixed_target_pack
from utils.cases import CASE_NAMES, case_spec, load_case as load_shared_case, transfer_name


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "ec92b1cca5ce3779f0cec4435daa714f902e07e3"


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
    parser.add_argument("--episode_n_way", type=int, default=0,
                        help="0 selects 9 for case1/3 and 8 for case2/4")
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--query", type=int, default=0,
                        help="0 selects 19 for case1/3 and 4 for case2/4")
    parser.add_argument("--mapping_dim", type=int, default=100)
    parser.add_argument("--condition_dim", type=int, default=1024)
    parser.add_argument("--domain_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grl_max_iter", type=int, default=10000)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--max_test_per_class", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()
    if args.episode_n_way == 0:
        args.episode_n_way = case_spec(args.case)["episode_n_way"]
    if args.query == 0:
        args.query = case_spec(args.case)["query"]
    positive = (args.labeled_per_class, args.source_max_per_class,
                args.target_aug_per_class, args.episodes, args.episode_n_way,
                args.shot, args.query, args.mapping_dim, args.condition_dim,
                args.domain_batch_size, args.eval_batch_size, args.grl_max_iter,
                args.log_interval, args.threads)
    if min(positive) < 1 or args.lr <= 0 or args.weight_decay < 0 or args.max_test_per_class < 0:
        parser.error("Invalid nonpositive training option")
    suffix = f"{args.case}_{args.labeled_per_class}shot_split{args.split_seed}"
    if args.max_test_per_class:
        suffix += f"_DEBUG{args.max_test_per_class}"
    args.target_split = args.target_split or ROOT / "splits" / (suffix + ".npz")
    run_name = f"seed{args.seed}" + (f"_debug{args.max_test_per_class}" if args.max_test_per_class else "")
    args.output_dir = args.output_dir or ROOT / "results/comparisons" / transfer_name(args.case) / "dcfsl" / run_name
    return args


def load_case(args):
    bands = 100 if args.case in ("case3", "case4") else (102 if args.case == "case1" else 200)
    return load_shared_case(args.data_root, args.case, bands)


def augment_target_support(support_x, support_y, n_classes, per_class, rng):
    """Match DCFSL radiation-noise expansion using only the five labeled pixels."""
    pools = {}
    for class_id in range(n_classes):
        class_x = support_x[support_y == class_id].cpu().numpy()
        if len(class_x) == 0:
            raise ValueError(f"Target support has no class {class_id}")
        generated = []
        for index in range(per_class):
            patch = class_x[index % len(class_x)].astype(np.float32, copy=True)
            alpha = rng.uniform(0.9, 1.1)
            noise = rng.normal(0.0, 1.0, size=patch.shape).astype(np.float32)
            generated.append((alpha * patch + (1.0 / 25.0) * noise).astype(np.float32))
        pools[class_id] = generated
    return pools


def pool_as_dataset(pool):
    xs, ys = [], []
    for class_id in sorted(pool):
        xs.extend(pool[class_id])
        ys.extend([class_id] * len(pool[class_id]))
    return torch.utils.data.TensorDataset(
        torch.from_numpy(np.stack(xs).astype(np.float32)), torch.tensor(ys, dtype=torch.long))


def sample_episode(pool, eligible, n_way, shot, query):
    selected = random.sample(eligible, n_way)
    support, query_data, support_y, query_y = [], [], [], []
    for local_label, class_id in enumerate(selected):
        chosen = random.sample(pool[class_id], shot + query)
        support.extend(chosen[:shot])
        query_data.extend(chosen[shot:])
        support_y.extend([local_label] * shot)
        query_y.extend([local_label] * query)
    return (torch.from_numpy(np.stack(support).astype(np.float32)),
            torch.tensor(support_y, dtype=torch.long),
            torch.from_numpy(np.stack(query_data).astype(np.float32)),
            torch.tensor(query_y, dtype=torch.long))


def next_cycling(iterator, loader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


@torch.no_grad()
def evaluate(model, support_loader, test_loader, device, n_classes):
    model.eval()
    support_x, support_y = next(iter(support_loader))
    support_features, _ = model(support_x.to(device), "target")
    minimum, maximum = support_features.min(), support_features.max()
    scale = (maximum - minimum).clamp_min(torch.finfo(support_features.dtype).eps)
    support_features = (support_features - minimum) / scale
    truth, prediction = [], []
    for x, y in test_loader:
        feature, _ = model(x.to(device), "target")
        feature = (feature - minimum) / scale
        nearest = torch.cdist(feature, support_features).argmin(dim=1)
        prediction.append(support_y[nearest.cpu()].numpy())
        truth.append(y.numpy())
    truth, prediction = np.concatenate(truth), np.concatenate(prediction)
    cm = confusion_matrix(truth, prediction, labels=np.arange(n_classes))
    per_class = 100.0 * cm.diagonal() / cm.sum(axis=1)
    metrics = {
        "oa": float(100.0 * np.mean(truth == prediction)),
        "aa": float(per_class.mean()),
        "kappa": float(cohen_kappa_score(truth, prediction)),
        "each_class_accuracy": per_class.tolist(),
        "confusion_matrix": cm.tolist(),
    }
    return metrics, truth, prediction


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
    # The official DCFSL encoder contains MaxPool3d. Its CUDA backward kernel
    # has no strictly deterministic implementation in the current PyTorch
    # build, so strict=True would abort before the first optimizer step.
    # warn_only preserves all available deterministic settings without
    # replacing the published max-pooling operation.
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    source_cube, source_gt, target_cube, target_gt, n_classes = load_case(args)
    source_bands, target_bands = source_cube.shape[2], target_cube.shape[2]
    source_pool = hsi.build_metatrain_dict(
        source_cube, source_gt, max_per_class=args.source_max_per_class,
        half_width=4, seed=args.source_sample_seed)
    support_loader, raw_test_loader, *_ = fixed_target_pack(
        target_cube, target_gt, args.target_split, args.labeled_per_class,
        args.split_seed, args.max_test_per_class)
    del source_cube, target_cube
    support_x, support_y = next(iter(support_loader))
    target_pool = augment_target_support(
        support_x, support_y, n_classes, args.target_aug_per_class,
        np.random.RandomState(args.seed))
    source_eligible = sorted(k for k, values in source_pool.items()
                             if len(values) >= args.shot + args.query)
    target_eligible = sorted(k for k, values in target_pool.items()
                             if len(values) >= args.shot + args.query)
    if len(source_eligible) < args.episode_n_way:
        raise ValueError(
            f"Only {len(source_eligible)} source classes have at least "
            f"{args.shot + args.query} samples; episode_n_way={args.episode_n_way}")
    if len(target_eligible) < args.episode_n_way:
        raise ValueError("Not enough target classes for requested episode_n_way")

    source_generator = torch.Generator().manual_seed(args.seed + 101)
    target_generator = torch.Generator().manual_seed(args.seed + 202)
    source_loader = torch.utils.data.DataLoader(
        pool_as_dataset(source_pool), batch_size=args.domain_batch_size,
        shuffle=True, generator=source_generator)
    target_loader = torch.utils.data.DataLoader(
        pool_as_dataset(target_pool), batch_size=args.domain_batch_size,
        shuffle=True, generator=target_generator)
    test_loader = torch.utils.data.DataLoader(
        raw_test_loader.dataset, batch_size=args.eval_batch_size, shuffle=False)

    model = DCFSLNetwork(source_bands=source_bands,
                         target_bands=target_bands,
                         condition_classes=args.episode_n_way,
                         mapping_dim=args.mapping_dim)
    discriminator = DomainClassifier(args.condition_dim)
    conditional_map = RandomizedMultilinearMap(
        model.feature_dim, args.episode_n_way, args.condition_dim)
    # The released code creates the random conditioning matrices before
    # applying Xavier initialization to the trainable networks.
    model.apply(initialize_official)
    discriminator.apply(initialize_official)
    model, discriminator, conditional_map = (
        model.to(device), discriminator.to(device), conditional_map.to(device))
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    domain_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    domain_criterion = torch.nn.BCEWithLogitsLoss()

    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(
        method="DCFSL", architecture="official_mapping_dres3d_cdan",
        source_dataset=case_spec(args.case)["source"],
        target_dataset=case_spec(args.case)["target"],
        source_bands=source_bands, target_bands=target_bands,
        protocol="shared_cross_domain_few_shot", target_classes=n_classes,
        patch_size=9, normalization="whole_image_bandwise_scale",
        padding="zero", checkpoint_selection="fixed_final_episode",
        target_augmentation="official_radiation_noise_alpha_0.9_1.1_beta_1_over_25",
        source_classes_total=len(source_pool), source_classes_eligible=len(source_eligible),
        source_samples_per_class={str(k): len(v) for k, v in source_pool.items()},
        target_train_count=len(support_loader.dataset), test_count=len(test_loader.dataset),
        split_sha256=hashlib.sha256(args.target_split.read_bytes()).hexdigest(),
        deterministic_algorithms="warn_only_maxpool3d_cuda",
        cudnn_benchmark=False, cudnn_deterministic=True,
        torch_version=str(torch.__version__),
        python_version=sys.version, python_executable=sys.executable,
        cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
        gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        implementation_source="Li-ZK/DCFSL-2021",
        upstream_commit=UPSTREAM_COMMIT,
        adaptations=("project source-target pairs; shared fixed target splits; "
                     "case2 8-way/4-query; 5000 episodes; final-only evaluation"),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")
    print(f"[DCFSL] {args.case} source->target | {args.episode_n_way}-way "
          f"{args.shot}-shot/{args.query}-query | device={device}", flush=True)
    print(f"[split] target train={len(support_loader.dataset)} test={len(test_loader.dataset)} "
          f"file={args.target_split}", flush=True)
    print(f"[source] classes={len(source_pool)} eligible={len(source_eligible)} "
          f"samples={[len(source_pool[k]) for k in sorted(source_pool)]}", flush=True)

    source_iterator, target_iterator = iter(source_loader), iter(target_loader)
    correct_total = sample_total = 0
    history = []
    start = time.perf_counter()
    model.train()
    discriminator.train()
    for episode in range(args.episodes):
        (source_da, _), source_iterator = next_cycling(source_iterator, source_loader)
        (target_da, _), target_iterator = next_cycling(target_iterator, target_loader)
        if episode % 2 == 0:
            support, support_label, query, query_label = sample_episode(
                source_pool, source_eligible, args.episode_n_way, args.shot, args.query)
            domain = "source"
            opposite_data, opposite_domain = target_da, "target"
            source_count = len(support) + len(query)
        else:
            support, support_label, query, query_label = sample_episode(
                target_pool, target_eligible, args.episode_n_way, args.shot, args.query)
            domain = "target"
            opposite_data, opposite_domain = source_da, "source"
            source_count = len(opposite_data)
        support = support.to(device)
        query = query.to(device)
        support_label = support_label.to(device)
        query_label = query_label.to(device)
        opposite_data = opposite_data.to(device)
        support_feature, support_output = model(support, domain)
        query_feature, query_output = model(query, domain)
        opposite_feature, opposite_output = model(opposite_data, opposite_domain)
        prototypes = support_feature.reshape(
            args.episode_n_way, args.shot, -1).mean(dim=1)
        logits = euclidean_logits(query_feature, prototypes)
        fsl_loss = torch.nn.functional.cross_entropy(logits, query_label)

        features = torch.cat((support_feature, query_feature, opposite_feature), dim=0)
        outputs = torch.cat((support_output, query_output, opposite_output), dim=0)
        conditioned = conditional_map(features, torch.softmax(outputs, dim=1))
        domain_label = torch.zeros((len(features), 1), device=device)
        if episode % 2 == 0:
            domain_label[:source_count] = 1.0
        else:
            domain_label[-source_count:] = 1.0
        coefficient = grl_coefficient(episode, args.grl_max_iter)
        domain_logits = discriminator(conditioned, coefficient)
        domain_loss = domain_criterion(domain_logits, domain_label)
        loss = fsl_loss + domain_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at episode {episode + 1}")
        model_optimizer.zero_grad(set_to_none=True)
        domain_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        model_optimizer.step()
        domain_optimizer.step()
        correct_total += int((logits.argmax(1) == query_label).sum().item())
        sample_total += len(query_label)
        if (episode + 1) % args.log_interval == 0 or episode == 0 or episode + 1 == args.episodes:
            row = {
                "episode": episode + 1, "loss": float(loss.item()),
                "fsl_loss": float(fsl_loss.item()), "domain_loss": float(domain_loss.item()),
                "train_episode_accuracy": float(correct_total / sample_total),
                "grl_coefficient": coefficient,
            }
            history.append(row)
            print(f"ep {episode + 1} loss={row['loss']:.4f} fsl={row['fsl_loss']:.4f} "
                  f"domain={row['domain_loss']:.4f} acc={row['train_episode_accuracy']:.4f} "
                  f"grl={coefficient:.4f}", flush=True)

    train_seconds = time.perf_counter() - start
    checkpoint = {"model": model.state_dict(), "discriminator": discriminator.state_dict(),
                  "conditional_map": conditional_map.state_dict(), "config": config,
                  "episode": args.episodes}
    torch.save(checkpoint, args.output_dir / "final.pt")
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")
    metrics, truth, prediction = evaluate(
        model, support_loader, test_loader, device, n_classes)
    metrics.update(config=config, train_seconds=train_seconds,
                   selected_episode=args.episodes, selected_checkpoint="final.pt")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    with np.load(args.target_split, allow_pickle=False) as saved:
        test_indices = saved["test"]
    np.savez_compressed(args.output_dir / "predictions.npz", flat_indices=test_indices,
                        truth=truth + 1, prediction=prediction + 1,
                        image_shape=target_gt.shape)
    print(f"[final] OA={metrics['oa']:.2f}% AA={metrics['aa']:.2f}% "
          f"Kappa={metrics['kappa']:.4f}", flush=True)
    print(f"[saved] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
