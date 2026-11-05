"""Train the official-style Gia-CFSL baseline under the shared project protocol."""
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
from models.giacfsl import (
    CSABlock,
    GiaNetwork,
    IDEBlock,
    euclidean_logits,
    initialize_graph,
    initialize_official,
    optimal_transport_alignment,
)
from utils.splits import fixed_target_pack
from utils.cases import CASE_NAMES, case_spec, load_case as load_shared_case, transfer_name


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "c0479e5b68c896479b7bb96bd6631d28ab6e833c"


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
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda_domain", type=float, default=1.0)
    parser.add_argument("--lambda_cross_attention", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--eval_batch_size", type=int, default=32)
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
    integer_positive = (args.labeled_per_class, args.source_max_per_class,
                        args.target_aug_per_class, args.episodes, args.episode_n_way,
                        args.shot, args.query, args.mapping_dim, args.embedding_dim,
                        args.generations, args.eval_batch_size, args.log_interval, args.threads)
    if min(integer_positive) < 1 or args.lr <= 0 or args.weight_decay < 0:
        parser.error("Invalid nonpositive training option")
    if not 0 <= args.dropout < 1 or min(args.lambda_domain, args.lambda_cross_attention) < 0:
        parser.error("Invalid loss/dropout option")
    suffix = f"{args.case}_{args.labeled_per_class}shot_split{args.split_seed}"
    if args.max_test_per_class:
        suffix += f"_DEBUG{args.max_test_per_class}"
    args.target_split = args.target_split or ROOT / "splits" / (suffix + ".npz")
    run_name = f"seed{args.seed}" + (f"_debug{args.max_test_per_class}" if args.max_test_per_class else "")
    args.output_dir = args.output_dir or ROOT / "results/comparisons" / transfer_name(args.case) / "giacfsl" / run_name
    return args


def load_case(args):
    bands = 100 if args.case in ("case3", "case4") else (102 if args.case == "case1" else 200)
    return load_shared_case(args.data_root, args.case, bands)


def augment_target_support(support_x, support_y, classes, per_class, rng):
    """Released radiation-noise expansion, restricted to the fixed five labels."""
    pools = {}
    for class_id in range(classes):
        original = support_x[support_y == class_id].cpu().numpy()
        if len(original) == 0:
            raise ValueError(f"Target support has no class {class_id}")
        generated = []
        for index in range(per_class):
            patch = original[index % len(original)].astype(np.float32, copy=True)
            alpha = rng.uniform(0.9, 1.1)
            noise = rng.normal(0.0, 1.0, patch.shape).astype(np.float32)
            generated.append((alpha * patch + noise / 25.0).astype(np.float32))
        pools[class_id] = generated
    return pools


def sample_episode(pool, eligible, n_way, shot, query):
    selected = random.sample(eligible, n_way)
    support_x, query_x, support_y, query_y = [], [], [], []
    for local_label, class_id in enumerate(selected):
        chosen = random.sample(pool[class_id], shot + query)
        support_x.extend(chosen[:shot])
        query_x.extend(chosen[shot:])
        support_y.extend([local_label] * shot)
        query_y.extend([local_label] * query)
    return (torch.from_numpy(np.stack(support_x).astype(np.float32)),
            torch.tensor(support_y, dtype=torch.long),
            torch.from_numpy(np.stack(query_x).astype(np.float32)),
            torch.tensor(query_y, dtype=torch.long))


def encode_episode(network, ide, episode, domain, device, num_support):
    support_x, support_y, query_x, query_y = episode
    support_y, query_y = support_y.to(device), query_y.to(device)
    support_features = network(support_x.to(device), domain)
    query_features = network(query_x.to(device), domain)
    labels = torch.cat((support_y, query_y))
    distribution_node, point_edge, distribution_edge = initialize_graph(
        labels, num_support, device)
    middle = torch.cat((support_features[0], query_features[0])).unsqueeze(0)
    final = torch.cat((support_features[1], query_features[1])).unsqueeze(0)
    point_nodes, distribution_nodes = ide(
        middle, final, distribution_node, distribution_edge, point_edge)
    return support_features, query_features, support_y, query_y, point_nodes, distribution_nodes


@torch.no_grad()
def evaluate(network, support_loader, test_loader, device, classes):
    network.eval()
    support_x, support_y = next(iter(support_loader))
    support_features = network(support_x.to(device), "target")[1]
    minimum, maximum = support_features.min(), support_features.max()
    scale = (maximum - minimum).clamp_min(torch.finfo(support_features.dtype).eps)
    support_features = (support_features - minimum) / scale
    truth, prediction = [], []
    for x, y in test_loader:
        features = (network(x.to(device), "target")[1] - minimum) / scale
        nearest = torch.cdist(features, support_features).argmin(1)
        prediction.append(support_y[nearest.cpu()].numpy())
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
    # The published encoder uses MaxPool3d, whose CUDA backward kernel is not
    # strictly deterministic in current PyTorch. Preserve it and warn instead.
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    source_cube, source_gt, target_cube, target_gt, classes = load_case(args)
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
        support_x, support_y, classes, args.target_aug_per_class,
        np.random.RandomState(args.seed))
    minimum_episode_size = args.shot + args.query
    source_eligible = sorted(k for k, values in source_pool.items()
                             if len(values) >= minimum_episode_size)
    target_eligible = sorted(k for k, values in target_pool.items()
                             if len(values) >= minimum_episode_size)
    if len(source_eligible) < args.episode_n_way:
        raise ValueError(f"Only {len(source_eligible)} source classes are eligible for "
                         f"{args.episode_n_way}-way {args.shot}-shot/{args.query}-query")
    if len(target_eligible) < args.episode_n_way:
        raise ValueError("Not enough eligible target classes")
    test_loader = torch.utils.data.DataLoader(
        raw_test_loader.dataset, batch_size=args.eval_batch_size, shuffle=False)

    num_support = args.episode_n_way * args.shot
    network = GiaNetwork(source_bands, target_bands, args.mapping_dim, args.embedding_dim)
    ide_source = IDEBlock(args.generations, args.dropout, num_support,
                          embedding_dim=args.embedding_dim)
    ide_target = IDEBlock(args.generations, args.dropout, num_support,
                          embedding_dim=args.embedding_dim)
    csa = CSABlock(num_support, args.embedding_dim, args.generations)
    for module in (network, ide_source, ide_target, csa):
        module.apply(initialize_official)
        module.to(device)
        module.train()
    optimizers = [
        torch.optim.Adam(network.parameters(), lr=args.lr),
        torch.optim.Adam(ide_source.parameters(), lr=args.lr, weight_decay=args.weight_decay),
        torch.optim.Adam(ide_target.parameters(), lr=args.lr, weight_decay=args.weight_decay),
        torch.optim.Adam(csa.parameters(), lr=args.lr, weight_decay=args.weight_decay),
    ]

    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(
        method="Gia-CFSL", architecture="official_mapping_dres3d_IDE_CSA_OT",
        source_dataset=case_spec(args.case)["source"],
        target_dataset=case_spec(args.case)["target"],
        source_bands=source_bands, target_bands=target_bands,
        protocol="shared_cross_domain_few_shot", target_classes=classes,
        patch_size=9, normalization="whole_image_bandwise_scale", padding="zero",
        checkpoint_selection="fixed_final_episode",
        target_augmentation="official_radiation_noise_alpha_0.9_1.1_beta_1_over_25",
        source_classes_total=len(source_pool), source_classes_eligible=len(source_eligible),
        source_samples_per_class={str(k): len(v) for k, v in source_pool.items()},
        target_train_count=len(support_loader.dataset), test_count=len(test_loader.dataset),
        split_sha256=hashlib.sha256(args.target_split.read_bytes()).hexdigest(),
        deterministic_algorithms="warn_only_maxpool3d_cuda",
        cudnn_benchmark=False, cudnn_deterministic=True,
        torch_version=str(torch.__version__), python_version=sys.version,
        python_executable=sys.executable, cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        implementation_source="YuxiangZhang-BIT/IEEE_TNNLS_Gia-CFSL",
        upstream_commit=UPSTREAM_COMMIT,
        upstream_actual_task_note=("released trainer uses CLASS_NUM rather than the unused "
                                   "config num_ways field"),
        adaptations=("project source-target pairs; shared fixed target splits; case2 "
                     "8-way/4-query; 5000 episodes; final-only evaluation; device-neutral OT"),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")
    print(f"[Gia-CFSL] {args.case} source->target | {args.episode_n_way}-way "
          f"{args.shot}-shot/{args.query}-query | device={device}", flush=True)
    print(f"[split] target train={len(support_loader.dataset)} test={len(test_loader.dataset)} "
          f"file={args.target_split}", flush=True)
    print(f"[source] classes={len(source_pool)} eligible={len(source_eligible)} "
          f"samples={[len(source_pool[k]) for k in sorted(source_pool)]}", flush=True)

    history, correct_source, total_source, correct_target, total_target = [], 0, 0, 0, 0
    start = time.perf_counter()
    for episode_index in range(args.episodes):
        source_episode = sample_episode(
            source_pool, source_eligible, args.episode_n_way, args.shot, args.query)
        target_episode = sample_episode(
            target_pool, target_eligible, args.episode_n_way, args.shot, args.query)
        source = encode_episode(
            network, ide_source, source_episode, "source", device, num_support)
        target = encode_episode(
            network, ide_target, target_episode, "target", device, num_support)
        source_support, source_query, _, source_query_y, source_points, source_distributions = source
        target_support, target_query, _, target_query_y, target_points, target_distributions = target
        source_proto = source_support[1].reshape(
            args.episode_n_way, args.shot, -1).mean(1)
        target_proto = target_support[1].reshape(
            args.episode_n_way, args.shot, -1).mean(1)
        source_logits = euclidean_logits(source_query[1], source_proto)
        target_logits = euclidean_logits(target_query[1], target_proto)
        fsl_loss = (torch.nn.functional.cross_entropy(source_logits, source_query_y) +
                    torch.nn.functional.cross_entropy(target_logits, target_query_y))
        # CSA intentionally runs before OT: the released code transforms the
        # graph-node lists in place and aligns those transformed nodes.
        cross_attention_loss = csa(
            source_points, target_points, source_distributions, target_distributions)
        domain_loss = (optimal_transport_alignment(source_points, target_points) +
                       optimal_transport_alignment(source_distributions, target_distributions))
        loss = (fsl_loss + args.lambda_domain * domain_loss +
                args.lambda_cross_attention * cross_attention_loss)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at episode {episode_index + 1}")
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        correct_source += int((source_logits.argmax(1) == source_query_y).sum())
        total_source += len(source_query_y)
        correct_target += int((target_logits.argmax(1) == target_query_y).sum())
        total_target += len(target_query_y)
        current = episode_index + 1
        if current == 1 or current % args.log_interval == 0 or current == args.episodes:
            row = {
                "episode": current, "loss": float(loss.item()),
                "fsl_loss": float(fsl_loss.item()),
                "domain_loss": float(domain_loss.item()),
                "cross_attention_loss": float(cross_attention_loss.item()),
                "source_accuracy": correct_source / total_source,
                "target_accuracy": correct_target / total_target,
            }
            history.append(row)
            print(f"ep {current} loss={row['loss']:.4f} fsl={row['fsl_loss']:.4f} "
                  f"domain={row['domain_loss']:.4f} csa={row['cross_attention_loss']:.4f} "
                  f"acc_s={row['source_accuracy']:.4f} acc_t={row['target_accuracy']:.4f}",
                  flush=True)

    train_seconds = time.perf_counter() - start
    checkpoint = {
        "network": network.state_dict(), "ide_source": ide_source.state_dict(),
        "ide_target": ide_target.state_dict(), "csa": csa.state_dict(),
        "config": config, "episode": args.episodes,
    }
    torch.save(checkpoint, args.output_dir / "final.pt")
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")
    metrics, truth, prediction = evaluate(
        network, support_loader, test_loader, device, classes)
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
