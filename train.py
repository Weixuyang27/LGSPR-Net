"""Unified LGSPR-Net training entry for all four transfer settings."""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

from utils import checkpoint as paired_resume
from utils import data
from utils.metrics import average_accuracy
from models.lgspr_net import LGSPRNet, PCRA, SPS, prototype_logits


CONFIGS = {
    "case1": dict(name="PU-to-PC", bands=102, classes=9, episode_classes=9,
                  source_query=19, target_views=4, source_bands=102, target_bands=102),
    "case2": dict(name="IP-to-SA", bands=200, classes=16, episode_classes=8,
                  source_query=4, target_views=1, source_bands=200, target_bands=200),
    "case3": dict(name="CH-to-PC", bands=100, classes=9, episode_classes=9,
                  source_query=19, target_views=4, source_bands=128, target_bands=102),
    "case4": dict(name="CH-to-SA", bands=100, classes=16, episode_classes=8,
                  source_query=4, target_views=1, source_bands=128, target_bands=204),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LGSPR-Net on one transfer setting")
    parser.add_argument("--case", choices=tuple(CONFIGS), required=True,
                        help="case1: PU-to-PC; case2: IP-to-SA; case3: CH-to-PC; case4: CH-to-SA")
    parser.add_argument("--target_split", default=None)
    parser.add_argument("--split_seed", type=int, default=1337)
    parser.add_argument("--indian_mat", default="./Datasets/indianPines/Indian_pines_corrected.mat")
    parser.add_argument("--indian_gt", default="./Datasets/indianPines/Indian_pines_gt.mat")
    parser.add_argument("--salinas_mat", default="./Datasets/salinas/Salinas_corrected.mat")
    parser.add_argument("--salinas_gt", default="./Datasets/salinas/Salinas_gt.mat")
    parser.add_argument("--chikusei_mat", default="./Datasets/Chikusei_MATLAB/HyperspecVNIR_Chikusei_20140729.mat")
    parser.add_argument("--chikusei_gt", default="./Datasets/Chikusei_MATLAB/HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat")
    parser.add_argument("--pavia_c_mat", default="./Datasets/paviaC/Pavia.mat")
    parser.add_argument("--pavia_c_gt", default="./Datasets/paviaC/Pavia_gt.mat")
    parser.add_argument("--pavia_u_mat", default="./Datasets/paviaU/PaviaU.mat")
    parser.add_argument("--pavia_u_gt", default="./Datasets/paviaU/PaviaU_gt.mat")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--target_query", type=int, default=0)
    parser.add_argument("--proto_temperature", type=float, default=0.5)
    parser.add_argument("--episode", type=int, default=5000)
    parser.add_argument(
        "--train_log_interval",
        type=int,
        default=None,
        help="training-log interval; defaults to 1 for <=20 episodes and 200 otherwise",
    )
    parser.add_argument("--test_episode_interval", type=int, default=1000)
    parser.add_argument("--eval_at_start", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--category_dim", type=int, default=128)
    parser.add_argument("--response_dim", type=int, default=32)
    lgsse = parser.add_mutually_exclusive_group()
    lgsse.add_argument("--lgsse", dest="lgsse", action="store_true")
    lgsse.add_argument("--no_lgsse", dest="lgsse", action="store_false")
    parser.set_defaults(lgsse=True)
    parser.add_argument("--exp_name", default="default")
    parser.add_argument("--metrics_json", default="")
    parser.add_argument("--labeled_per_class", type=int, default=5)
    parser.add_argument("--patch_size", type=int, default=9)
    parser.add_argument("--max_test_per_class", type=int, default=0)
    parser.add_argument("--data_progress_interval", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--source_sample_seed", type=int, default=0)
    parser.add_argument("--save_dir", default="checkpoints")
    SPS.add_arguments(parser)
    PCRA.add_arguments(parser)
    paired_resume.add_arguments(parser)
    args = parser.parse_args()
    if args.train_log_interval is None:
        args.train_log_interval = 1 if args.episode <= 20 else 200
    if args.train_log_interval < 0:
        parser.error("--train_log_interval must be nonnegative")
    config = CONFIGS[args.case]
    for key, value in config.items():
        setattr(args, key, value)
    return args


def _resolve_chikusei_paths(args):
    if args.case in ("case1", "case2"):
        return
    folder = Path(args.chikusei_mat).parent
    if not folder.is_dir():
        return
    files = list(folder.glob("*.mat"))
    if not Path(args.chikusei_gt).exists():
        candidates = [p for p in files if "gt" in p.stem.lower() or "ground" in p.stem.lower()]
        if len(candidates) == 1:
            args.chikusei_gt = str(candidates[0])
    if not Path(args.chikusei_mat).exists():
        candidates = [p for p in files if "gt" not in p.stem.lower() and "ground" not in p.stem.lower()]
        if len(candidates) == 1:
            args.chikusei_mat = str(candidates[0])


def _load_cubes(args):
    if args.case == "case1":
        paths = (args.pavia_u_mat, args.pavia_u_gt, args.pavia_c_mat, args.pavia_c_gt)
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing required dataset file: {path}")
        source_cube, source_gt = data.load_pavia_university(
            args.pavia_u_mat, args.pavia_u_gt, args.bands
        )
        target_cube, target_gt = data.load_pavia_centre(
            args.pavia_c_mat, args.pavia_c_gt, args.bands
        )
    elif args.case == "case2":
        paths = (args.indian_mat, args.indian_gt, args.salinas_mat, args.salinas_gt)
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing required dataset file: {path}")
        source_cube, source_gt = data.load_indian_pines(args.indian_mat, args.indian_gt, args.bands)
        target_cube, target_gt = data.load_salinas(args.salinas_mat, args.salinas_gt, args.bands)
    elif args.case == "case3":
        paths = (args.chikusei_mat, args.chikusei_gt, args.pavia_c_mat, args.pavia_c_gt)
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing required dataset file: {path}")
        source_cube, source_gt = data.load_chikusei(args.chikusei_mat, args.chikusei_gt, args.source_bands)
        target_cube, target_gt = data.load_pavia_centre(args.pavia_c_mat, args.pavia_c_gt, n_bands=None)
    else:
        paths = (args.chikusei_mat, args.chikusei_gt, args.salinas_mat, args.salinas_gt)
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing required dataset file: {path}")
        source_cube, source_gt = data.load_chikusei(args.chikusei_mat, args.chikusei_gt, args.source_bands)
        target_cube, target_gt = data.load_salinas(args.salinas_mat, args.salinas_gt, n_bands=None)
    return source_cube, source_gt, target_cube, target_gt


def run_episode(model, source_pool, target_pool, args, target_query, device):
    source_classes = random.sample(sorted(source_pool), args.episode_classes)
    target_classes = random.sample(sorted(target_pool), args.episode_classes)
    source_task = data.TaskMatchedClasses(source_pool, source_classes, args.shot, args.source_query)
    target_task = data.TaskMatchedClasses(target_pool, target_classes, args.shot, target_query)
    sup_s, y_sup_s = next(iter(data.get_loader_for_task(source_task, args.shot, "train", shuffle=False)))
    qry_s, y_qry_s = next(iter(data.get_loader_for_task(source_task, args.source_query, "test", shuffle=True)))
    sup_t, y_sup_t = next(iter(data.get_loader_for_task(target_task, args.shot, "train", shuffle=False)))
    qry_t, y_qry_t = next(iter(data.get_loader_for_task(target_task, target_query, "test", shuffle=True)))
    sup_s, qry_s, sup_t, qry_t = [x.float().to(device) for x in (sup_s, qry_s, sup_t, qry_t)]
    y_sup_s, y_qry_s, y_sup_t, y_qry_t = [x.long().to(device) for x in (y_sup_s, y_qry_s, y_sup_t, y_qry_t)]

    if args.target_views > 1:
        sup_t = sup_t.repeat_interleave(args.target_views, dim=0)
        qry_t = qry_t.repeat_interleave(args.target_views, dim=0)
        y_sup_t = y_sup_t.repeat_interleave(args.target_views, dim=0)
        y_qry_t = y_qry_t.repeat_interleave(args.target_views, dim=0)
    if args.case == "case1":
        # Keep the recorded Case-1 RNG order for seeded reproduction.
        alpha_sup = torch.empty((len(sup_t), 1, 1, 1), device=device).uniform_(0.9, 1.1)
        alpha_qry = torch.empty((len(qry_t), 1, 1, 1), device=device).uniform_(0.9, 1.1)
        sup_t = alpha_sup * sup_t + torch.randn_like(sup_t) / 25.0
        qry_t = alpha_qry * qry_t + torch.randn_like(qry_t) / 25.0
    else:
        sup_t = torch.empty((len(sup_t), 1, 1, 1), device=device).uniform_(0.9, 1.1) * sup_t + torch.randn_like(sup_t) / 25.0
        qry_t = torch.empty((len(qry_t), 1, 1, 1), device=device).uniform_(0.9, 1.1) * qry_t + torch.randn_like(qry_t) / 25.0

    sup_s = model.map_spectral_input(sup_s, "source")
    qry_s = model.map_spectral_input(qry_s, "source")
    sup_t = model.map_spectral_input(sup_t, "target")
    qry_t = model.map_spectral_input(qry_t, "target")
    (zc_sup_s, zr_sup_s), (zc_qry_s, zr_qry_s), (zc_sup_t, zr_sup_t), (zc_qry_t, zr_qry_t) = [
        model.forward_embeddings(x) for x in (sup_s, qry_s, sup_t, qry_t)
    ]
    logits_s = prototype_logits(zc_qry_s, zc_sup_s, y_sup_s, args.episode_classes, args.proto_temperature)
    logits_t = prototype_logits(zc_qry_t, zc_sup_t, y_sup_t, args.episode_classes, args.proto_temperature)
    clean_s = F.cross_entropy(logits_s, y_qry_s)
    clean_t = F.cross_entropy(logits_t, y_qry_t)
    clean = 0.5 * (clean_s + clean_t)
    loss = clean
    if model.sps_enabled:
        perturbation_loss, _ = SPS.loss(model, [
            (qry_s, zc_qry_s, zr_qry_s, zc_sup_s, y_sup_s, y_qry_s),
            (qry_t, zc_qry_t, zr_qry_t, zc_sup_t, y_sup_t, y_qry_t),
        ], args.episode_classes, args.proto_temperature, return_augmented=True)
        loss = 0.5 * clean + perturbation_loss
    if model.pcra_enabled:
        loss = loss + PCRA.loss(model, [
            (zc_sup_s, y_sup_s, zc_qry_s, y_qry_s),
            (zc_sup_t, y_sup_t, zc_qry_t, y_qry_t),
        ], args.episode_classes, args.target_views)
    accuracy_s = float((logits_s.argmax(1) == y_qry_s).float().mean())
    accuracy_t = float((logits_t.argmax(1) == y_qry_t).float().mean())
    return loss, float(clean.detach()), accuracy_s, accuracy_t


def evaluate(model, train_loader, test_loader, args, device):
    model.eval()
    support, support_labels = next(iter(train_loader))
    support = model.map_spectral_input(support.float().to(device), "target")
    support_labels = support_labels.long().to(device)
    with torch.no_grad():
        support_features = model(support)
        predictions, labels = [], []
        for queries, query_labels in test_loader:
            queries = model.map_spectral_input(queries.float().to(device), "target")
            logits = prototype_logits(model(queries), support_features, support_labels,
                                      args.classes, args.proto_temperature)
            predictions.extend(logits.argmax(1).cpu().tolist())
            labels.extend(query_labels.tolist())
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    confusion = metrics.confusion_matrix(labels, predictions, labels=list(range(args.classes)))
    each, aa = average_accuracy(confusion)
    return (100.0 * float((predictions == labels).mean()), 100.0 * aa,
            metrics.cohen_kappa_score(labels, predictions), 100.0 * each)


def main():
    args = parse_args()
    case = args.case
    if args.patch_size < 3 or args.patch_size % 2 == 0:
        raise ValueError("patch_size must be an odd integer >= 3")
    _resolve_chikusei_paths(args)
    paired_resume.setup(args)
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    source_cube, source_gt, target_cube, target_gt = _load_cubes(args)
    half_width = args.patch_size // 2
    source_pool = data.build_metatrain_dict(source_cube, source_gt, max_per_class=200,
                                            half_width=half_width, seed=args.source_sample_seed)
    target_pack_builder = (
        data.get_target_pack if case == "case1" else data.get_target_pack_generic
    )
    train_loader, test_loader, *_ = target_pack_builder(
        target_cube, target_gt, class_num=args.classes,
        shot_num_per_class=args.labeled_per_class,
        test_lsample_num_per_class=args.labeled_per_class,
        max_test_per_class=args.max_test_per_class,
        progress_interval=args.data_progress_interval, build_da=False,
        target_split=args.target_split, split_seed=args.split_seed,
        half_width=half_width,
    )
    target_data, target_labels = next(iter(train_loader))
    target_pool = {}
    for patch, label in zip(target_data, target_labels):
        target_pool.setdefault(int(label), []).append(patch.numpy())
    target_query = args.target_query or (args.labeled_per_class - args.shot)
    if target_query < 1:
        raise ValueError("target_query must be positive; labeled_per_class must exceed shot")
    if len(source_pool) < args.episode_classes or len(target_pool) != args.classes:
        raise ValueError("source or target episodic pool does not contain the required classes")
    if any(len(items) < args.shot + target_query for items in target_pool.values()):
        raise ValueError("not enough distinct target samples for the requested episode")

    model = LGSPRNet(
        args.bands,
        args.category_dim,
        args.response_dim,
        use_lgsse=args.lgsse,
    )
    model.configure_spectral_mappings(args.source_bands, args.target_bands)
    model = model.to(device)
    SPS.configure(model, args)
    PCRA.configure(model, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint = os.path.join(args.save_dir, f"lgspr_net_{case}_best.pth")
    start, best_oa, best_episode = paired_resume.resume(args, model, optimizer, checkpoint)
    components = []
    if args.lgsse:
        components.append("LGSSE")
    if model.sps_enabled:
        components.append("SPS")
    if model.pcra_enabled:
        components.append("PCRA")
    print(
        f"[train] {args.name}; "
        f"{(' + '.join(components) if components else 'local encoder baseline')}",
        flush=True,
    )
    started = time.time()
    for episode in range(start, args.episode):
        paired_resume.episode_seed(args, episode + 1)
        SPS.set_episode(model, args, episode + 1)
        PCRA.set_episode(model, args, episode + 1)
        model.train()
        loss, clean, accuracy_s, accuracy_t = run_episode(
            model, source_pool, target_pool, args, target_query, device
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.train_log_interval and (episode + 1) % args.train_log_interval == 0:
            print(f"ep {episode + 1} loss={loss.item():.4f} fsl={clean:.4f} "
                  f"acc_s={accuracy_s:.4f} acc_t={accuracy_t:.4f}", flush=True)
        should_evaluate = ((args.test_episode_interval and
                            (episode + 1) % args.test_episode_interval == 0)
                           or (args.eval_at_start and episode == 0)
                           or episode + 1 == args.episode)
        if should_evaluate:
            oa, aa, kappa, each = evaluate(model, train_loader, test_loader, args, device)
            if oa > best_oa:
                best_oa, best_episode = oa, episode + 1
                torch.save({
                    "model": model.state_dict(), "episode": best_episode,
                    "oa": oa, "aa": aa, "kappa": kappa,
                    "case": case, "seed": args.seed,
                    "split_seed": args.split_seed,
                    "target_split": args.target_split,
                    "labeled_per_class": args.labeled_per_class,
                    "n_bands": args.bands,
                    "source_input_bands": args.source_bands,
                    "target_input_bands": args.target_bands,
                    "category_dim": args.category_dim,
                    "response_dim": args.response_dim,
                    "n_way": args.classes, "patch_size": args.patch_size,
                    "proto_temperature": args.proto_temperature,
                    "architecture": "LGSPR-Net", "lgsse": bool(args.lgsse),
                    **SPS.metadata(args),
                    **PCRA.metadata(args), **paired_resume.metadata(args),
                }, checkpoint)
            print(f"[eval] OA={oa:.2f}% AA={aa:.2f}% Kappa={kappa:.4f}", flush=True)
        if args.paired_stop_at and episode + 1 == args.paired_stop_at:
            paired_resume.save(args, model, optimizer, episode + 1,
                               best_oa, best_episode, checkpoint)
            return
    if args.paired_snapshot:
        paired_resume.save(args, model, optimizer, args.episode,
                           best_oa, best_episode, checkpoint)
    packet = load_checkpoint(checkpoint, device)
    model.load_state_dict(packet["model"])
    oa, aa, kappa, each = evaluate(model, train_loader, test_loader, args, device)
    print(f"best episode={best_episode} OA={oa:.2f}% AA={aa:.2f}% Kappa={kappa:.4f}", flush=True)
    if args.metrics_json:
        folder = os.path.dirname(args.metrics_json)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(args.metrics_json, "w", encoding="utf-8") as stream:
            json.dump({
                "exp_name": args.exp_name, "case": case, "seed": args.seed,
                "training_time_seconds": time.time() - started,
                "selected_episode": best_episode, "oa": oa, "aa": aa,
                "kappa": kappa, "each_class_accuracy": each.tolist(),
                "split_seed": args.split_seed,
                "target_split": args.target_split,
                "labeled_per_class": args.labeled_per_class,
                "patch_size": args.patch_size,
                "proto_temperature": args.proto_temperature,
                "source_sample_seed": args.source_sample_seed,
                "source_input_bands": args.source_bands,
                "target_input_bands": args.target_bands,
                "architecture": "LGSPR-Net", "lgsse": bool(args.lgsse),
                **SPS.metadata(args),
                **PCRA.metadata(args), **paired_resume.metadata(args),
            }, stream, indent=2)


if __name__ == "__main__":
    main()
