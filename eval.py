"""Unified LGSPR-Net evaluation and classification-map entry."""

import argparse
import os

from utils import data
from utils.evaluation import run_target_map


CASE_CONFIGS = {
    "case1": dict(target="pc", bands=102, classes=9),
    "case2": dict(target="sa", bands=200, classes=16),
    "case3": dict(target="pc", bands=100, classes=9),
    "case4": dict(target="sa", bands=100, classes=16),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LGSPR-Net and generate a target classification map"
    )
    parser.add_argument("--case", choices=tuple(CASE_CONFIGS), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument(
        "--target_split",
        default=None,
        help="fixed 5-shot split used by training; recommended for exact reproduction",
    )
    parser.add_argument("--pavia_c_mat", default="./Datasets/paviaC/Pavia.mat")
    parser.add_argument("--pavia_c_gt", default="./Datasets/paviaC/Pavia_gt.mat")
    parser.add_argument("--salinas_mat", default="./Datasets/salinas/Salinas_corrected.mat")
    parser.add_argument("--salinas_gt", default="./Datasets/salinas/Salinas_gt.mat")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--labeled_per_class", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--proto_temperature", type=float, default=0.5)
    parser.add_argument("--category_dim", type=int, default=128)
    parser.add_argument("--response_dim", type=int, default=32)
    parser.add_argument("--max_test_per_class", type=int, default=0)
    parser.add_argument("--patch_size", type=int, default=9)
    return parser.parse_args()


def main():
    args = parse_args()
    config = CASE_CONFIGS[args.case]
    if config["target"] == "pc":
        cube, ground_truth = data.load_pavia_centre(
            args.pavia_c_mat,
            args.pavia_c_gt,
            n_bands=102 if args.case == "case1" else None,
        )
    else:
        cube, ground_truth = data.load_salinas(
            args.salinas_mat,
            args.salinas_gt,
            n_bands=200 if args.case == "case2" else None,
        )

    run_target_map(
        case_name=args.case,
        cube=cube,
        ground_truth=ground_truth,
        checkpoint=args.checkpoint,
        out_dir=args.out_dir or os.path.join("classification_results", args.case),
        gpu=args.gpu,
        seed=args.seed,
        n_bands=config["bands"],
        n_way=config["classes"],
        labeled_per_class=args.labeled_per_class,
        batch_size=args.batch_size,
        proto_temperature=args.proto_temperature,
        category_dim=args.category_dim,
        response_dim=args.response_dim,
        max_test_per_class=args.max_test_per_class,
        patch_size=args.patch_size,
        target_split=args.target_split,
    )


if __name__ == "__main__":
    main()
