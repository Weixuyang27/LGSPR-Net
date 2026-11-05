"""Unified training entry for LGSPR-Net and the comparison methods."""

import argparse
import importlib
import sys


TRANSFERS = {
    "pu_to_pc": "case1",
    "ip_to_sa": "case2",
    "ch_to_pc": "case3",
    "ch_to_sa": "case4",
}

TRAINERS = {
    "lgspr": "train",
    "ssrn": "baselines.train_ssrn",
    "rnfsc": "baselines.train_rnfsc",
    "dfsl": "baselines.train_dfsl",
    "giacfsl": "baselines.train_giacfsl",
    "dcfsl": "baselines.train_dcfsl",
    "cmfsl": "baselines.train_cmfsl",
}


def parse_entry_args():
    parser = argparse.ArgumentParser(
        description="Train LGSPR-Net or a comparison method on one transfer setting"
    )
    parser.add_argument("--model", choices=tuple(TRAINERS), required=True)
    parser.add_argument("--transfer", choices=tuple(TRANSFERS), required=True)
    args, remaining = parser.parse_known_args()
    return args, remaining


def main():
    args, remaining = parse_entry_args()
    module = importlib.import_module(TRAINERS[args.model])
    sys.argv = [sys.argv[0], "--case", TRANSFERS[args.transfer], *remaining]
    module.main()


if __name__ == "__main__":
    main()
