from __future__ import annotations

import argparse
from pathlib import Path

from svhn_detector.svhn import download_svhn, prepare_yolo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SVHN and convert it to a YOLO dataset.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/svhn"), help="Where to store raw SVHN archives.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/svhn_yolo"),
        help="Where to save YOLO-formatted images and labels.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio sampled from the train split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the train/val split.")
    parser.add_argument("--skip-download", action="store_true", help="Assume raw SVHN has already been downloaded.")
    parser.add_argument("--train-limit", type=int, default=None, help="Optional cap for train+val samples.")
    parser.add_argument("--test-limit", type=int, default=None, help="Optional cap for test samples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_download:
        download_svhn(args.raw_dir)

    dataset_yaml = prepare_yolo_dataset(
        raw_root=args.raw_dir,
        output_root=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
    )
    print(dataset_yaml.resolve())


if __name__ == "__main__":
    main()
