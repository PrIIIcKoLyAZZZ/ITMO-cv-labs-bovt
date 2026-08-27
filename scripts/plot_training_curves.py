from __future__ import annotations

import argparse
from pathlib import Path

from celeba_gan.plotting import save_training_curves


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot GAN training curves from history.csv.")
    parser.add_argument("--history", type=Path, required=True, help="Path to history.csv")
    parser.add_argument("--output", type=Path, required=True, help="Path to output PNG")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    save_training_curves(args.history, args.output)


if __name__ == "__main__":
    main()
