from __future__ import annotations

import argparse
from pathlib import Path

from lab6_pipeline import (
    dataset_summary,
    download_rtsd_segmentation_dataset,
    prepare_yolo_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data") / "yolo_dataset",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw_root = args.raw_root or download_rtsd_segmentation_dataset()
    prepared = prepare_yolo_dataset(raw_root=raw_root, output_root=args.output_root, overwrite=args.overwrite)
    print(f"Raw dataset: {prepared.raw_root}")
    print(f"YOLO dataset: {prepared.yolo_root}")
    print(f"YAML file: {prepared.data_yaml}")
    print(dataset_summary(prepared.raw_root).to_string(index=False))


if __name__ == "__main__":
    main()

