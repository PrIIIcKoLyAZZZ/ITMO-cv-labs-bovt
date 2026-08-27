from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PARTITION_MAP = {0: "train", 1: "valid", 2: "test"}


def parse_attributes(attr_path: Path) -> pd.DataFrame:
    with attr_path.open("r", encoding="utf-8") as handle:
        handle.readline()
        attr_names = handle.readline().strip().split()

    frame = pd.read_csv(
        attr_path,
        sep=r"\s+",
        skiprows=2,
        names=["image_id", *attr_names],
    )
    for column in attr_names:
        frame[column] = (frame[column] == 1).astype(int)
    return frame


def parse_partitions(partition_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(partition_path, sep=r"\s+", names=["image_id", "partition"])
    frame["split"] = frame["partition"].map(PARTITION_MAP)
    return frame


def build_metadata(
    raw_root: Path,
    output_root: Path,
    train_limit: int | None,
    valid_limit: int | None,
    test_limit: int | None,
) -> None:
    attrs = parse_attributes(raw_root / "list_attr_celeba.txt")
    parts = parse_partitions(raw_root / "list_eval_partition.txt")
    metadata = attrs.merge(parts, on="image_id", how="inner")

    limits = {"train": train_limit, "valid": valid_limit, "test": test_limit}
    chunks = []
    for split_name, chunk in metadata.groupby("split", sort=False):
        limit = limits[split_name]
        if limit is not None:
            chunk = chunk.head(limit)
        chunks.append(chunk)
    metadata = pd.concat(chunks, ignore_index=True)

    metadata["processed_rel_path"] = metadata["image_id"].map(lambda image_id: f"img_align_celeba/{image_id}")
    metadata["detected_face"] = 1

    output_root.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_root / "metadata.csv", index=False)
    pd.DataFrame(
        [
            {
                "train_count": int((metadata["split"] == "train").sum()),
                "valid_count": int((metadata["split"] == "valid").sum()),
                "test_count": int((metadata["split"] == "test").sum()),
            }
        ]
    ).to_csv(output_root / "subset_stats.csv", index=False)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build metadata for aligned CelebA images without recropping.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--valid-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    build_metadata(
        raw_root=args.raw_root,
        output_root=args.output_root,
        train_limit=args.train_limit,
        valid_limit=args.valid_limit,
        test_limit=args.test_limit,
    )


if __name__ == "__main__":
    main()
