from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

from celeba_gan.utils import choose_device, ensure_dir


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


def center_crop_fallback(image: Image.Image, image_size: int) -> Image.Image:
    width, height = image.size
    edge = min(width, height)
    left = (width - edge) // 2
    top = (height - edge) // 2
    image = image.crop((left, top, left + edge, top + edge))
    return image.resize((image_size, image_size), Image.Resampling.BILINEAR)


def detect_face(mtcnn: MTCNN, image: Image.Image) -> Image.Image | None:
    face_tensor = mtcnn(image)
    if face_tensor is None:
        return None

    face_tensor = face_tensor.detach().cpu().clamp(0, 255).to(torch.uint8)
    face_array = face_tensor.permute(1, 2, 0).numpy()
    return Image.fromarray(face_array)


def preprocess_dataset(
    raw_root: Path,
    output_root: Path,
    image_dir_name: str,
    image_size: int,
    margin: int,
    fallback_mode: str,
    max_images: int | None,
    max_per_split: int | None,
) -> None:
    image_root = raw_root / image_dir_name
    attr_path = raw_root / "list_attr_celeba.txt"
    partition_path = raw_root / "list_eval_partition.txt"

    attrs = parse_attributes(attr_path)
    parts = parse_partitions(partition_path)
    metadata = attrs.merge(parts, on="image_id", how="inner")
    if max_per_split is not None:
        metadata = (
            metadata.groupby("split", group_keys=False)
            .head(max_per_split)
            .reset_index(drop=True)
        )
    elif max_images is not None:
        metadata = metadata.head(max_images).reset_index(drop=True)

    ensure_dir(output_root)
    ensure_dir(output_root / "images")
    for split_name in PARTITION_MAP.values():
        ensure_dir(output_root / "images" / split_name)

    device = choose_device()
    mtcnn = MTCNN(
        image_size=image_size,
        margin=margin,
        post_process=False,
        select_largest=True,
        keep_all=False,
        device=device,
    )

    processed_rows: list[dict] = []
    detected_count = 0
    fallback_count = 0
    skipped_count = 0

    records = metadata.to_dict(orient="records")
    for row in tqdm(records, total=len(records), desc="Preprocessing CelebA"):
        src_path = image_root / row["image_id"]
        dst_rel_path = Path("images") / row["split"] / row["image_id"]
        dst_path = output_root / dst_rel_path

        image = Image.open(src_path).convert("RGB")
        face = detect_face(mtcnn, image)
        detected = face is not None

        if face is None:
            if fallback_mode == "skip":
                skipped_count += 1
                continue
            face = center_crop_fallback(image, image_size=image_size)
            fallback_count += 1
        else:
            detected_count += 1

        face.save(dst_path, quality=95)

        row_dict = {
            "image_id": row["image_id"],
            "split": row["split"],
            "partition": int(row["partition"]),
            "processed_rel_path": dst_rel_path.as_posix(),
            "detected_face": int(detected),
        }
        for column in attrs.columns:
            if column != "image_id":
                row_dict[column] = int(row[column])
        processed_rows.append(row_dict)

    processed_frame = pd.DataFrame(processed_rows)
    processed_frame.to_csv(output_root / "metadata.csv", index=False)

    stats = {
        "total_raw_images": int(len(metadata)),
        "saved_images": int(len(processed_frame)),
        "detected_faces": int(detected_count),
        "fallback_used": int(fallback_count),
        "skipped_images": int(skipped_count),
        "image_size": int(image_size),
        "fallback_mode": fallback_mode,
        "max_images": None if max_images is None else int(max_images),
        "max_per_split": None if max_per_split is None else int(max_per_split),
    }
    pd.DataFrame([stats]).to_csv(output_root / "preprocess_stats.csv", index=False)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crop CelebA faces with MTCNN.")
    parser.add_argument("--raw-root", type=Path, required=True, help="CelebA root with images and annotation txt files.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output directory for cropped images.")
    parser.add_argument("--image-dir-name", type=str, default="img_align_celeba", help="Folder with raw images inside raw-root.")
    parser.add_argument("--image-size", type=int, default=64, help="Square image size for GAN training.")
    parser.add_argument("--margin", type=int, default=20, help="Extra crop margin passed to MTCNN.")
    parser.add_argument(
        "--fallback-mode",
        choices=("center-crop", "skip"),
        default="center-crop",
        help="Fallback when the detector does not find a face.",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Optional global image limit for quick experiments.")
    parser.add_argument(
        "--max-per-split",
        type=int,
        default=None,
        help="Optional per-split image limit. If set, it has priority over --max-images.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    preprocess_dataset(
        raw_root=args.raw_root,
        output_root=args.output_root,
        image_dir_name=args.image_dir_name,
        image_size=args.image_size,
        margin=args.margin,
        fallback_mode=args.fallback_mode,
        max_images=args.max_images,
        max_per_split=args.max_per_split,
    )


if __name__ == "__main__":
    main()
