from __future__ import annotations

import json
import random
import shutil
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import yaml
from PIL import Image
from requests import Session
from tqdm import tqdm


SVHN_URLS = {
    "train": "http://ufldl.stanford.edu/housenumbers/train.tar.gz",
    "test": "http://ufldl.stanford.edu/housenumbers/test.tar.gz",
}


@dataclass(slots=True)
class DigitBox:
    label: int
    left: float
    top: float
    width: float
    height: float


@dataclass(slots=True)
class NumberAnnotation:
    image_name: str
    text: str
    left: float
    top: float
    width: float
    height: float
    digits: list[DigitBox]


def download_svhn(root: Path, overwrite: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    session = Session()

    for split, url in SVHN_URLS.items():
        archive_path = root / f"{split}.tar.gz"
        target_dir = root / split

        if archive_path.exists() and not overwrite:
            pass
        else:
            _download_file(session, url, archive_path)

        if target_dir.exists() and not overwrite:
            continue

        if target_dir.exists():
            shutil.rmtree(target_dir)

        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(root)


def parse_digit_struct(mat_path: Path) -> list[NumberAnnotation]:
    annotations: list[NumberAnnotation] = []
    with h5py.File(mat_path, "r") as handle:
        digit_struct = handle["digitStruct"]
        names = digit_struct["name"]
        bboxes = digit_struct["bbox"]

        for idx in tqdm(range(len(names)), desc=f"Parsing {mat_path.parent.name}", unit="image"):
            image_name = _decode_name(handle, names[idx][0])
            bbox_group = handle[bboxes[idx][0]]
            digits = _read_digits(handle, bbox_group)
            annotations.append(_merge_digits(image_name, digits))

    return annotations


def prepare_yolo_dataset(
    raw_root: Path,
    output_root: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
    train_limit: int | None = None,
    test_limit: int | None = None,
) -> Path:
    train_annotations = parse_digit_struct(raw_root / "train" / "digitStruct.mat")
    test_annotations = parse_digit_struct(raw_root / "test" / "digitStruct.mat")

    random.Random(seed).shuffle(train_annotations)
    if train_limit is not None:
        train_annotations = train_annotations[:train_limit]
    if test_limit is not None:
        test_annotations = test_annotations[:test_limit]

    val_size = max(1, int(len(train_annotations) * val_ratio))
    val_annotations = train_annotations[:val_size]
    train_annotations = train_annotations[val_size:]

    splits = {
        "train": {"annotations": train_annotations, "source_split": "train"},
        "val": {"annotations": val_annotations, "source_split": "train"},
        "test": {"annotations": test_annotations, "source_split": "test"},
    }

    for split in splits:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split, split_config in splits.items():
        _export_split(
            annotations=split_config["annotations"],
            split=split,
            source_dir=raw_root / split_config["source_split"],
            output_root=output_root,
        )

    dataset_yaml = output_root / "dataset.yaml"
    with dataset_yaml.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            {
                "path": output_root.resolve().as_posix(),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "number"},
            },
            fh,
            sort_keys=False,
            allow_unicode=True,
        )

    return dataset_yaml


def _download_file(session: Session, url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with destination.open("wb") as fh, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {destination.name}",
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                fh.write(chunk)
                progress.update(len(chunk))


def _decode_name(handle: h5py.File, reference: h5py.Reference) -> str:
    data = handle[reference][()]
    return "".join(chr(value[0]) for value in data)


def _read_digits(handle: h5py.File, bbox_group: h5py.Group) -> list[DigitBox]:
    labels = _read_attribute(handle, bbox_group, "label")
    lefts = _read_attribute(handle, bbox_group, "left")
    tops = _read_attribute(handle, bbox_group, "top")
    widths = _read_attribute(handle, bbox_group, "width")
    heights = _read_attribute(handle, bbox_group, "height")

    digits: list[DigitBox] = []
    for label, left, top, width, height in zip(labels, lefts, tops, widths, heights, strict=True):
        label_int = int(label) % 10
        digits.append(
            DigitBox(
                label=label_int,
                left=float(left),
                top=float(top),
                width=float(width),
                height=float(height),
            )
        )
    return digits


def _read_attribute(handle: h5py.File, bbox_group: h5py.Group, key: str) -> list[float]:
    dataset = bbox_group[key]
    raw = dataset[()]

    if np.isscalar(raw):
        return [float(raw)]

    values: list[float] = []
    for item in np.array(raw).reshape(-1):
        if isinstance(item, h5py.Reference):
            value = np.array(handle[item][()]).reshape(-1)[0]
            values.append(float(value))
        else:
            values.append(float(np.array(item).reshape(-1)[0]))
    return values


def _merge_digits(image_name: str, digits: list[DigitBox]) -> NumberAnnotation:
    if not digits:
        raise ValueError(f"{image_name} does not contain any digit boxes")

    left = min(digit.left for digit in digits)
    top = min(digit.top for digit in digits)
    right = max(digit.left + digit.width for digit in digits)
    bottom = max(digit.top + digit.height for digit in digits)
    text = "".join(str(digit.label) for digit in digits)

    return NumberAnnotation(
        image_name=image_name,
        text=text,
        left=left,
        top=top,
        width=right - left,
        height=bottom - top,
        digits=digits,
    )


def _export_split(
    annotations: Iterable[NumberAnnotation],
    split: str,
    source_dir: Path,
    output_root: Path,
) -> None:
    image_dir = output_root / "images" / split
    label_dir = output_root / "labels" / split
    metadata_path = output_root / f"{split}_metadata.json"

    metadata: list[dict[str, object]] = []
    for annotation in tqdm(list(annotations), desc=f"Exporting {split}", unit="image"):
        src_image = source_dir / annotation.image_name
        dst_image = image_dir / annotation.image_name
        dst_label = label_dir / f"{Path(annotation.image_name).stem}.txt"

        shutil.copy2(src_image, dst_image)

        with Image.open(dst_image) as image:
            img_w, img_h = image.size

        x1 = min(max(annotation.left, 0.0), float(img_w))
        y1 = min(max(annotation.top, 0.0), float(img_h))
        x2 = min(max(annotation.left + annotation.width, 0.0), float(img_w))
        y2 = min(max(annotation.top + annotation.height, 0.0), float(img_h))

        width_abs = max(0.0, x2 - x1)
        height_abs = max(0.0, y2 - y1)
        x_center = ((x1 + x2) / 2.0) / img_w
        y_center = ((y1 + y2) / 2.0) / img_h
        width = width_abs / img_w
        height = height_abs / img_h

        with dst_label.open("w", encoding="utf-8") as label_file:
            label_file.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        item = asdict(annotation)
        item["image_width"] = img_w
        item["image_height"] = img_h
        metadata.append(item)

    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
