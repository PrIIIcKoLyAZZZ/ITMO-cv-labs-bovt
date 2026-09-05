from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import yaml
from ultralytics import YOLO


@dataclass(slots=True)
class Detection:
    box: tuple[float, float, float, float]
    confidence: float
    cls: int


@dataclass(slots=True)
class GroundTruth:
    box: tuple[float, float, float, float]
    cls: int


def evaluate_on_test_split(
    model_path: Path,
    data_yaml: Path,
    conf: float = 0.25,
    iou_threshold: float = 0.5,
    device: str | int | None = None,
    predict_batch: int = 32,
    output_json: Path | None = None,
) -> dict[str, float]:
    model = YOLO(model_path)
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        conf=conf,
        iou=iou_threshold,
        device=device,
        plots=False,
        verbose=False,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    image_paths = _resolve_test_images(data_yaml)

    tp = 0
    fp = 0
    fn = 0
    matched_ious: list[float] = []

    for batch_paths in _batched(image_paths, predict_batch):
        results = model.predict(
            source=[str(path) for path in batch_paths],
            conf=conf,
            iou=iou_threshold,
            device=device,
            batch=len(batch_paths),
            verbose=False,
            stream=True,
        )
        for image_path, prediction in zip(batch_paths, results, strict=True):
            height, width = prediction.orig_shape
            predictions = [
                Detection(
                    box=tuple(float(x) for x in box.tolist()),
                    confidence=float(score),
                    cls=int(cls_idx),
                )
                for box, score, cls_idx in zip(
                    prediction.boxes.xyxy,
                    prediction.boxes.conf,
                    prediction.boxes.cls,
                    strict=True,
                )
            ]
            targets = _load_ground_truths(image_path, width=width, height=height)
            batch_tp, batch_fp, batch_fn, batch_ious = _match_predictions(predictions, targets, iou_threshold)
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn
            matched_ious.extend(batch_ious)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0

    summary = {
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def _resolve_test_images(data_yaml: Path) -> list[Path]:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = Path(config["path"])
    test_path = base / config["test"]
    return sorted(path for path in test_path.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})


def _batched(items: list[Path], batch_size: int) -> Iterable[list[Path]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _load_ground_truths(image_path: Path, width: int, height: int) -> list[GroundTruth]:
    label_path = Path(str(image_path).replace("\\images\\", "\\labels\\")).with_suffix(".txt")
    if not label_path.exists():
        label_path = Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")
    targets: list[GroundTruth] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        cls_idx_str, xc_str, yc_str, w_str, h_str = line.strip().split()
        xc = float(xc_str) * width
        yc = float(yc_str) * height
        box_w = float(w_str) * width
        box_h = float(h_str) * height
        x1 = xc - box_w / 2.0
        y1 = yc - box_h / 2.0
        x2 = xc + box_w / 2.0
        y2 = yc + box_h / 2.0
        targets.append(GroundTruth(box=(x1, y1, x2, y2), cls=int(cls_idx_str)))
    return targets


def _match_predictions(
    predictions: Iterable[Detection],
    targets: list[GroundTruth],
    iou_threshold: float,
) -> tuple[int, int, int, list[float]]:
    predictions = sorted(predictions, key=lambda item: item.confidence, reverse=True)
    matched_targets: set[int] = set()
    tp = 0
    fp = 0
    ious: list[float] = []

    for prediction in predictions:
        best_iou = 0.0
        best_idx: int | None = None
        for idx, target in enumerate(targets):
            if idx in matched_targets or prediction.cls != target.cls:
                continue
            current_iou = _box_iou(prediction.box, target.box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_idx = idx

        if best_idx is not None and best_iou >= iou_threshold:
            matched_targets.add(best_idx)
            tp += 1
            ious.append(best_iou)
        else:
            fp += 1

    fn = len(targets) - len(matched_targets)
    return tp, fp, fn, ious


def _box_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area
