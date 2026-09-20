from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO


RAW_CLASS_IDS = [1, 2, 3, 4, 6, 8, 10, 13]
CLASS_NAMES = [f"class_{class_id}" for class_id in RAW_CLASS_IDS]
CLASS_ID_TO_INDEX = {class_id: index for index, class_id in enumerate(RAW_CLASS_IDS)}


@dataclass
class PreparedDataset:
    raw_root: Path
    yolo_root: Path
    data_yaml: Path


def download_rtsd_segmentation_dataset() -> Path:
    dataset_root = Path(
        kagglehub.dataset_download(
            "viacheslavshalamov/russian-road-signs-segmentation-dataset"
        )
    )
    return dataset_root / "sign_dataset"


def _read_annotation(annotation_path: Path) -> dict:
    return json.loads(annotation_path.read_text(encoding="utf-8"))


def _image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        width, height = image.size
    return height, width


def _restore_full_masks(
    annotation: dict,
    image_height: int,
    image_width: int,
) -> list[np.ndarray]:
    num_instances = len(annotation["class_ids"])
    if num_instances == 0:
        return []

    mini_masks = np.asarray(annotation["masks"], dtype=np.uint8)
    if mini_masks.ndim == 2:
        mini_masks = mini_masks[:, :, None]

    full_masks: list[np.ndarray] = []
    for instance_index, roi in enumerate(annotation["rois"]):
        y1, x1, y2, x2 = [int(value) for value in roi]
        y1 = max(0, min(y1, image_height))
        x1 = max(0, min(x1, image_width))
        y2 = max(0, min(y2, image_height))
        x2 = max(0, min(x2, image_width))

        if y2 <= y1 or x2 <= x1:
            full_masks.append(np.zeros((image_height, image_width), dtype=np.uint8))
            continue

        mini_mask = mini_masks[:, :, instance_index]
        resized_mask = cv2.resize(
            mini_mask,
            (x2 - x1, y2 - y1),
            interpolation=cv2.INTER_NEAREST,
        )
        full_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = (resized_mask > 0).astype(np.uint8)
        full_masks.append(full_mask)

    return full_masks


def load_rtsd_instances(image_path: Path) -> list[tuple[int, np.ndarray]]:
    annotation_path = image_path.with_name(f"{image_path.name}_coco.json")
    annotation = _read_annotation(annotation_path)
    image_height, image_width = _image_size(image_path)
    masks = _restore_full_masks(annotation, image_height, image_width)

    instances: list[tuple[int, np.ndarray]] = []
    for raw_class_id, mask in zip(annotation["class_ids"], masks):
        class_index = CLASS_ID_TO_INDEX.get(raw_class_id)
        if class_index is None:
            continue
        if mask.sum() == 0:
            continue
        instances.append((class_index, mask))
    return instances


def load_rtsd_binary_mask(image_path: Path) -> np.ndarray:
    image_height, image_width = _image_size(image_path)
    combined_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    for _, instance_mask in load_rtsd_instances(image_path):
        combined_mask = np.maximum(combined_mask, instance_mask.astype(np.uint8))
    return combined_mask


def load_binary_mask(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as image:
        mask = np.array(image.convert("L"))
    return (mask > 0).astype(np.uint8)


def _mask_to_yolo_polygon(
    mask: np.ndarray,
    image_height: int,
    image_width: int,
    min_area: float = 10.0,
) -> list[float] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None

    points = contour.squeeze(axis=1)
    if points.ndim != 2 or len(points) < 3:
        return None

    polygon: list[float] = []
    for x, y in points:
        polygon.append(float(x) / image_width)
        polygon.append(float(y) / image_height)
    return polygon


def dataset_summary(raw_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for split in ("train", "val"):
        image_paths = sorted((raw_root / split).glob("*.jpg"))
        instance_count = 0
        class_counter = {name: 0 for name in CLASS_NAMES}

        for image_path in image_paths:
            instances = load_rtsd_instances(image_path)
            instance_count += len(instances)
            for class_index, _ in instances:
                class_counter[CLASS_NAMES[class_index]] += 1

        row = {
            "split": split,
            "images": len(image_paths),
            "instances": instance_count,
        }
        row.update(class_counter)
        rows.append(row)

    return pd.DataFrame(rows)


def prepare_yolo_dataset(
    raw_root: Path,
    output_root: Path,
    overwrite: bool = False,
) -> PreparedDataset:
    raw_root = Path(raw_root)
    output_root = Path(output_root)

    if overwrite and output_root.exists():
        shutil.rmtree(output_root)

    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        image_paths = sorted((raw_root / split).glob("*.jpg"))
        for image_path in tqdm(image_paths, desc=f"Preparing {split}", leave=False):
            target_image_path = output_root / "images" / split / image_path.name
            shutil.copy2(image_path, target_image_path)

            image_height, image_width = _image_size(image_path)
            label_lines: list[str] = []
            for class_index, mask in load_rtsd_instances(image_path):
                polygon = _mask_to_yolo_polygon(mask, image_height, image_width)
                if polygon is None:
                    continue
                polygon_str = " ".join(f"{value:.6f}" for value in polygon)
                label_lines.append(f"{class_index} {polygon_str}")

            target_label_path = output_root / "labels" / split / f"{image_path.stem}.txt"
            target_label_path.write_text("\n".join(label_lines), encoding="utf-8")

    data_yaml = output_root / "dataset.yaml"
    yaml_text = "\n".join(
        [
            f"path: {output_root.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            *[f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)],
            "",
        ]
    )
    data_yaml.write_text(yaml_text, encoding="utf-8")
    return PreparedDataset(raw_root=raw_root, yolo_root=output_root, data_yaml=data_yaml)


def auto_device() -> str | int:
    try:
        import torch
    except ImportError:
        return "cpu"
    return 0 if torch.cuda.is_available() else "cpu"


def resolve_device(device: str | int | None = None) -> str | int:
    if device is None or device == "auto":
        return auto_device()
    if isinstance(device, int):
        return device

    normalized = str(device).strip().lower()
    if normalized == "cpu":
        return "cpu"

    if normalized.startswith("cuda"):
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is not installed. Cannot use CUDA.") from error

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but current PyTorch build does not see a CUDA device. "
                "Install a CUDA-enabled PyTorch build in the virtual environment."
            )
        return device

    return device


def train_segmentation_model(
    data_yaml: Path,
    output_root: Path,
    model_name: str = "yolo11n-seg.pt",
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 8,
    run_name: str = "road_signs_seg",
    device: str | int | None = None,
) -> Path:
    model = YOLO(model_name)
    train_result = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(output_root),
        name=run_name,
        exist_ok=True,
        seed=42,
        workers=0,
        device=resolve_device(device),
        verbose=True,
    )
    return Path(train_result.save_dir)


def plot_training_curves(run_dir: Path, save_path: Path | None = None) -> Path:
    run_dir = Path(run_dir)
    results_csv = run_dir / "results.csv"
    results_df = pd.read_csv(results_csv)

    loss_columns = [column for column in results_df.columns if "loss" in column]
    metric_columns = [
        column for column in results_df.columns if column.startswith("metrics/")
    ]

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for column in loss_columns:
        axes[0].plot(results_df["epoch"], results_df[column], label=column)
    axes[0].set_title("Training Losses")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for column in metric_columns:
        axes[1].plot(results_df["epoch"], results_df[column], label=column)
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    figure.tight_layout()

    if save_path is None:
        save_path = run_dir / "training_curves.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return save_path


def _result_to_binary_mask(result) -> np.ndarray:
    binary_mask = np.zeros(result.orig_shape, dtype=np.uint8)
    if result.masks is None:
        return binary_mask

    for segment in result.masks.xy:
        points = np.round(segment).astype(np.int32)
        if points.ndim != 2 or len(points) < 3:
            continue
        cv2.fillPoly(binary_mask, [points], 1)
    return binary_mask


def predict_binary_masks(
    model_source: str | Path,
    image_paths: Iterable[Path],
    imgsz: int = 640,
    conf: float = 0.25,
    device: str | int | None = None,
) -> list[np.ndarray]:
    model = YOLO(str(model_source))
    image_paths = [Path(image_path) for image_path in image_paths]
    results = model.predict(
        source=[str(image_path) for image_path in image_paths],
        imgsz=imgsz,
        conf=conf,
        device=resolve_device(device),
        verbose=False,
    )
    return [_result_to_binary_mask(result) for result in results]


def save_prediction_overlays(
    model_source: str | Path,
    image_paths: Iterable[Path],
    output_dir: Path,
    imgsz: int = 640,
    conf: float = 0.25,
    device: str | int | None = None,
) -> list[Path]:
    model = YOLO(str(model_source))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    image_paths = [Path(image_path) for image_path in image_paths]

    results = model.predict(
        source=[str(image_path) for image_path in image_paths],
        imgsz=imgsz,
        conf=conf,
        device=resolve_device(device),
        verbose=False,
    )

    for image_path, result in zip(image_paths, results):
        plotted = result.plot()
        save_path = output_dir / Path(image_path).name
        cv2.imwrite(str(save_path), plotted)
        saved_paths.append(save_path)

    return saved_paths


def save_binary_masks(
    binary_masks: Iterable[np.ndarray],
    image_paths: Iterable[Path],
    output_dir: Path,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for image_path, binary_mask in zip(image_paths, binary_masks):
        save_path = output_dir / f"{Path(image_path).stem}.png"
        mask_to_save = ((binary_mask > 0).astype(np.uint8) * 255)
        Image.fromarray(mask_to_save).save(save_path)
        saved_paths.append(save_path)

    return saved_paths


def export_custom_predictions(
    model_source: str | Path,
    images_dir: Path,
    output_dir: Path,
    imgsz: int = 640,
    conf: float = 0.25,
    device: str | int | None = None,
) -> pd.DataFrame:
    image_paths = sorted(Path(images_dir).glob("*"))
    image_paths = [
        path for path in image_paths if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]

    if not image_paths:
        return pd.DataFrame(columns=["image_name", "predicted_pixels", "mask_path", "overlay_path"])

    binary_masks = predict_binary_masks(
        model_source=model_source,
        image_paths=image_paths,
        imgsz=imgsz,
        conf=conf,
        device=device,
    )
    mask_paths = save_binary_masks(
        binary_masks=binary_masks,
        image_paths=image_paths,
        output_dir=Path(output_dir) / "predicted_masks",
    )
    overlay_paths = save_prediction_overlays(
        model_source=model_source,
        image_paths=image_paths,
        output_dir=Path(output_dir) / "overlays",
        imgsz=imgsz,
        conf=conf,
        device=device,
    )

    rows: list[dict] = []
    for image_path, binary_mask, mask_path, overlay_path in zip(
        image_paths, binary_masks, mask_paths, overlay_paths
    ):
        rows.append(
            {
                "image_name": image_path.name,
                "predicted_pixels": int((binary_mask > 0).sum()),
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
            }
        )

    predictions_df = pd.DataFrame(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_dir / "custom_predictions.csv", index=False)
    return predictions_df


def compute_mask_metrics(
    ground_truth_mask: np.ndarray,
    predicted_mask: np.ndarray,
) -> dict[str, float]:
    ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)
    predicted_mask = (predicted_mask > 0).astype(np.uint8)

    intersection = np.logical_and(ground_truth_mask, predicted_mask).sum()
    union = np.logical_or(ground_truth_mask, predicted_mask).sum()
    true_positive = intersection
    false_positive = np.logical_and(predicted_mask == 1, ground_truth_mask == 0).sum()
    false_negative = np.logical_and(predicted_mask == 0, ground_truth_mask == 1).sum()

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    iou = intersection / union if union else 1.0
    l2 = float(np.sqrt(np.mean((predicted_mask.astype(np.float32) - ground_truth_mask.astype(np.float32)) ** 2)))

    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "l2": l2,
    }


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "images": int(len(metrics_df)),
                "mean_iou": float(metrics_df["iou"].mean()),
                "mean_precision": float(metrics_df["precision"].mean()),
                "mean_recall": float(metrics_df["recall"].mean()),
                "mean_l2": float(metrics_df["l2"].mean()),
                "share_iou_ge_0_5": float((metrics_df["iou"] >= 0.5).mean()),
                "share_iou_ge_0_75": float((metrics_df["iou"] >= 0.75).mean()),
                "share_iou_ge_0_9": float((metrics_df["iou"] >= 0.9).mean()),
            }
        ]
    )


def evaluate_rtsd_validation(
    model_source: str | Path,
    raw_root: Path,
    output_dir: Path,
    imgsz: int = 640,
    conf: float = 0.25,
    limit: int | None = None,
    device: str | int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    image_paths = sorted((Path(raw_root) / "val").glob("*.jpg"))
    if limit is not None:
        image_paths = image_paths[:limit]

    predicted_masks = predict_binary_masks(
        model_source=model_source,
        image_paths=image_paths,
        imgsz=imgsz,
        conf=conf,
        device=device,
    )

    rows: list[dict] = []
    for image_path, predicted_mask in zip(image_paths, predicted_masks):
        ground_truth_mask = load_rtsd_binary_mask(image_path)
        metrics = compute_mask_metrics(ground_truth_mask, predicted_mask)
        metrics["image_name"] = image_path.name
        rows.append(metrics)

    metrics_df = pd.DataFrame(rows)
    summary_df = summarize_metrics(metrics_df)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "validation_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "validation_summary.csv", index=False)
    return metrics_df, summary_df


def _find_mask_path(image_path: Path, masks_dir: Path) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        candidate = masks_dir / f"{image_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Mask not found for {image_path.name}")


def evaluate_custom_images(
    model_source: str | Path,
    images_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    imgsz: int = 640,
    conf: float = 0.25,
    device: str | int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    image_paths = sorted(Path(images_dir).glob("*"))
    image_paths = [path for path in image_paths if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]

    predicted_masks = predict_binary_masks(
        model_source=model_source,
        image_paths=image_paths,
        imgsz=imgsz,
        conf=conf,
        device=device,
    )

    rows: list[dict] = []
    for image_path, predicted_mask in zip(image_paths, predicted_masks):
        mask_path = _find_mask_path(image_path, Path(masks_dir))
        ground_truth_mask = load_binary_mask(mask_path)
        if ground_truth_mask.shape != predicted_mask.shape:
            ground_truth_mask = cv2.resize(
                ground_truth_mask.astype(np.uint8),
                (predicted_mask.shape[1], predicted_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        metrics = compute_mask_metrics(ground_truth_mask, predicted_mask)
        metrics["image_name"] = image_path.name
        rows.append(metrics)

    metrics_df = pd.DataFrame(rows)
    summary_df = summarize_metrics(metrics_df)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "custom_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "custom_summary.csv", index=False)
    return metrics_df, summary_df


def plot_iou_thresholds(
    summary_df: pd.DataFrame,
    title: str,
    save_path: Path | None = None,
) -> Path | None:
    values = [
        float(summary_df.iloc[0]["share_iou_ge_0_5"]),
        float(summary_df.iloc[0]["share_iou_ge_0_75"]),
        float(summary_df.iloc[0]["share_iou_ge_0_9"]),
    ]
    labels = ["IoU >= 0.5", "IoU >= 0.75", "IoU >= 0.9"]

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(labels, values, color=["#2E86AB", "#F18F01", "#C73E1D"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Share of Images")
    axis.set_title(title)
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()

    if save_path is None:
        return None

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return save_path


def preview_ground_truth(
    raw_root: Path,
    split: str = "train",
    limit: int = 4,
) -> None:
    image_paths = sorted((Path(raw_root) / split).glob("*.jpg"))[:limit]
    if not image_paths:
        return

    figure, axes = plt.subplots(1, len(image_paths), figsize=(5 * len(image_paths), 5))
    if len(image_paths) == 1:
        axes = [axes]

    for axis, image_path in zip(axes, image_paths):
        with Image.open(image_path) as image:
            image_array = np.array(image.convert("RGB"))
        mask = load_rtsd_binary_mask(image_path)

        overlay = image_array.copy()
        overlay[mask > 0] = [255, 80, 80]

        axis.imshow(cv2.addWeighted(image_array, 0.65, overlay, 0.35, 0))
        axis.set_title(image_path.name)
        axis.axis("off")

    figure.tight_layout()
