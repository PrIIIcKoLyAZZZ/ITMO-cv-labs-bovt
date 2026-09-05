from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the trained detector on local photos.")
    parser.add_argument("--model", type=Path, required=True, help="Path to YOLO weights.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/custom_images"),
        help="Image file or directory with user photos.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/predict/custom"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--batch", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    image_paths = _collect_images(args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detections: list[dict[str, object]] = []
    for batch_paths in _batched(image_paths, args.batch):
        results = model.predict(
            source=[str(path) for path in batch_paths],
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            batch=len(batch_paths),
            verbose=False,
            stream=True,
        )
        for image_path, result in zip(batch_paths, results, strict=True):
            rendered = result.plot()
            output_image = args.output_dir / image_path.name
            cv2.imwrite(str(output_image), rendered)

            boxes = []
            for xyxy, confidence in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), strict=True):
                boxes.append({"bbox_xyxy": [round(float(v), 2) for v in xyxy], "confidence": round(float(confidence), 4)})

            detections.append({"image": image_path.name, "detections": boxes})

    (args.output_dir / "predictions.json").write_text(
        json.dumps(detections, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(args.output_dir.resolve())


def _collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        images = sorted(path for path in source.iterdir() if path.suffix.lower() in VALID_SUFFIXES)
        if images:
            return images
    raise FileNotFoundError(f"No images found in {source.resolve()}")


def _batched(items: list[Path], batch_size: int) -> list[list[Path]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


if __name__ == "__main__":
    main()
