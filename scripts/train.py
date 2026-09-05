from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO detector on the prepared SVHN dataset.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/svhn_yolo/dataset.yaml"))
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base checkpoint or YAML model config.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--name", type=str, default="svhn_number_detector")
    parser.add_argument("--patience", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project.resolve()),
        name=args.name,
        patience=args.patience,
        pretrained=True,
        exist_ok=True,
        seed=42,
        cache=False,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.0,
    )


if __name__ == "__main__":
    main()
