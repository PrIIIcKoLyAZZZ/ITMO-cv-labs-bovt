from __future__ import annotations

import argparse
from pathlib import Path

from lab6_pipeline import plot_training_curves, train_segmentation_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=Path("data") / "yolo_dataset" / "dataset.yaml",
    )
    parser.add_argument("--model-name", default="yolo11n-seg.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--run-name", default="road_signs_seg")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    run_dir = train_segmentation_model(
        data_yaml=args.data_yaml,
        output_root=args.output_root,
        model_name=args.model_name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        run_name=args.run_name,
        device=args.device,
    )
    curve_path = plot_training_curves(run_dir)
    print(f"Run directory: {run_dir}")
    print(f"Training curves: {curve_path}")


if __name__ == "__main__":
    main()
