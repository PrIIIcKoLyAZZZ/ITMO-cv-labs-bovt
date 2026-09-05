from __future__ import annotations

import argparse
import json
from pathlib import Path

from svhn_detector.evaluation import evaluate_on_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained detector on the SVHN test split.")
    parser.add_argument("--model", type=Path, required=True, help="Path to YOLO weights, e.g. runs/.../weights/best.pt")
    parser.add_argument("--data", type=Path, default=Path("data/processed/svhn_yolo/dataset.yaml"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--predict-batch", type=int, default=32, help="Batch size for custom IoU/precision/recall pass.")
    parser.add_argument("--output-json", type=Path, default=Path("runs/eval/test_metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_on_test_split(
        model_path=args.model,
        data_yaml=args.data,
        conf=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        predict_batch=args.predict_batch,
        output_json=args.output_json,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
