from __future__ import annotations

import argparse
from pathlib import Path

from lab6_pipeline import (
    download_rtsd_segmentation_dataset,
    evaluate_custom_images,
    evaluate_rtsd_validation,
    export_custom_predictions,
    plot_iou_thresholds,
    save_prediction_overlays,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data") / "raw" / "sign_dataset",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("reports"),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--custom-images", type=Path, default=Path("data") / "custom" / "images")
    parser.add_argument("--custom-masks", type=Path, default=Path("data") / "custom" / "masks")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    raw_root = args.raw_root if args.raw_root.exists() else download_rtsd_segmentation_dataset()

    validation_report_dir = args.report_root / "validation"
    validation_metrics, validation_summary = evaluate_rtsd_validation(
        model_source=args.model_path,
        raw_root=raw_root,
        output_dir=validation_report_dir,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
    )
    plot_iou_thresholds(
        validation_summary,
        title="Validation IoU Thresholds",
        save_path=validation_report_dir / "validation_iou_thresholds.png",
    )
    save_prediction_overlays(
        model_source=args.model_path,
        image_paths=sorted((raw_root / "val").glob("*.jpg"))[:10],
        output_dir=validation_report_dir / "overlays",
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
    )
    print(validation_summary.to_string(index=False))

    if args.custom_images.exists() and any(args.custom_images.iterdir()):
        custom_report_dir = args.report_root / "custom"
        predictions_df = export_custom_predictions(
            model_source=args.model_path,
            images_dir=args.custom_images,
            output_dir=custom_report_dir,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
        )
        print(predictions_df.to_string(index=False))

        custom_images = [
            path for path in sorted(args.custom_images.glob("*"))
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        all_masks_exist = True
        for image_path in custom_images:
            stem = image_path.stem
            if not any((args.custom_masks / f"{stem}{suffix}").exists() for suffix in [".png", ".jpg", ".jpeg", ".bmp"]):
                all_masks_exist = False
                break

        if all_masks_exist and custom_images:
            custom_metrics, custom_summary = evaluate_custom_images(
                model_source=args.model_path,
                images_dir=args.custom_images,
                masks_dir=args.custom_masks,
                output_dir=custom_report_dir,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
            )
            plot_iou_thresholds(
                custom_summary,
                title="Custom Photos IoU Thresholds",
                save_path=custom_report_dir / "custom_iou_thresholds.png",
            )
            print(custom_summary.to_string(index=False))
        else:
            print("Ground-truth masks were not found for all custom images. Predictions were saved, metrics were skipped.")


if __name__ == "__main__":
    main()
