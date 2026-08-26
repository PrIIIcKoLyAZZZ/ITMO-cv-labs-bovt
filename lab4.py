from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ViT_B_16_Weights, vit_b_16
from tqdm import tqdm


@dataclass(slots=True)
class Lab4Config:
    seed: int
    classes: list[str]
    prepared_root: Path
    artifacts_root: Path
    scratch_image_size: int
    scratch_batch_size: int
    scratch_epochs: int
    scratch_lr: float
    scratch_weight_decay: float
    patience: int
    feature_batch_size: int
    num_workers: int

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> "Lab4Config":
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return cls(
            seed=payload["seed"],
            classes=payload["classes"],
            prepared_root=Path(payload["prepared_root"]),
            artifacts_root=Path(payload["artifacts_root"]),
            scratch_image_size=payload["scratch_image_size"],
            scratch_batch_size=payload["scratch_batch_size"],
            scratch_epochs=payload["scratch_epochs"],
            scratch_lr=payload["scratch_lr"],
            scratch_weight_decay=payload["scratch_weight_decay"],
            patience=payload["patience"],
            feature_batch_size=payload["feature_batch_size"],
            num_workers=payload["num_workers"],
        )


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, target, path


class ScratchClassifier(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.features(inputs)
        return self.classifier(outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/prepared_lab4/manifest.json"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_split_table(config: Lab4Config) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name in ("train", "val", "test"):
        split_root = config.prepared_root / split_name
        for class_name in config.classes:
            class_root = split_root / class_name
            rows.append(
                {
                    "split": split_name,
                    "class_name": class_name,
                    "count": sum(1 for path in class_root.iterdir() if path.is_file()),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(config.artifacts_root / "class_distribution.csv", index=False)
    return table


def plot_split_table(table: pd.DataFrame, path: Path) -> None:
    pivot = table.pivot(index="class_name", columns="split", values="count")
    figure, axis = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=axis)
    axis.set_xlabel("Class")
    axis.set_ylabel("Images")
    axis.set_title("Split Distribution")
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    axis.legend(title="Split")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def create_scratch_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size + 16, image_size + 16)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def build_image_folder_loaders(
    config: Lab4Config,
    image_size: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_transform, eval_transform = create_scratch_transforms(image_size)
    train_dataset = ImageFolderWithPaths(config.prepared_root / "train", transform=train_transform)
    val_dataset = ImageFolderWithPaths(config.prepared_root / "val", transform=eval_transform)
    test_dataset = ImageFolderWithPaths(config.prepared_root / "test", transform=eval_transform)
    validate_class_order(train_dataset.classes, config.classes)
    validate_class_order(val_dataset.classes, config.classes)
    validate_class_order(test_dataset.classes, config.classes)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader, test_loader


def build_feature_loader(
    split_root: Path,
    weights: ViT_B_16_Weights,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, list[str]]:
    dataset = ImageFolderWithPaths(split_root, transform=weights.transforms())
    return (
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        dataset.classes,
    )


def validate_class_order(found_classes: list[str], expected_classes: list[str]) -> None:
    if list(found_classes) != list(expected_classes):
        raise ValueError(f"Class order mismatch: {found_classes} != {expected_classes}")


def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    description: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    collected_features: list[np.ndarray] = []
    collected_targets: list[np.ndarray] = []
    collected_paths: list[str] = []
    model.eval()
    with torch.inference_mode():
        for images, targets, paths in tqdm(loader, desc=description):
            outputs = model(images.to(device))
            collected_features.append(outputs.cpu().numpy())
            collected_targets.append(targets.numpy())
            collected_paths.extend(paths)
    features = np.concatenate(collected_features, axis=0)
    targets = np.concatenate(collected_targets, axis=0)
    return features, targets, np.array(collected_paths)


def load_or_extract_feature_pack(
    split_name: str,
    model: nn.Module,
    weights: ViT_B_16_Weights,
    config: Lab4Config,
    output_root: Path,
    force: bool,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_path = output_root / f"{split_name}_features.npz"
    if feature_path.exists() and not force:
        payload = np.load(feature_path, allow_pickle=True)
        return payload["features"], payload["targets"], payload["paths"]
    loader, found_classes = build_feature_loader(
        config.prepared_root / split_name,
        weights,
        batch_size=batch_size,
        num_workers=config.num_workers,
    )
    validate_class_order(found_classes, config.classes)
    features, targets, paths = extract_features(
        model=model,
        loader=loader,
        device=device,
        description=f"features:{output_root.name}:{split_name}",
    )
    np.savez_compressed(feature_path, features=features, targets=targets, paths=paths)
    return features, targets, paths


def fit_linear_classifier(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    seed: int,
) -> tuple[Pipeline, dict[str, float]]:
    best_model: Pipeline | None = None
    best_score = -1.0
    best_c = 1.0
    for c_value in (0.1, 1.0, 3.0):
        candidate = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=c_value,
                        max_iter=2000,
                        random_state=seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        candidate.fit(train_features, train_targets)
        val_predictions = candidate.predict(val_features)
        score = f1_score(val_targets, val_predictions, average="macro")
        if score > best_score:
            best_score = score
            best_c = c_value
            best_model = candidate
    assert best_model is not None
    return best_model, {"selected_c": best_c, "val_f1_macro": float(best_score)}


def compute_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
) -> tuple[dict[str, float], pd.DataFrame]:
    metrics = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "f1_macro": float(f1_score(targets, predictions, average="macro")),
    }
    report = classification_report(
        targets,
        predictions,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    report_frame = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"})
    return metrics, report_frame


def save_predictions(
    path: Path,
    sample_paths: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
) -> None:
    frame = pd.DataFrame(
        {
            "path": sample_paths,
            "true_label": [class_names[index] for index in targets],
            "predicted_label": [class_names[index] for index in predictions],
        }
    )
    frame.to_csv(path, index=False)


def save_confusion_plot(
    path: Path,
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    title: str,
) -> None:
    matrix = confusion_matrix(targets, predictions, labels=np.arange(len(class_names)))
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha="right")
    axis.set_yticklabels(class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
                color="black",
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def run_transfer_experiment(
    experiment_name: str,
    weights: ViT_B_16_Weights,
    config: Lab4Config,
    device: torch.device,
    force: bool,
) -> dict[str, float | str]:
    experiment_root = config.artifacts_root / experiment_name
    ensure_directory(experiment_root)
    model = vit_b_16(weights=weights)
    model.heads = nn.Identity()
    model.to(device)
    batch_size = config.feature_batch_size if device.type == "cuda" else min(config.feature_batch_size, 16)
    train_features, train_targets, _ = load_or_extract_feature_pack(
        split_name="train",
        model=model,
        weights=weights,
        config=config,
        output_root=experiment_root,
        force=force,
        batch_size=batch_size,
        device=device,
    )
    val_features, val_targets, _ = load_or_extract_feature_pack(
        split_name="val",
        model=model,
        weights=weights,
        config=config,
        output_root=experiment_root,
        force=force,
        batch_size=batch_size,
        device=device,
    )
    test_features, test_targets, test_paths = load_or_extract_feature_pack(
        split_name="test",
        model=model,
        weights=weights,
        config=config,
        output_root=experiment_root,
        force=force,
        batch_size=batch_size,
        device=device,
    )
    classifier, tuning_metrics = fit_linear_classifier(
        train_features=train_features,
        train_targets=train_targets,
        val_features=val_features,
        val_targets=val_targets,
        seed=config.seed,
    )
    test_predictions = classifier.predict(test_features)
    metrics, report = compute_metrics(test_targets, test_predictions, config.classes)
    save_predictions(
        path=config.artifacts_root / f"{experiment_name}_test_predictions.csv",
        sample_paths=test_paths,
        targets=test_targets,
        predictions=test_predictions,
        class_names=config.classes,
    )
    report.to_csv(config.artifacts_root / f"{experiment_name}_report.csv", index=False)
    save_confusion_plot(
        path=config.artifacts_root / f"{experiment_name}_confusion.png",
        targets=test_targets,
        predictions=test_predictions,
        class_names=config.classes,
        title=experiment_name,
    )
    payload = {
        "model": experiment_name,
        "weights": weights.name,
        "selected_c": tuning_metrics["selected_c"],
        "val_f1_macro": tuning_metrics["val_f1_macro"],
        "test_accuracy": metrics["accuracy"],
        "test_f1_macro": metrics["f1_macro"],
    }
    save_json(config.artifacts_root / f"{experiment_name}_metrics.json", payload)
    return payload


def evaluate_torch_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    description: str,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    collected_targets: list[np.ndarray] = []
    collected_predictions: list[np.ndarray] = []
    collected_paths: list[str] = []
    collected_losses: list[float] = []
    criterion = nn.CrossEntropyLoss()
    model.eval()
    with torch.inference_mode():
        for images, targets, paths in tqdm(loader, desc=description):
            logits = model(images.to(device))
            loss = criterion(logits, targets.to(device))
            predictions = torch.argmax(logits, dim=1)
            collected_losses.append(loss.item())
            collected_targets.append(targets.numpy())
            collected_predictions.append(predictions.cpu().numpy())
            collected_paths.extend(paths)
    targets_array = np.concatenate(collected_targets, axis=0)
    predictions_array = np.concatenate(collected_predictions, axis=0)
    metrics, report = compute_metrics(targets_array, predictions_array, class_names)
    metrics["loss"] = float(np.mean(collected_losses))
    return metrics, report, targets_array, predictions_array, np.array(collected_paths)


def train_scratch_model(
    config: Lab4Config,
    device: torch.device,
    force: bool,
) -> dict[str, float | int | str]:
    checkpoint_path = config.artifacts_root / "scratch_best.pt"
    history_path = config.artifacts_root / "scratch_history.csv"
    train_loader, val_loader, test_loader = build_image_folder_loaders(
        config=config,
        image_size=config.scratch_image_size,
        batch_size=config.scratch_batch_size,
    )
    model = ScratchClassifier(num_classes=len(config.classes)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.scratch_lr,
        weight_decay=config.scratch_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )
    criterion = nn.CrossEntropyLoss()
    history_rows: list[dict[str, float | int]] = []
    best_epoch = 0
    best_val_f1 = -1.0
    patience_counter = 0
    if checkpoint_path.exists() and history_path.exists() and not force:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        history_rows = pd.read_csv(history_path).to_dict(orient="records")
        if history_rows:
            best_row = max(history_rows, key=lambda row: row["val_f1_macro"])
            best_epoch = int(best_row["epoch"])
            best_val_f1 = float(best_row["val_f1_macro"])
    else:
        for epoch in range(1, config.scratch_epochs + 1):
            model.train()
            batch_losses: list[float] = []
            progress = tqdm(train_loader, desc=f"scratch:train:{epoch}")
            for images, targets, _ in progress:
                images = images.to(device)
                targets = targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
                progress.set_postfix(loss=f"{loss.item():.4f}")
            val_metrics, _, _, _, _ = evaluate_torch_model(
                model=model,
                loader=val_loader,
                device=device,
                class_names=config.classes,
                description=f"scratch:val:{epoch}",
            )
            scheduler.step(val_metrics["f1_macro"])
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(batch_losses)),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
            }
            history_rows.append(row)
            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = float(val_metrics["f1_macro"])
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
            if patience_counter >= config.patience:
                break
        pd.DataFrame(history_rows).to_csv(history_path, index=False)
    best_state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_state)
    test_metrics, report, test_targets, test_predictions, test_paths = evaluate_torch_model(
        model=model,
        loader=test_loader,
        device=device,
        class_names=config.classes,
        description="scratch:test",
    )
    save_predictions(
        path=config.artifacts_root / "scratch_test_predictions.csv",
        sample_paths=test_paths,
        targets=test_targets,
        predictions=test_predictions,
        class_names=config.classes,
    )
    report.to_csv(config.artifacts_root / "from_scratch_report.csv", index=False)
    save_confusion_plot(
        path=config.artifacts_root / "from_scratch_confusion.png",
        targets=test_targets,
        predictions=test_predictions,
        class_names=config.classes,
        title="from_scratch",
    )
    history_frame = pd.DataFrame(history_rows)
    if not history_frame.empty:
        plot_learning_curve(history_frame, config.artifacts_root / "scratch_learning_curve.png")
    stats_payload = {
        "model": "scratch_cnn",
        "image_size": config.scratch_image_size,
        "epochs_trained": int(len(history_rows)),
        "best_epoch": int(best_epoch),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    metrics_payload = {
        "model": "scratch_cnn",
        "test_accuracy": test_metrics["accuracy"],
        "test_f1_macro": test_metrics["f1_macro"],
        "best_val_f1_macro": best_val_f1,
        "best_epoch": int(best_epoch),
    }
    save_json(config.artifacts_root / "scratch_stats.json", stats_payload)
    save_json(config.artifacts_root / "scratch_metrics.json", metrics_payload)
    return metrics_payload


def plot_learning_curve(history_frame: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(history_frame["epoch"], history_frame["train_loss"], label="train_loss")
    axis.plot(history_frame["epoch"], history_frame["val_loss"], label="val_loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("Scratch Model Learning Curve")
    axis.grid(alpha=0.3, linestyle="--")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def build_conclusion(summary_frame: pd.DataFrame) -> dict[str, object]:
    best_row = summary_frame.sort_values("test_f1_macro", ascending=False).iloc[0]
    worst_row = summary_frame.sort_values("test_f1_macro", ascending=True).iloc[0]
    return {
        "best_model": best_row["model"],
        "best_test_f1_macro": float(best_row["test_f1_macro"]),
        "worst_model": worst_row["model"],
        "worst_test_f1_macro": float(worst_row["test_f1_macro"]),
        "quality_interpretation": "F1_macro averages F1 across all color classes and does not let a single class dominate the final score.",
    }


def main() -> None:
    started_at = time.time()
    args = parse_args()
    config = Lab4Config.from_manifest(args.manifest)
    set_seed(config.seed)
    device = resolve_device()
    ensure_directory(config.artifacts_root)
    split_table = build_split_table(config)
    plot_split_table(split_table, config.artifacts_root / "class_distribution.png")
    imagenet_metrics = run_transfer_experiment(
        experiment_name="vit_b16_imagenet",
        weights=ViT_B_16_Weights.IMAGENET1K_V1,
        config=config,
        device=device,
        force=args.force,
    )
    swag_metrics = run_transfer_experiment(
        experiment_name="vit_b16_swag",
        weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1,
        config=config,
        device=device,
        force=args.force,
    )
    scratch_metrics = train_scratch_model(
        config=config,
        device=device,
        force=args.force,
    )
    summary = pd.DataFrame(
        [
            {
                "model": "vit_b16_imagenet",
                "test_accuracy": imagenet_metrics["test_accuracy"],
                "test_f1_macro": imagenet_metrics["test_f1_macro"],
            },
            {
                "model": "vit_b16_swag",
                "test_accuracy": swag_metrics["test_accuracy"],
                "test_f1_macro": swag_metrics["test_f1_macro"],
            },
            {
                "model": "scratch_cnn",
                "test_accuracy": scratch_metrics["test_accuracy"],
                "test_f1_macro": scratch_metrics["test_f1_macro"],
            },
        ]
    ).sort_values("test_f1_macro", ascending=False)
    summary.to_csv(config.artifacts_root / "results_summary.csv", index=False)
    conclusion = build_conclusion(summary)
    conclusion["device"] = str(device)
    conclusion["elapsed_minutes"] = round((time.time() - started_at) / 60, 2)
    save_json(config.artifacts_root / "conclusion.json", conclusion)


if __name__ == "__main__":
    main()
