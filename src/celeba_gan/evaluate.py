from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from celeba_gan.config import DataConfig, EvalConfig, ExperimentConfig, ModelConfig, TrainConfig
from celeba_gan.data import CelebAFacesDataset
from celeba_gan.metrics import InceptionFeatureExtractor, MetricResult, compute_fid, compute_inception_score
from celeba_gan.models import Generator
from celeba_gan.utils import choose_device, write_json


def config_from_checkpoint(payload: dict) -> ExperimentConfig:
    cfg = payload["config"]
    return ExperimentConfig(
        data=DataConfig(**cfg["data"]),
        model=ModelConfig(**cfg["model"]),
        train=TrainConfig(**cfg["train"]),
        eval=EvalConfig(**cfg["eval"]),
    )


def gather_real_batches(loader: DataLoader, conditional: bool):
    for batch in loader:
        if conditional:
            images, _ = batch
            yield images
        else:
            yield batch


def gather_fake_batches(
    generator: Generator,
    total_samples: int,
    batch_size: int,
    latent_dim: int,
    device: torch.device,
    conditional: bool,
    num_classes: int,
):
    generated = 0
    while generated < total_samples:
        current_batch = min(batch_size, total_samples - generated)
        noise = torch.randn(current_batch, latent_dim, device=device)
        labels = torch.randint(0, num_classes, (current_batch,), device=device) if conditional else None
        with torch.no_grad():
            fake_images = generator(noise, labels).cpu()
        yield fake_images
        generated += current_batch


def collect_features_and_probs(
    extractor: InceptionFeatureExtractor,
    image_batches,
    device: torch.device,
    keep_probs: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    all_features: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    for images in tqdm(image_batches, desc="Collecting Inception statistics"):
        images = images.to(device)
        with torch.no_grad():
            features, probs = extractor(images)
        all_features.append(features.cpu().numpy())
        if keep_probs:
            all_probs.append(probs.cpu().numpy())

    feature_array = np.concatenate(all_features, axis=0)
    prob_array = np.concatenate(all_probs, axis=0) if keep_probs else None
    return feature_array, prob_array


def evaluate(checkpoint_path: Path, output_path: Path) -> MetricResult:
    device = choose_device()
    payload = torch.load(checkpoint_path, map_location=device)
    config = config_from_checkpoint(payload)

    dataset = CelebAFacesDataset(
        processed_root=config.data.processed_root,
        metadata_path=config.data.metadata_path,
        split=config.eval.split,
        image_size=config.data.image_size,
        condition_attr=config.data.condition_attr,
        conditional=config.model.conditional,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.eval.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    generator = Generator(
        latent_dim=config.model.latent_dim,
        base_channels=config.model.base_channels,
        image_channels=config.model.image_channels,
        conditional=config.model.conditional,
        num_classes=config.model.num_classes,
        class_embed_dim=config.model.class_embed_dim,
    ).to(device)
    generator.load_state_dict(payload["generator_state"])
    generator.eval()

    extractor = InceptionFeatureExtractor().to(device)
    real_features, _ = collect_features_and_probs(
        extractor=extractor,
        image_batches=gather_real_batches(loader, conditional=config.model.conditional),
        device=device,
        keep_probs=False,
    )
    fake_features, fake_probs = collect_features_and_probs(
        extractor=extractor,
        image_batches=gather_fake_batches(
            generator=generator,
            total_samples=config.eval.num_samples,
            batch_size=config.eval.batch_size,
            latent_dim=config.model.latent_dim,
            device=device,
            conditional=config.model.conditional,
            num_classes=config.model.num_classes,
        ),
        device=device,
        keep_probs=True,
    )

    real_features = real_features[: config.eval.num_samples]
    fid = compute_fid(real_features, fake_features)
    is_mean, is_std = compute_inception_score(fake_probs, splits=config.eval.inception_splits)
    result = MetricResult(fid=fid, inception_score_mean=is_mean, inception_score_std=is_std)

    pd.DataFrame(
        [
            {
                "checkpoint": str(checkpoint_path),
                "split": config.eval.split,
                "fid": result.fid,
                "is_mean": result.inception_score_mean,
                "is_std": result.inception_score_std,
                "num_samples": config.eval.num_samples,
            }
        ]
    ).to_csv(output_path, index=False)
    write_json(
        {
            "checkpoint": str(checkpoint_path),
            "split": config.eval.split,
            "fid": result.fid,
            "is_mean": result.inception_score_mean,
            "is_std": result.inception_score_std,
            "num_samples": config.eval.num_samples,
        },
        output_path.with_suffix(".json"),
    )
    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a GAN checkpoint with FID and IS.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to saved generator/discriminator checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="CSV path for evaluation results.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    evaluate(checkpoint_path=args.checkpoint, output_path=args.output)


if __name__ == "__main__":
    main()
