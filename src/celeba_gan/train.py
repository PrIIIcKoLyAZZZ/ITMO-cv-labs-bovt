from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from celeba_gan.config import ExperimentConfig
from celeba_gan.data import CelebAFacesDataset
from celeba_gan.models import Discriminator, Generator
from celeba_gan.plotting import save_training_curves
from celeba_gan.utils import choose_device, ensure_dir, save_image_grid, set_seed, weights_init, write_json


def make_dataloader(config: ExperimentConfig) -> DataLoader:
    dataset = CelebAFacesDataset(
        processed_root=config.data.processed_root,
        metadata_path=config.data.metadata_path,
        split="train",
        image_size=config.data.image_size,
        condition_attr=config.data.condition_attr,
        conditional=config.model.conditional,
    )
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def prepare_run_dir(config: ExperimentConfig) -> Path:
    run_dir = ensure_dir(Path(config.train.output_root) / config.train.run_name)
    ensure_dir(run_dir / "samples")
    ensure_dir(run_dir / "checkpoints")
    write_json(config.to_dict(), run_dir / "config.json")
    return run_dir


def make_models(config: ExperimentConfig, device: torch.device) -> tuple[Generator, Discriminator]:
    generator = Generator(
        latent_dim=config.model.latent_dim,
        base_channels=config.model.base_channels,
        image_channels=config.model.image_channels,
        conditional=config.model.conditional,
        num_classes=config.model.num_classes,
        class_embed_dim=config.model.class_embed_dim,
    ).to(device)
    discriminator = Discriminator(
        base_channels=config.model.base_channels,
        image_channels=config.model.image_channels,
        conditional=config.model.conditional,
        num_classes=config.model.num_classes,
    ).to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)
    return generator, discriminator


def load_compatible_weights(target: torch.nn.Module, source_state: dict[str, torch.Tensor]) -> None:
    target_state = target.state_dict()
    patched_state = {}

    for name, target_tensor in target_state.items():
        source_tensor = source_state.get(name)
        if source_tensor is None:
            patched_state[name] = target_tensor
            continue

        if source_tensor.shape == target_tensor.shape:
            patched_state[name] = source_tensor
            continue

        if name == "net.0.weight" and source_tensor.ndim == target_tensor.ndim:
            patched = target_tensor.clone()
            slices = tuple(slice(0, min(src, dst)) for src, dst in zip(source_tensor.shape, target_tensor.shape))
            patched[slices] = source_tensor[slices]
            patched_state[name] = patched
            continue

        patched_state[name] = target_tensor

    target.load_state_dict(patched_state)


def initialize_from_unconditional_checkpoint(
    generator: Generator,
    discriminator: Discriminator,
    checkpoint_path: str | None,
    device: torch.device,
) -> None:
    if checkpoint_path is None:
        return

    payload = torch.load(checkpoint_path, map_location=device)
    load_compatible_weights(generator, payload["generator_state"])
    load_compatible_weights(discriminator, payload["discriminator_state"])


def sample_labels(batch_size: int, num_classes: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, num_classes, (batch_size,), device=device)


def build_fixed_inputs(config: ExperimentConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    count = config.train.fixed_sample_count
    noise = torch.randn(count, config.model.latent_dim, device=device)
    if not config.model.conditional:
        return noise, None

    pattern = torch.arange(config.model.num_classes, device=device)
    repeats = (count + config.model.num_classes - 1) // config.model.num_classes
    labels = pattern.repeat(repeats)[:count]
    return noise, labels


def save_checkpoint(
    path: Path,
    generator: Generator,
    discriminator: Discriminator,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    epoch: int,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "config": config.to_dict(),
            "generator_state": generator.state_dict(),
            "discriminator_state": discriminator.state_dict(),
            "g_optimizer_state": g_optimizer.state_dict(),
            "d_optimizer_state": d_optimizer.state_dict(),
        },
        path,
    )


def train(config: ExperimentConfig) -> None:
    set_seed(config.train.seed)
    device = choose_device()
    dataloader = make_dataloader(config)
    generator, discriminator = make_models(config, device=device)
    initialize_from_unconditional_checkpoint(
        generator=generator,
        discriminator=discriminator,
        checkpoint_path=config.train.init_checkpoint,
        device=device,
    )
    run_dir = prepare_run_dir(config)

    criterion = nn.BCEWithLogitsLoss()
    g_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=config.train.lr,
        betas=(config.train.beta1, config.train.beta2),
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.train.lr,
        betas=(config.train.beta1, config.train.beta2),
    )

    scaler = GradScaler(enabled=config.train.amp and device.type == "cuda")
    fixed_noise, fixed_labels = build_fixed_inputs(config, device=device)
    history_rows: list[dict] = []
    global_step = 0

    for epoch in range(1, config.train.epochs + 1):
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{config.train.epochs}")
        for batch in progress:
            if config.model.conditional:
                real_images, labels = batch
                labels = labels.to(device, non_blocking=True)
            else:
                real_images = batch
                labels = None

            real_images = real_images.to(device, non_blocking=True)
            batch_size = real_images.size(0)
            real_targets = torch.ones(batch_size, device=device)
            fake_targets = torch.zeros(batch_size, device=device)
            noise = torch.randn(batch_size, config.model.latent_dim, device=device)
            sampled_labels = sample_labels(batch_size, config.model.num_classes, device) if config.model.conditional else None

            d_optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                fake_images = generator(noise, sampled_labels)
                real_logits = discriminator(real_images, labels)
                fake_logits = discriminator(fake_images.detach(), sampled_labels)
                d_loss_real = criterion(real_logits, real_targets)
                d_loss_fake = criterion(fake_logits, fake_targets)
                d_loss = 0.5 * (d_loss_real + d_loss_fake)

            scaler.scale(d_loss).backward()
            scaler.step(d_optimizer)

            g_optimizer.zero_grad(set_to_none=True)
            noise = torch.randn(batch_size, config.model.latent_dim, device=device)
            sampled_labels = sample_labels(batch_size, config.model.num_classes, device) if config.model.conditional else None
            with autocast(enabled=scaler.is_enabled()):
                generated = generator(noise, sampled_labels)
                gen_logits = discriminator(generated, sampled_labels)
                g_loss = criterion(gen_logits, real_targets)

            scaler.scale(g_loss).backward()
            scaler.step(g_optimizer)
            scaler.update()

            global_step += 1
            if global_step % config.train.log_every == 0 or global_step == 1:
                history_rows.append(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "d_loss": float(d_loss.item()),
                        "g_loss": float(g_loss.item()),
                        "d_real_mean": float(real_logits.mean().item()),
                        "d_fake_mean": float(fake_logits.mean().item()),
                    }
                )
                progress.set_postfix(d_loss=float(d_loss.item()), g_loss=float(g_loss.item()))

        if epoch % config.train.sample_every == 0:
            generator.eval()
            with torch.no_grad():
                sample_batch = generator(fixed_noise, fixed_labels)
            save_image_grid(sample_batch, run_dir / "samples" / f"epoch_{epoch:03d}.png")
            generator.train()

        if epoch % config.train.save_every == 0 or epoch == config.train.epochs:
            save_checkpoint(
                run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt",
                generator,
                discriminator,
                g_optimizer,
                d_optimizer,
                config,
                epoch,
            )

    history = pd.DataFrame(history_rows)
    history_path = run_dir / "history.csv"
    history.to_csv(history_path, index=False)
    save_training_curves(history_path, run_dir / "training_curves.png")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DCGAN or conditional DCGAN on cropped CelebA faces.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    train(config)


if __name__ == "__main__":
    main()
