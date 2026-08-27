from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from celeba_gan.config import ExperimentConfig
from celeba_gan.data import CelebAFacesDataset
from celeba_gan.models import Critic, Generator
from celeba_gan.plotting import save_training_curves
from celeba_gan.utils import choose_device, ensure_dir, save_image_grid, set_seed, weights_init, write_json


def make_dataloader(config: ExperimentConfig) -> DataLoader:
    dataset = CelebAFacesDataset(
        processed_root=config.data.processed_root,
        metadata_path=config.data.metadata_path,
        split="train",
        image_size=config.data.image_size,
        condition_attr=config.data.condition_attr,
        conditional=False,
    )
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def gradient_penalty(
    critic: Critic,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batch_size = real_images.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
    mixed_scores = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def prepare_run_dir(config: ExperimentConfig) -> Path:
    run_dir = ensure_dir(Path(config.train.output_root) / config.train.run_name)
    ensure_dir(run_dir / "samples")
    ensure_dir(run_dir / "checkpoints")
    write_json(config.to_dict(), run_dir / "config.json")
    return run_dir


def save_checkpoint(
    path: Path,
    generator: Generator,
    critic: Critic,
    g_optimizer: torch.optim.Optimizer,
    c_optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    epoch: int,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "config": config.to_dict(),
            "generator_state": generator.state_dict(),
            "critic_state": critic.state_dict(),
            "g_optimizer_state": g_optimizer.state_dict(),
            "c_optimizer_state": c_optimizer.state_dict(),
        },
        path,
    )


def train_wgan(config: ExperimentConfig, n_critic: int, lambda_gp: float) -> None:
    set_seed(config.train.seed)
    device = choose_device()
    dataloader = make_dataloader(config)
    run_dir = prepare_run_dir(config)

    generator = Generator(
        latent_dim=config.model.latent_dim,
        base_channels=config.model.base_channels,
        image_channels=config.model.image_channels,
        conditional=False,
    ).to(device)
    critic = Critic(
        base_channels=config.model.base_channels,
        image_channels=config.model.image_channels,
        conditional=False,
    ).to(device)
    generator.apply(weights_init)
    critic.apply(weights_init)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=config.train.lr, betas=(0.0, 0.9))
    c_optimizer = torch.optim.Adam(critic.parameters(), lr=config.train.lr, betas=(0.0, 0.9))
    scaler = GradScaler(enabled=config.train.amp and device.type == "cuda")

    fixed_noise = torch.randn(config.train.fixed_sample_count, config.model.latent_dim, device=device)
    history_path = run_dir / "history.csv"
    history_rows: list[dict] = []
    global_step = 0
    start_epoch = 1

    if config.train.init_checkpoint is not None:
        checkpoint = torch.load(config.train.init_checkpoint, map_location=device, weights_only=False)
        generator.load_state_dict(checkpoint["generator_state"])
        critic.load_state_dict(checkpoint["critic_state"])
        if "g_optimizer_state" in checkpoint:
            g_optimizer.load_state_dict(checkpoint["g_optimizer_state"])
        if "c_optimizer_state" in checkpoint:
            c_optimizer.load_state_dict(checkpoint["c_optimizer_state"])

        start_epoch = int(checkpoint["epoch"]) + 1
        if history_path.exists():
            history = pd.read_csv(history_path)
            history_rows = history.to_dict("records")
            if not history.empty:
                global_step = int(history["global_step"].max())

    for epoch in range(start_epoch, config.train.epochs + 1):
        progress = tqdm(dataloader, desc=f"WGAN-GP Epoch {epoch}/{config.train.epochs}")
        for real_images in progress:
            real_images = real_images.to(device, non_blocking=True)
            batch_size = real_images.size(0)

            c_loss_value = 0.0
            wasserstein_value = 0.0
            gp_value = 0.0
            real_mean_value = 0.0
            fake_mean_value = 0.0

            for _ in range(n_critic):
                noise = torch.randn(batch_size, config.model.latent_dim, device=device)
                c_optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=scaler.is_enabled()):
                    fake_images = generator(noise).detach()
                    real_scores = critic(real_images)
                    fake_scores = critic(fake_images)
                    gp = gradient_penalty(critic, real_images, fake_images, device)
                    wasserstein = fake_scores.mean() - real_scores.mean()
                    c_loss = wasserstein + lambda_gp * gp

                scaler.scale(c_loss).backward()
                scaler.step(c_optimizer)
                scaler.update()

                if not torch.isfinite(c_loss):
                    raise FloatingPointError(
                        f"Non-finite critic loss at epoch={epoch}, step={global_step + 1}: {c_loss.item()}"
                    )

                c_loss_value = float(c_loss.item())
                wasserstein_value = float(wasserstein.item())
                gp_value = float(gp.item())
                real_mean_value = float(real_scores.mean().item())
                fake_mean_value = float(fake_scores.mean().item())

            noise = torch.randn(batch_size, config.model.latent_dim, device=device)
            g_optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                generated = generator(noise)
                g_loss = -critic(generated).mean()
            scaler.scale(g_loss).backward()
            scaler.step(g_optimizer)
            scaler.update()

            if not torch.isfinite(g_loss):
                raise FloatingPointError(
                    f"Non-finite generator loss at epoch={epoch}, step={global_step + 1}: {g_loss.item()}"
                )

            global_step += 1
            if global_step % config.train.log_every == 0 or global_step == 1:
                history_rows.append(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "d_loss": c_loss_value,
                        "g_loss": float(g_loss.item()),
                        "d_real_mean": real_mean_value,
                        "d_fake_mean": fake_mean_value,
                        "wasserstein": wasserstein_value,
                        "gradient_penalty": gp_value,
                    }
                )
                progress.set_postfix(c_loss=c_loss_value, g_loss=float(g_loss.item()), gp=gp_value)

        if epoch % config.train.sample_every == 0:
            generator.eval()
            with torch.no_grad():
                sample_batch = generator(fixed_noise)
            save_image_grid(sample_batch, run_dir / "samples" / f"epoch_{epoch:03d}.png")
            generator.train()

        if epoch % config.train.save_every == 0 or epoch == config.train.epochs:
            save_checkpoint(
                run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt",
                generator,
                critic,
                g_optimizer,
                c_optimizer,
                config,
                epoch,
            )

    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    save_training_curves(history_path, run_dir / "training_curves.png")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train WGAN-GP on CelebA faces.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n-critic", type=int, default=3)
    parser.add_argument("--lambda-gp", type=float, default=10.0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    train_wgan(config, n_critic=args.n_critic, lambda_gp=args.lambda_gp)


if __name__ == "__main__":
    main()
