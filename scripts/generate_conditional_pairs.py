from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.utils import make_grid

from celeba_gan.config import ExperimentConfig
from celeba_gan.models import Generator
from celeba_gan.utils import denormalize


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate pairs from identical noise with different conditional labels.")
    parser.add_argument("--config", type=Path, required=True, help="Conditional GAN config.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Conditional GAN checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG.")
    parser.add_argument("--pairs", type=int, default=16, help="Number of noise vectors to compare.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for fixed noise.")
    return parser


def tensor_to_pil_grid(images: torch.Tensor, nrow: int) -> Image.Image:
    grid = make_grid(denormalize(images), nrow=nrow, padding=2)
    grid = grid.mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(grid)


def add_title(image: Image.Image, title: str) -> Image.Image:
    title_height = 26
    canvas = Image.new("RGB", (image.width, image.height + title_height), "white")
    canvas.paste(image, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill="black")
    return canvas


def generate_pairs(config_path: Path, checkpoint_path: Path, output_path: Path, pairs: int, seed: int) -> None:
    config = ExperimentConfig.from_yaml(config_path)
    if not config.model.conditional:
        raise ValueError("The supplied config must describe a conditional generator.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location=device)

    generator = Generator(
        latent_dim=config.model.latent_dim,
        base_channels=config.model.base_channels,
        image_channels=config.model.image_channels,
        conditional=True,
        num_classes=config.model.num_classes,
        class_embed_dim=config.model.class_embed_dim,
    ).to(device)
    generator.load_state_dict(payload["generator_state"])
    generator.eval()

    torch.manual_seed(seed)
    noise = torch.randn(pairs, config.model.latent_dim, device=device)
    labels_0 = torch.zeros(pairs, dtype=torch.long, device=device)
    labels_1 = torch.ones(pairs, dtype=torch.long, device=device)

    with torch.no_grad():
        images_0 = generator(noise, labels_0).cpu()
        images_1 = generator(noise, labels_1).cpu()

    row_0 = add_title(tensor_to_pil_grid(images_0, nrow=8), "Same noise z, label Male=0")
    row_1 = add_title(tensor_to_pil_grid(images_1, nrow=8), "Same noise z, label Male=1")
    gap = 8
    width = max(row_0.width, row_1.width)
    height = row_0.height + row_1.height + gap
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(row_0, (0, 0))
    canvas.paste(row_1, (0, row_0.height + gap))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = build_argparser().parse_args()
    generate_pairs(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        pairs=args.pairs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
