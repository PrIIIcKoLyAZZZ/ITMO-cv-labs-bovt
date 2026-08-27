from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid, save_image


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_init(module: torch.nn.Module) -> None:
    classname = module.__class__.__name__
    if "Conv" in classname:
        torch.nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            torch.nn.init.zeros_(module.bias.data)
    elif "BatchNorm" in classname:
        torch.nn.init.normal_(module.weight.data, 1.0, 0.02)
        torch.nn.init.zeros_(module.bias.data)


def denormalize(images: torch.Tensor) -> torch.Tensor:
    return images.mul(0.5).add(0.5).clamp(0.0, 1.0)


def save_image_grid(images: torch.Tensor, path: str | Path, nrow: int = 8) -> None:
    grid = make_grid(denormalize(images.detach().cpu()), nrow=nrow)
    save_image(grid, path)


def write_json(data: dict, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def load_pil_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")
