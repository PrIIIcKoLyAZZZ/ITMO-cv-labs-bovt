from __future__ import annotations

import torch
from torch import nn


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        base_channels: int = 64,
        image_channels: int = 3,
        conditional: bool = False,
        num_classes: int = 2,
        class_embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.conditional = conditional
        self.latent_dim = latent_dim
        input_dim = latent_dim

        if conditional:
            self.label_embedding = nn.Embedding(num_classes, class_embed_dim)
            input_dim += class_embed_dim
        else:
            self.label_embedding = None

        self.net = nn.Sequential(
            nn.ConvTranspose2d(input_dim, base_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels, image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if self.conditional:
            if labels is None:
                raise ValueError("Conditional generator requires class labels.")
            embedded = self.label_embedding(labels)
            noise = torch.cat([noise, embedded], dim=1)

        return self.net(noise.unsqueeze(-1).unsqueeze(-1))


class Discriminator(nn.Module):
    def __init__(
        self,
        base_channels: int = 64,
        image_channels: int = 3,
        conditional: bool = False,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.conditional = conditional
        input_channels = image_channels + (num_classes if conditional else 0)
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if self.conditional:
            if labels is None:
                raise ValueError("Conditional discriminator requires class labels.")

            condition = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).float()
            condition = condition[:, :, None, None].expand(-1, -1, images.size(2), images.size(3))
            images = torch.cat([images, condition], dim=1)

        logits = self.net(images)
        return logits.view(-1)


class Critic(nn.Module):
    def __init__(
        self,
        base_channels: int = 64,
        image_channels: int = 3,
        conditional: bool = False,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.conditional = conditional
        self.num_classes = num_classes
        input_channels = image_channels + (num_classes if conditional else 0)

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, 1, 4, 1, 0, bias=True),
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if self.conditional:
            if labels is None:
                raise ValueError("Conditional critic requires class labels.")
            condition = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).float()
            condition = condition[:, :, None, None].expand(-1, -1, images.size(2), images.size(3))
            images = torch.cat([images, condition], dim=1)

        return self.net(images).view(-1)
