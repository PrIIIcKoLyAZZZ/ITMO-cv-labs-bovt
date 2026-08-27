from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
from torch import nn
from torchvision.models import Inception_V3_Weights, inception_v3


@dataclass
class MetricResult:
    fid: float
    inception_score_mean: float
    inception_score_std: float


class InceptionFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        self.model.eval()

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
        if images.min() < 0:
            images = (images + 1.0) / 2.0
        images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        return (images - self.mean) / self.std

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.preprocess(images)

        x = self.model.Conv2d_1a_3x3(x)
        x = self.model.Conv2d_2a_3x3(x)
        x = self.model.Conv2d_2b_3x3(x)
        x = self.model.maxpool1(x)
        x = self.model.Conv2d_3b_1x1(x)
        x = self.model.Conv2d_4a_3x3(x)
        x = self.model.maxpool2(x)
        x = self.model.Mixed_5b(x)
        x = self.model.Mixed_5c(x)
        x = self.model.Mixed_5d(x)
        x = self.model.Mixed_6a(x)
        x = self.model.Mixed_6b(x)
        x = self.model.Mixed_6c(x)
        x = self.model.Mixed_6d(x)
        x = self.model.Mixed_6e(x)
        x = self.model.Mixed_7a(x)
        x = self.model.Mixed_7b(x)
        x = self.model.Mixed_7c(x)
        x = self.model.avgpool(x)
        x = self.model.dropout(x)
        features = torch.flatten(x, 1)
        logits = self.model.fc(features)
        return features, torch.softmax(logits, dim=1)


def compute_activation_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def compute_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    mu_real, sigma_real = compute_activation_stats(real_features)
    mu_fake, sigma_fake = compute_activation_stats(fake_features)

    diff = mu_real - mu_fake
    cov_prod = sigma_real @ sigma_fake
    cov_mean = sqrtm(cov_prod)
    if not np.isfinite(cov_mean).all():
        eps = np.eye(sigma_real.shape[0]) * 1e-6
        cov_mean = sqrtm((sigma_real + eps) @ (sigma_fake + eps))
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * cov_mean)
    return float(fid)


def compute_inception_score(probabilities: np.ndarray, splits: int = 10) -> tuple[float, float]:
    chunks = np.array_split(probabilities, splits)
    scores = []
    for chunk in chunks:
        py = np.mean(chunk, axis=0, keepdims=True)
        kl = chunk * (np.log(chunk + 1e-8) - np.log(py + 1e-8))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
    return float(np.mean(scores)), float(np.std(scores))
