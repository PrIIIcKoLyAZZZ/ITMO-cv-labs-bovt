from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    processed_root: str
    metadata_path: str
    image_size: int = 64
    batch_size: int = 128
    num_workers: int = 4
    condition_attr: str = "Male"


@dataclass
class ModelConfig:
    latent_dim: int = 128
    base_channels: int = 64
    image_channels: int = 3
    conditional: bool = False
    num_classes: int = 2
    class_embed_dim: int = 32


@dataclass
class TrainConfig:
    run_name: str = "dcgan"
    output_root: str = "runs"
    epochs: int = 50
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    seed: int = 42
    sample_every: int = 1
    save_every: int = 5
    log_every: int = 100
    fixed_sample_count: int = 64
    amp: bool = False
    init_checkpoint: str | None = None


@dataclass
class EvalConfig:
    split: str = "test"
    num_samples: int = 10000
    batch_size: int = 128
    inception_splits: int = 10


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    eval: EvalConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        return cls(
            data=DataConfig(**raw["data"]),
            model=ModelConfig(**raw["model"]),
            train=TrainConfig(**raw["train"]),
            eval=EvalConfig(**raw["eval"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
