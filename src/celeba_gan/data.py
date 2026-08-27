from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transforms.Compose(ops)


@dataclass
class SampleRecord:
    image_path: Path
    label: int


class CelebAFacesDataset(Dataset):
    def __init__(
        self,
        processed_root: str | Path,
        metadata_path: str | Path,
        split: str,
        image_size: int,
        condition_attr: str = "Male",
        conditional: bool = False,
    ) -> None:
        self.processed_root = Path(processed_root)
        self.metadata_path = Path(metadata_path)
        self.split = split
        self.conditional = conditional
        self.condition_attr = condition_attr
        self.transform = build_transforms(image_size=image_size, train=split == "train")

        frame = pd.read_csv(self.metadata_path)
        if condition_attr not in frame.columns:
            raise ValueError(f"Attribute '{condition_attr}' is not present in metadata.")

        frame = frame[frame["split"] == split].reset_index(drop=True)
        self.records = [
            SampleRecord(
                image_path=self.processed_root / row["processed_rel_path"],
                label=int(row[condition_attr]),
            )
            for _, row in frame.iterrows()
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        image_tensor = self.transform(image)

        if self.conditional:
            return image_tensor, torch.tensor(record.label, dtype=torch.long)
        return image_tensor
