from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_training_curves(history_path: str | Path, output_path: str | Path) -> None:
    history = pd.read_csv(history_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["global_step"], history["d_loss"], label="D loss")
    axes[0].plot(history["global_step"], history["g_loss"], label="G loss")
    axes[0].set_title("Generator / Discriminator Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(history["global_step"], history["d_real_mean"], label="D(real)")
    axes[1].plot(history["global_step"], history["d_fake_mean"], label="D(fake)")
    axes[1].set_title("Discriminator Scores")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Logit Mean")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
