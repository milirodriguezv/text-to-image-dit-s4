import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class CocoLatentDataset(Dataset):
    """
    Dataset for text-conditioned DiT training.

    Each item returns:
        z_0: sampled VAE latent, shape (4, 32, 32), float32.
        e_pooled: CLIP text embedding, shape (768,), float32.
    """

    def __init__(self, feature_dir="features/train"):
        feature_dir = Path(feature_dir)

        self.latents = np.load(
            feature_dir / "latents.npy",
            mmap_mode="r"
        )
        self.e_pooled = np.load(
            feature_dir / "e_pooled.npy",
            mmap_mode="r"
        )
        self.image_ids = np.load(
            feature_dir / "image_ids.npy",
            mmap_mode="r"
        )

        if not (
            len(self.latents)
            == len(self.e_pooled)
            == len(self.image_ids)
        ):
            raise ValueError(
                "Latents, text embeddings and image IDs "
                "must have the same number of entries."
            )

        if self.latents.shape[1:] != (2, 8, 32, 32):
            raise ValueError(
                f"Unexpected latent shape: {self.latents.shape}"
            )

        if self.e_pooled.shape[1:] != (5, 768):
            raise ValueError(
                f"Unexpected text embedding shape: "
                f"{self.e_pooled.shape}"
            )

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        # Randomly choose the original or horizontally flipped view.
        flip_idx = torch.randint(0, 2, (1,)).item()

        # Randomly choose one of the five captions.
        caption_idx = torch.randint(0, 5, (1,)).item()

        # Copy the selected arrays to avoid issues with read-only
        # memory-mapped NumPy arrays.
        stats = torch.from_numpy(
            np.array(self.latents[idx, flip_idx], copy=True)
        ).float()

        e_pooled = torch.from_numpy(
            np.array(
                self.e_pooled[idx, caption_idx],
                copy=True
            )
        ).float()

        # The first four channels contain the posterior mean.
        # The last four contain the posterior standard deviation.
        mean = stats[:4]
        std = stats[4:]

        # Sample a latent from the VAE posterior.
        # The saved statistics already include the 0.18215 scale.
        z_0 = mean + std * torch.randn_like(std)
        z_0 = z_0 * 0.18215

        return z_0, e_pooled



if __name__ == "__main__":
    dataset = CocoLatentDataset(feature_dir="features/train")

    print(f"Total de imágenes: {len(dataset)}")
    print(f"IDs de imágenes: {dataset.image_ids[:]}")

    # Probar varias muestras usando el Dataset real
    for i in range(5):
        idx = i % len(dataset)
        z_0, e_pooled = dataset[idx]

        assert z_0.shape == (4, 32, 32), (
            f"Forma incorrecta del latente: {z_0.shape}"
        )
        assert e_pooled.shape == (768,), (
            f"Forma incorrecta del embedding: {e_pooled.shape}"
        )
        assert z_0.dtype == torch.float32
        assert e_pooled.dtype == torch.float32
        assert torch.isfinite(z_0).all()
        assert torch.isfinite(e_pooled).all()

        print(
            f"Muestra {i + 1}: "
            f"z_0={tuple(z_0.shape)}, "
            f"embedding={tuple(e_pooled.shape)}, "
            f"valores finitos=True"
        )

    print("\nDataset comprobado correctamente.")

