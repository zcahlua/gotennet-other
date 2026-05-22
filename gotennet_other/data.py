from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.utils.data import Dataset


@dataclass
class MoleculeBatch:
    z: torch.Tensor
    pos: torch.Tensor
    energy: torch.Tensor
    force: torch.Tensor
    batch: torch.Tensor


class Transition1XLoader(Dataset):
    """Adapter over OpenQDC Transition1X with cache checks and normalized samples."""

    def __init__(self, split: str = "train", cache_dir: Optional[str] = None, dataset: Any = None):
        self.split = split
        self.cache_dir = cache_dir
        self.dataset = dataset or self._load_real_dataset(split=split, cache_dir=cache_dir)

    @staticmethod
    def _load_real_dataset(split: str, cache_dir: Optional[str]):
        try:
            from openqdc.datasets import Transition1X
        except Exception as exc:  # import/package error
            raise RuntimeError(
                "openqdc is required to use Transition1X. Install with `pip install openqdc`."
            ) from exc

        root = Path(cache_dir).expanduser() if cache_dir else Path.home() / ".cache" / "openqdc"
        if not root.exists():
            raise FileNotFoundError(
                f"Transition1X cache directory not found at '{root}'. "
                "Create it and pre-download OpenQDC data before running real training."
            )

        try:
            return Transition1X(split=split, root=str(root), download=False)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load Transition1X from local cache. "
                "Ensure the dataset exists locally and rerun."
            ) from exc

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        return {
            "z": torch.as_tensor(item["z"], dtype=torch.long),
            "pos": torch.as_tensor(item["pos"], dtype=torch.float32),
            "energy": torch.as_tensor(item["energy"], dtype=torch.float32).reshape(1),
            "force": torch.as_tensor(item["force"], dtype=torch.float32),
        }


def collate_molecules(samples: Iterable[Dict[str, torch.Tensor]]) -> MoleculeBatch:
    samples = list(samples)
    z = torch.cat([s["z"] for s in samples], dim=0)
    pos = torch.cat([s["pos"] for s in samples], dim=0)
    energy = torch.cat([s["energy"] for s in samples], dim=0)
    force = torch.cat([s["force"] for s in samples], dim=0)
    batch = torch.cat([
        torch.full((s["z"].shape[0],), i, dtype=torch.long) for i, s in enumerate(samples)
    ])
    return MoleculeBatch(z=z, pos=pos, energy=energy, force=force, batch=batch)
