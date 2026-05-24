from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from torch.utils.data import Dataset


@dataclass
class MoleculeBatch:
    z: torch.Tensor
    pos: torch.Tensor
    energy: torch.Tensor
    force: torch.Tensor
    batch: torch.Tensor
    charge: torch.Tensor | None = None


class OpenQDCLoader(Dataset):
    """Adapter over OpenQDC datasets with cache checks and normalized samples."""

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        cache_dir: Optional[str] = None,
        dataset: Any = None,
        max_samples: int | None = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.cache_dir = cache_dir
        self.max_samples = max_samples
        self.dataset = dataset or self._load_real_dataset(dataset_name=dataset_name, split=split, cache_dir=cache_dir)

    @staticmethod
    def _load_real_dataset(dataset_name: str, split: str, cache_dir: Optional[str]):
        root = Path(cache_dir).expanduser() if cache_dir else Path.home() / ".cache" / "openqdc"
        if not root.exists():
            raise FileNotFoundError(
                f"{dataset_name} cache directory not found at '{root}'. "
                "Create it and pre-download OpenQDC data before running real training."
            )

        try:
            from openqdc import datasets as oqdc_datasets
        except Exception as exc:
            raise RuntimeError(
                f"openqdc is required to use {dataset_name}. Install with `pip install -e \".[openqdc]\"`."
            ) from exc

        if not hasattr(oqdc_datasets, dataset_name):
            raise RuntimeError(f"OpenQDC dataset '{dataset_name}' is not available in installed openqdc version.")
        dataset_cls = getattr(oqdc_datasets, dataset_name)

        kwargs = dict(
            cache_dir=str(root),
            array_format="torch",
            energy_unit="ev",
            distance_unit="ang",
            energy_type="formation",
        )
        if "skip_statistics" in inspect.signature(dataset_cls).parameters:
            kwargs["skip_statistics"] = True

        try:
            return dataset_cls(**kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load {dataset_name} from local cache '{root}' for split '{split}'. "
                "Ensure the dataset exists locally and rerun."
            ) from exc

    def __len__(self) -> int:
        size = len(self.dataset)
        return min(size, self.max_samples) if self.max_samples is not None else size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        get = (lambda k, d=None: item.get(k, d)) if isinstance(item, dict) else (lambda k, d=None: getattr(item, k, d))

        z = get("atomic_numbers", get("z"))
        pos = get("positions", get("pos"))
        charge = get("charges", get("charge"))
        energy = get("formation_energies", get("energy"))
        force = get("forces", get("force"))

        energy_t = torch.as_tensor(energy, dtype=torch.float32)
        if energy_t.ndim > 0:
            energy_t = energy_t.reshape(-1)[0:1]
        else:
            energy_t = energy_t.reshape(1)

        force_t = torch.as_tensor(force, dtype=torch.float32)
        if force_t.ndim == 3 and force_t.shape[-1] == 1:
            force_t = force_t[..., 0]

        sample = {
            "z": torch.as_tensor(z, dtype=torch.long),
            "pos": torch.as_tensor(pos, dtype=torch.float32),
            "energy": energy_t,
            "force": force_t,
        }
        if charge is not None:
            sample["charge"] = torch.as_tensor(charge, dtype=torch.float32)
        for key in ("name", "subset"):
            value = get(key, None)
            if value is not None:
                sample[key] = value
        return sample


class Transition1XLoader(OpenQDCLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(dataset_name="Transition1X", *args, **kwargs)


class SN2RXNLoader(OpenQDCLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(dataset_name="SN2RXN", *args, **kwargs)


def collate_molecules(samples: Iterable[Dict[str, torch.Tensor]]) -> MoleculeBatch:
    samples = list(samples)
    z = torch.cat([s["z"] for s in samples], dim=0)
    pos = torch.cat([s["pos"] for s in samples], dim=0)
    energy = torch.cat([s["energy"] for s in samples], dim=0)
    force = torch.cat([s["force"] for s in samples], dim=0)
    batch = torch.cat([
        torch.full((s["z"].shape[0],), i, dtype=torch.long) for i, s in enumerate(samples)
    ])
    charge = None
    if all("charge" in s for s in samples):
        charge = torch.cat([s["charge"] for s in samples], dim=0)
    return MoleculeBatch(z=z, pos=pos, energy=energy, force=force, batch=batch, charge=charge)
