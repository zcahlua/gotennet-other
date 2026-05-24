from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from torch.utils.data import Dataset, Subset


@dataclass
class MoleculeBatch:
    z: torch.Tensor
    pos: torch.Tensor
    energy: torch.Tensor
    force: torch.Tensor | None
    batch: torch.Tensor
    charge: torch.Tensor | None = None


class OpenQDCLoader(Dataset):
    """Adapter over OpenQDC datasets with split handling and normalized samples."""

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        cache_dir: Optional[str] = None,
        dataset: Any = None,
        max_samples: int | None = None,
        energy_target: str | None = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.cache_dir = cache_dir
        self.max_samples = max_samples
        self.energy_target = energy_target or ("energies" if dataset_name == "SN2RXN" else "formation_energies")
        self.dataset = dataset or self._load_real_dataset(dataset_name=dataset_name, split=split, cache_dir=cache_dir)
        self.indices = _resolve_split_indices(self.dataset, split=split)

    @staticmethod
    def _load_real_dataset(dataset_name: str, split: str, cache_dir: Optional[str]):
        root = Path(cache_dir).expanduser() if cache_dir else Path.home() / ".cache" / "openqdc"
        if not root.exists():
            raise FileNotFoundError(f"{dataset_name} cache directory not found at '{root}'.")
        from openqdc import datasets as oqdc_datasets
        if not hasattr(oqdc_datasets, dataset_name):
            raise RuntimeError(f"OpenQDC dataset '{dataset_name}' is not available.")
        dataset_cls = getattr(oqdc_datasets, dataset_name)
        kwargs = dict(cache_dir=str(root), array_format="torch", energy_unit="ev", distance_unit="ang", energy_type="formation")
        if "skip_statistics" in inspect.signature(dataset_cls).parameters:
            kwargs["skip_statistics"] = True
        return dataset_cls(**kwargs)

    def __len__(self) -> int:
        size = len(self.indices)
        return min(size, self.max_samples) if self.max_samples is not None else size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        mapped_idx = self.indices[idx]
        item = self.dataset[mapped_idx]
        get = (lambda k, d=None: item.get(k, d)) if isinstance(item, dict) else (lambda k, d=None: getattr(item, k, d))

        z = get("atomic_numbers", get("z"))
        pos = get("positions", get("pos"))
        charge = get("charges", get("charge"))
        energy = get(self.energy_target, get("formation_energies", get("energies", get("energy"))))
        force = get("forces", get("force"))

        energy_t = torch.as_tensor(energy, dtype=torch.float32).reshape(-1)[:1]

        force_t = None
        if force is not None:
            force_t = torch.as_tensor(force, dtype=torch.float32)
            if force_t.ndim == 1:
                force_t = force_t.reshape(-1, 3)
            elif force_t.ndim == 3:
                force_t = force_t[:, :, 0]
            if force_t.ndim == 2 and force_t.shape[0] == 3 and force_t.shape[1] != 3:
                force_t = force_t.transpose(0, 1)

        sample: Dict[str, Any] = {
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


def _resolve_split_indices(dataset: Any, split: str) -> list[int]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split '{split}'.")
    n = len(dataset)
    if not hasattr(dataset, "data"):
        return list(range(n))
    data = getattr(dataset, "data")
    names = data.get("name") if isinstance(data, dict) else None
    subsets = data.get("subset") if isinstance(data, dict) else None
    if names is None and subsets is None:
        return list(range(n))
    result = []
    for i in range(n):
        subset_v = subsets[i] if subsets is not None else None
        name_v = names[i] if names is not None else ""
        tag = str(subset_v if subset_v is not None else name_v).lower()
        if split == "val" and tag in {"val", "valid", "validation"}:
            result.append(i)
        elif split == "test" and tag == "test":
            result.append(i)
        elif split == "train" and tag in {"train", "training"}:
            result.append(i)
    return result or list(range(n))


class Transition1XLoader(OpenQDCLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(dataset_name="Transition1X", *args, **kwargs)


class SN2RXNLoader(OpenQDCLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(dataset_name="SN2RXN", *args, **kwargs)


def split_dataset(dataset: Dataset, split: str, seed: int = 0, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dataset:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split '{split}'. Expected one of: train, val, test.")
    n = len(dataset)
    if n == 0:
        return Subset(dataset, [])
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    idx = {"train": perm[:n_train], "val": perm[n_train:n_train+n_val], "test": perm[n_train+n_val:]}[split]
    return Subset(dataset, idx)


def collate_molecules(samples: Iterable[Dict[str, torch.Tensor]]) -> MoleculeBatch:
    samples = list(samples)
    z = torch.cat([s["z"] for s in samples], dim=0)
    pos = torch.cat([s["pos"] for s in samples], dim=0)
    energy = torch.cat([s["energy"] for s in samples], dim=0)
    has_force = all(s.get("force") is not None for s in samples)
    force = torch.cat([s["force"] for s in samples], dim=0) if has_force else None
    batch = torch.cat([torch.full((s["z"].shape[0],), i, dtype=torch.long) for i, s in enumerate(samples)])
    charge = torch.cat([s["charge"] for s in samples], dim=0) if all("charge" in s for s in samples) else None
    return MoleculeBatch(z=z, pos=pos, energy=energy, force=force, batch=batch, charge=charge)
