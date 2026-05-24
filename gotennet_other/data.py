from __future__ import annotations

import inspect
import math
import re
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
    force: torch.Tensor | None
    batch: torch.Tensor
    n_atoms: torch.Tensor
    has_force: torch.Tensor
    charge: torch.Tensor | None = None


def normalize_dataset_name(name: str) -> str:
    token = re.sub(r"[^a-z0-9]", "", name.lower())
    aliases = {
        "transition1x": "transition1x",
        "sn2rxn": "sn2rxn",
    }
    if token in aliases:
        return aliases[token]
    raise ValueError(f"Unsupported dataset_name '{name}'.")


class OpenQDCLoader(Dataset):
    def __init__(self, dataset_name: str, split: str = "train", cache_dir: Optional[str] = None, dataset: Any = None,
                 max_samples: int | None = None, energy_target: str | None = None, split_by_name: bool = True,
                 read_as_zarr: bool = False, train_fraction: float = 0.8, val_fraction: float = 0.1, test_fraction: float = 0.1, seed: int = 42):
        self.dataset_name = normalize_dataset_name(dataset_name)
        self.split = split
        self.max_samples = max_samples
        self.energy_target = energy_target
        self.split_by_name = split_by_name
        self.train_fraction, self.val_fraction, self.test_fraction = train_fraction, val_fraction, test_fraction
        self.seed = seed
        self.dataset = dataset or self._load_real_dataset(cache_dir=cache_dir, read_as_zarr=read_as_zarr)
        self.indices = self._compute_split_indices()
        if self.max_samples is not None:
            self.indices = self.indices[: self.max_samples]

    def _load_real_dataset(self, cache_dir: Optional[str], read_as_zarr: bool):
        root = Path(cache_dir).expanduser() if cache_dir else Path.home() / ".cache" / "openqdc"
        if not root.exists():
            raise FileNotFoundError(f"OpenQDC cache directory not found at '{root}'.")
        try:
            from openqdc.datasets import SN2RXN, Transition1X
        except Exception as exc:
            raise RuntimeError("openqdc is required. Install with `pip install -e \".[openqdc]\"`.") from exc
        cls = Transition1X if self.dataset_name == "transition1x" else SN2RXN
        kwargs = dict(cache_dir=str(root), array_format="torch", energy_unit="ev", distance_unit="ang", energy_type="formation")
        sig = inspect.signature(cls)
        if "skip_statistics" in sig.parameters:
            kwargs["skip_statistics"] = True
        if "read_as_zarr" in sig.parameters:
            kwargs["read_as_zarr"] = read_as_zarr
        return cls(**kwargs)

    def _get_field(self, idx: int, key: str):
        data = getattr(self.dataset, "data", None)
        if isinstance(data, dict) and key in data:
            return data[key][idx]
        item = self.dataset[idx]
        return item.get(key) if isinstance(item, dict) else getattr(item, key, None)

    def _compute_split_indices(self):
        n = len(self.dataset)
        all_idx = list(range(n))
        if self.split == "all":
            return all_idx
        if self.split_by_name:
            names = [self._get_field(i, "name") for i in all_idx]
            if any(v is not None for v in names):
                groups = {}
                for i, name in enumerate(names):
                    groups.setdefault(str(name), []).append(i)
                keys = sorted(groups)
                g = torch.Generator().manual_seed(self.seed)
                perm = torch.randperm(len(keys), generator=g).tolist()
                ordered = [keys[i] for i in perm]
                ntr = int(math.floor(len(keys) * self.train_fraction))
                nva = int(math.floor(len(keys) * self.val_fraction))
                split_keys = {
                    "train": set(ordered[:ntr]),
                    "val": set(ordered[ntr:ntr + nva]),
                    "test": set(ordered[ntr + nva:]),
                }
                return [i for i, name in enumerate(names) if str(name) in split_keys[self.split]]
        subsets = [self._get_field(i, "subset") for i in all_idx]
        if any(v is not None for v in subsets):
            wanted = {"val": {"val", "valid", "validation"}, "train": {"train"}, "test": {"test"}}[self.split]
            idxs = [i for i, s in enumerate(subsets) if str(s).lower() in wanted]
            if idxs:
                return idxs
        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n, generator=g).tolist()
        ntr = int(math.floor(n * self.train_fraction))
        nva = int(math.floor(n * self.val_fraction))
        if self.split == "train":
            return perm[:ntr]
        if self.split == "val":
            return perm[ntr:ntr + nva]
        return perm[ntr + nva:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, local_idx: int):
        idx = self.indices[local_idx]
        item = self.dataset[idx]
        get = (lambda k, d=None: item.get(k, d)) if isinstance(item, dict) else (lambda k, d=None: getattr(item, k, d))
        z = get("atomic_numbers", get("z"))
        pos = get("positions", get("pos"))
        charge = get("charges", get("charge"))
        name, subset = get("name", None), get("subset", None)
        default_target = "formation_energies" if self.dataset_name == "transition1x" else "energies"
        target = self.energy_target or default_target
        fallback = "energies" if target == "formation_energies" else "formation_energies"
        energy = get(target, get(fallback, get("energy")))
        force = get("forces", get("force"))
        et = torch.as_tensor(energy, dtype=torch.float32)
        if et.ndim > 1:
            et = et.reshape(-1)[0:1]
        else:
            et = et.reshape(-1)[:1] if et.ndim > 0 else et.reshape(1)
        ft = None
        if force is not None:
            ft = torch.as_tensor(force, dtype=torch.float32)
            if ft.ndim == 3:
                ft = ft[..., 0]
        sample = {"z": torch.as_tensor(z, dtype=torch.long), "pos": torch.as_tensor(pos, dtype=torch.float32), "energy": et, "force": ft,
                  "charge": None if charge is None else torch.as_tensor(charge, dtype=torch.float32), "n_atoms": int(len(z)), "name": name, "subset": subset}
        return sample


def collate_molecules(samples: Iterable[Dict[str, torch.Tensor]]) -> MoleculeBatch:
    samples = list(samples)
    z = torch.cat([s["z"] for s in samples], dim=0)
    pos = torch.cat([s["pos"] for s in samples], dim=0)
    energy = torch.cat([s["energy"] for s in samples], dim=0)
    batch = torch.cat([torch.full((s["z"].shape[0],), i, dtype=torch.long) for i, s in enumerate(samples)])
    n_atoms = torch.tensor([s["n_atoms"] for s in samples], dtype=torch.float32)
    has_force = torch.tensor([s["force"] is not None for s in samples], dtype=torch.bool)
    force = None
    if has_force.any():
        chunks = [s["force"] if s["force"] is not None else torch.zeros(s["n_atoms"], 3) for s in samples]
        force = torch.cat(chunks, dim=0)
    charge = None
    if all(s.get("charge") is not None for s in samples):
        charge = torch.cat([s["charge"] for s in samples], dim=0)
    return MoleculeBatch(z=z, pos=pos, energy=energy, force=force, batch=batch, n_atoms=n_atoms, has_force=has_force, charge=charge)
