from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset, Subset


@dataclass
class OpenQDCDatasetConfig:
    dataset_name: str
    cache_dir: str | None = None
    energy_target: str | None = None
    max_samples: int | None = None
    split: str = "train"
    split_by_name: bool = True
    seed: int = 42
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    read_as_zarr: bool = False


@dataclass
class MoleculeBatch:
    z: torch.Tensor
    pos: torch.Tensor
    energy: torch.Tensor
    batch: torch.Tensor
    force: torch.Tensor | None = None
    charge: torch.Tensor | None = None
    names: list[str] | None = None
    subsets: list[object] | None = None
    n_atoms: torch.Tensor | None = None


def normalize_dataset_name(name: str) -> str:
    mapping = {
        "transition1x": "transition1x", "Transition1X": "transition1x", "Transition1x": "transition1x",
        "sn2rxn": "sn2rxn", "SN2RXN": "sn2rxn", "sn2_rxn": "sn2rxn", "sn2-rxn": "sn2rxn",
        "synthetic": "synthetic", "synthetic_sn2rxn": "synthetic_sn2rxn",
    }
    if name not in mapping:
        raise ValueError(f"Unknown dataset name: {name}")
    return mapping[name]


def get_openqdc_dataset_class(dataset_name: str):
    normalized = normalize_dataset_name(dataset_name)
    if normalized == "transition1x":
        from openqdc.datasets import Transition1X
        return Transition1X
    if normalized == "sn2rxn":
        from openqdc.datasets import SN2RXN
        return SN2RXN
    raise ValueError(f"Unsupported OpenQDC dataset: {dataset_name}")


class OpenQDCLoader(Dataset):
    def __init__(self, config: OpenQDCDatasetConfig | None = None, *, dataset_name: str | None = None, split: str = "train", cache_dir: str | None = None, dataset: object | None = None, max_samples: int | None = None, energy_target: str | None = None, split_by_name: bool = True, seed: int = 42, train_fraction: float = 0.8, val_fraction: float = 0.1, test_fraction: float = 0.1, read_as_zarr: bool = False):
        self.config = config or OpenQDCDatasetConfig(dataset_name=dataset_name or "transition1x", split=split, cache_dir=cache_dir, max_samples=max_samples, energy_target=energy_target, split_by_name=split_by_name, seed=seed, train_fraction=train_fraction, val_fraction=val_fraction, test_fraction=test_fraction, read_as_zarr=read_as_zarr)
        self.dataset_name = normalize_dataset_name(self.config.dataset_name)
        if self.config.energy_target is None:
            self.config.energy_target = "energies" if self.dataset_name == "sn2rxn" else "formation_energies"
        self.dataset = dataset if dataset is not None else self._load_real_dataset()
        self.indices = self._build_split_indices()
        if self.config.max_samples is not None:
            self.indices = self.indices[: self.config.max_samples]

    def _load_real_dataset(self):
        cache_path = Path(self.config.cache_dir or "data/openqdc")
        if not cache_path.exists():
            if self.dataset_name == "transition1x":
                raise FileNotFoundError(f"Transition1X cache directory not found at '{cache_path}'.\n")
            raise FileNotFoundError(f"SN2RXN cache directory not found at '{cache_path}'.\n")
        try:
            dataset_cls = get_openqdc_dataset_class(self.dataset_name)
        except ImportError as exc:
            ds = "Transition1X" if self.dataset_name == "transition1x" else "SN2RXN"
            raise ImportError(f'openqdc is required to use {ds}. Install with:\n  pip install -e ".[openqdc]"') from exc
        kwargs = dict(cache_dir=str(cache_path), array_format="torch", energy_unit="ev", distance_unit="ang", energy_type="formation")
        sig = inspect.signature(dataset_cls)
        if "skip_statistics" in sig.parameters:
            kwargs["skip_statistics"] = True
        if "read_as_zarr" in sig.parameters:
            kwargs["read_as_zarr"] = self.config.read_as_zarr
        return dataset_cls(**kwargs)

    def _build_split_indices(self) -> list[int]:
        split = self.config.split
        if split == "all":
            return list(range(len(self.dataset)))
        n = len(self.dataset)
        if hasattr(self.dataset, "data") and isinstance(self.dataset.data, dict):
            data = self.dataset.data
            if self.config.split_by_name and data.get("name") is not None:
                names = [str(x) for x in data["name"]]
                uniq = sorted(set(names))
                g = torch.Generator().manual_seed(self.config.seed)
                perm = torch.randperm(len(uniq), generator=g).tolist()
                cut1, cut2 = int(len(uniq)*self.config.train_fraction), int(len(uniq)*(self.config.train_fraction+self.config.val_fraction))
                group = {"train":[uniq[i] for i in perm[:cut1]], "val":[uniq[i] for i in perm[cut1:cut2]], "test":[uniq[i] for i in perm[cut2:]]}[split]
                return [i for i, nm in enumerate(names) if nm in set(group)]
            if data.get("subset") is not None:
                allowed = {"train":{"train"}, "val":{"val","valid","validation"}, "test":{"test"}}[split]
                idx = [i for i, s in enumerate(data["subset"]) if str(s).lower() in allowed]
                if idx:
                    return idx
        g = torch.Generator().manual_seed(self.config.seed)
        perm = torch.randperm(n, generator=g).tolist()
        a, b = int(n*self.config.train_fraction), int(n*(self.config.train_fraction+self.config.val_fraction))
        return {"train": perm[:a], "val": perm[a:b], "test": perm[b:]}[split]

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx: int):
        item = self.dataset[self.indices[idx]]
        get = (lambda k, d=None: item.get(k, d)) if isinstance(item, dict) else (lambda k, d=None: getattr(item, k, d))
        z, pos, charge = get("atomic_numbers", get("z")), get("positions", get("pos")), get("charges", get("charge"))
        if z is None or pos is None:
            raise ValueError(f"{self.dataset_name} sample missing z or pos")
        et = self.config.energy_target
        order = ["energy", "formation_energies", "energies"] if et == "energy" else (["energies", "formation_energies"] if et == "energies" else ["formation_energies", "energies"])
        energy = None
        for k in order:
            energy = get(k)
            if energy is not None:
                break
        if energy is None:
            raise ValueError(f"{self.dataset_name} sample missing energy")
        energy_t = torch.as_tensor(energy, dtype=torch.float32).reshape(-1)[:1]
        force = get("forces", get("force"))
        force_t = None if force is None else torch.as_tensor(force, dtype=torch.float32)
        if force_t is not None and force_t.ndim == 3:
            force_t = force_t[:, :, 0]
        sample = {"z": torch.as_tensor(z, dtype=torch.long), "pos": torch.as_tensor(pos, dtype=torch.float32), "energy": energy_t, "force": force_t, "charge": None if charge is None else torch.as_tensor(charge, dtype=torch.float32), "n_atoms": int(torch.as_tensor(z).numel()), "name": get("name"), "subset": get("subset")}
        return sample


class Transition1XLoader(OpenQDCLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dataset_name="transition1x", **kwargs)


class SN2RXNLoader(OpenQDCLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dataset_name="sn2rxn", **kwargs)


def split_dataset(dataset: Dataset, split: str, seed: int = 0, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dataset:
    n = len(dataset); g = torch.Generator().manual_seed(seed); perm = torch.randperm(n, generator=g).tolist(); a, b = int(n*train_ratio), int(n*(train_ratio+val_ratio))
    idx = {"train": perm[:a], "val": perm[a:b], "test": perm[b:]}[split]
    return Subset(dataset, idx)


def collate_molecules(samples: Iterable[dict[str, Any]]) -> MoleculeBatch:
    s = list(samples)
    z = torch.cat([x["z"] for x in s]); pos = torch.cat([x["pos"] for x in s]); energy = torch.cat([x["energy"].reshape(1) for x in s]).reshape(-1)
    batch = torch.cat([torch.full((x["z"].numel(),), i, dtype=torch.long) for i, x in enumerate(s)])
    n_atoms = torch.tensor([x["z"].numel() for x in s], dtype=torch.long)
    force = None if any(x.get("force") is None for x in s) else torch.cat([x["force"] for x in s])
    charge = None if any(x.get("charge") is None for x in s) else torch.cat([x["charge"] for x in s])
    names = [x.get("name") for x in s]; subsets = [x.get("subset") for x in s]
    return MoleculeBatch(z=z, pos=pos, energy=energy, batch=batch, force=force, charge=charge, names=names, subsets=subsets, n_atoms=n_atoms)
