from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset

from .data import OpenQDCLoader, collate_molecules
from .metrics import compute_metrics
from .model import EnergyModel


@dataclass
class TrainerConfig:
    batch_size: int = 8
    epochs: int = 3
    lr: float = 1e-3
    force_weight: float = 10.0
    device: str = "cpu"
    max_samples: int | None = None


class SyntheticTransition1XDataset(Dataset):
    def __init__(self, size: int = 32):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        n = 2 + (idx % 3)
        z = torch.randint(1, 10, (n,))
        pos = torch.randn(n, 3)
        energy = pos.pow(2).sum().reshape(1)
        force = -2 * pos
        return {"z": z, "pos": pos, "energy": energy, "force": force}


def run_epoch(model: EnergyModel, loader: DataLoader, optimizer=None, force_weight: float = 10.0, device: str = "cpu") -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    all_pred_e, all_true_e, all_pred_f, all_true_f = [], [], [], []
    for batch in loader:
        z = batch.z.to(device)
        pos = batch.pos.to(device)
        b = batch.batch.to(device)
        true_e = batch.energy.to(device)
        true_f = batch.force.to(device)

        pred_e, pred_f = model.energy_and_force(z=z, pos=pos, batch=b)
        e_loss = torch.mean((pred_e - true_e) ** 2)
        f_loss = torch.mean((pred_f - true_f) ** 2)
        loss = e_loss + force_weight * f_loss

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        all_pred_e.append(pred_e.detach().cpu())
        all_true_e.append(true_e.detach().cpu())
        all_pred_f.append(pred_f.detach().cpu())
        all_true_f.append(true_f.detach().cpu())

    return compute_metrics(
        pred_energy=torch.cat(all_pred_e),
        true_energy=torch.cat(all_true_e),
        pred_force=torch.cat(all_pred_f),
        true_force=torch.cat(all_true_f),
    )


def train(config: TrainerConfig, split: str = "train", cache_dir: str | None = None) -> Dict[str, float]:
    return train_for_dataset(config=config, dataset_name="Transition1X", split=split, cache_dir=cache_dir)


def train_for_dataset(
    config: TrainerConfig,
    dataset_name: str = "Transition1X",
    split: str = "train",
    cache_dir: str | None = None,
) -> Dict[str, float]:
    if cache_dir is None:
        dataset = OpenQDCLoader(
            dataset_name=dataset_name,
            dataset=SyntheticTransition1XDataset(size=config.max_samples or 32),
            max_samples=config.max_samples,
        )
    else:
        dataset = OpenQDCLoader(
            dataset_name=dataset_name,
            split=split,
            cache_dir=cache_dir,
            max_samples=config.max_samples,
        )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_molecules)
    model = EnergyModel().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    metrics = {}
    for _ in range(config.epochs):
        metrics = run_epoch(model, loader, optimizer=optimizer, force_weight=config.force_weight, device=config.device)
    return metrics
