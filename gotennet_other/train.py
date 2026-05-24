from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset

from .data import OpenQDCLoader, collate_molecules, split_dataset
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
    split_seed: int = 0
    checkpoint_path: str | None = None
    hidden_dim: int = 64


class SyntheticTransition1XDataset(Dataset):
    def __init__(self, size: int = 32): self.size = size
    def __len__(self): return self.size
    def __getitem__(self, idx):
        n = 2 + (idx % 3)
        return {"z": torch.randint(1, 10, (n,)), "pos": (pos := torch.randn(n, 3)), "energy": pos.pow(2).sum().reshape(1), "force": -2 * pos}


class SyntheticSN2RXNDataset(Dataset):
    def __init__(self, size: int = 32): self.size = size
    def __len__(self): return self.size
    def __getitem__(self, idx):
        n = 4 + (idx % 3)
        halogens = torch.tensor([9, 17, 35, 53])
        z = halogens[torch.randint(0, len(halogens), (n,))]
        pos = torch.randn(n, 3)
        rc = pos[:, 0].sum()
        return {"z": z, "pos": pos, "energy": (0.5 * pos.pow(2).sum() + 0.1 * rc).reshape(1), "force": -(pos + torch.tensor([0.1, 0.0, 0.0]))}


def run_epoch(model: EnergyModel, loader: DataLoader, optimizer=None, force_weight: float = 10.0, device: str = "cpu") -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    all_pred_e, all_true_e, all_pred_f, all_true_f = [], [], [], []
    for batch in loader:
        pred_e, pred_f = model.energy_and_force(z=batch.z.to(device), pos=batch.pos.to(device), batch=batch.batch.to(device))
        true_e = batch.energy.to(device)
        e_loss = torch.mean((pred_e - true_e) ** 2)
        loss = e_loss
        if batch.force is not None:
            true_f = batch.force.to(device)
            f_loss = torch.mean((pred_f - true_f) ** 2)
            loss = loss + force_weight * f_loss
            all_true_f.append(true_f.detach().cpu()); all_pred_f.append(pred_f.detach().cpu())

        if train_mode:
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        all_pred_e.append(pred_e.detach().cpu()); all_true_e.append(true_e.detach().cpu())

    metrics = compute_metrics(pred_energy=torch.cat(all_pred_e), true_energy=torch.cat(all_true_e),
                              pred_force=torch.cat(all_pred_f) if all_pred_f else torch.zeros(0,3),
                              true_force=torch.cat(all_true_f) if all_true_f else torch.zeros(0,3))
    return metrics


def evaluate_checkpoint(config: TrainerConfig, checkpoint_path: str, dataset_name: str, split: str, cache_dir: str | None):
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model = EnergyModel(hidden_dim=checkpoint.get("model_hparams", {}).get("hidden_dim", config.hidden_dim)).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    synthetic_cls = SyntheticSN2RXNDataset if dataset_name == 'SN2RXN' else SyntheticTransition1XDataset
    dataset = OpenQDCLoader(dataset_name=dataset_name, split=split, cache_dir=cache_dir,
                            dataset=synthetic_cls(size=config.max_samples or 32) if cache_dir is None else None,
                            max_samples=config.max_samples)
    split_ds = split_dataset(dataset, split=split, seed=config.split_seed)
    loader = DataLoader(split_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_molecules)
    return run_epoch(model, loader, optimizer=None, force_weight=config.force_weight, device=config.device)


def train(config: TrainerConfig, split: str = "train", cache_dir: str | None = None) -> Dict[str, float]:
    return train_for_dataset(config=config, dataset_name="Transition1X", split=split, cache_dir=cache_dir)


def train_for_dataset(config: TrainerConfig, dataset_name: str = "Transition1X", split: str = "train", cache_dir: str | None = None) -> Dict[str, float]:
    synthetic_cls = SyntheticSN2RXNDataset if dataset_name == "SN2RXN" else SyntheticTransition1XDataset
    dataset = OpenQDCLoader(dataset_name=dataset_name, split=split, cache_dir=cache_dir,
                            dataset=synthetic_cls(size=config.max_samples or 32) if cache_dir is None else None,
                            max_samples=config.max_samples)
    split_ds = split_dataset(dataset, split=split, seed=config.split_seed)
    loader = DataLoader(split_ds, batch_size=config.batch_size, shuffle=(split == "train"), collate_fn=collate_molecules)
    model = EnergyModel(hidden_dim=config.hidden_dim).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_metrics, best_score = {}, float("inf")
    for epoch in range(1, config.epochs + 1):
        metrics = run_epoch(model, loader, optimizer=optimizer, force_weight=config.force_weight, device=config.device)
        score = metrics["energy_rmse"]
        if score < best_score:
            best_score, best_metrics = score, metrics
            if config.checkpoint_path:
                _save_ckpt(config.checkpoint_path, "best.pt", model, optimizer, epoch, metrics, config, dataset_name)
        if config.checkpoint_path:
            _save_ckpt(config.checkpoint_path, "last.pt", model, optimizer, epoch, metrics, config, dataset_name)
    return best_metrics or metrics


def _save_ckpt(base_path: str, name: str, model, optimizer, epoch: int, metrics: Dict[str, float], config: TrainerConfig, dataset_name: str):
    out_dir = Path(base_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch,
                "metrics": metrics, "config": asdict(config), "dataset_name": dataset_name,
                "model_hparams": {"hidden_dim": config.hidden_dim}}, out_dir / name)
