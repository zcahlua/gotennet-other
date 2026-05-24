from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from gotennet_other.data import collate_molecules
from gotennet_other.model import EnergyModel
from gotennet_other.train import TrainerConfig, run_epoch, train, train_for_dataset


class TinyMolDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        n = 2 + (idx % 2)
        z = torch.randint(1, 10, (n,))
        pos = torch.randn(n, 3)
        energy = pos.pow(2).sum().unsqueeze(0)
        force = -2 * pos
        return {"z": z, "pos": pos, "energy": energy, "force": force}


def test_run_epoch_computes_metrics_and_backprop():
    ds = TinyMolDataset()
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_molecules)
    model = EnergyModel(hidden_dim=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = run_epoch(model, loader, optimizer=opt)
    assert set(metrics.keys()) == {"energy_mae", "energy_rmse", "force_mae", "force_rmse"}
    assert all(v >= 0 for v in metrics.values())


def test_synthetic_training_runs_without_openqdc():
    metrics = train(TrainerConfig(epochs=1, batch_size=2, max_samples=8), cache_dir=None)
    assert "energy_mae" in metrics


def test_synthetic_sn2rxn_training_runs_without_openqdc():
    metrics = train_for_dataset(TrainerConfig(epochs=1, batch_size=2, max_samples=8), dataset_name="SN2RXN", cache_dir=None)
    assert "energy_mae" in metrics


def test_checkpoint_is_written_and_split_is_supported(tmp_path):
    ckpt = tmp_path / "model.pt"
    cfg = TrainerConfig(epochs=1, batch_size=2, max_samples=10, checkpoint_path=str(ckpt), split_seed=42)
    metrics = train_for_dataset(cfg, dataset_name="Transition1X", split="val", cache_dir=None)
    assert ckpt.exists()
    assert "energy_rmse" in metrics


def test_model_handles_single_atom_no_edges():
    model = EnergyModel(hidden_dim=8)
    z = torch.tensor([1])
    pos = torch.zeros(1, 3)
    batch = torch.tensor([0])
    e, f = model.energy_and_force(z, pos, batch)
    assert e.shape == (1,)
    assert f.shape == (1, 3)
