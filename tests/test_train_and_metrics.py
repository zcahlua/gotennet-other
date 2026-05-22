from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from gotennet_other.data import collate_molecules
from gotennet_other.model import EnergyModel
from gotennet_other.train import run_epoch


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
