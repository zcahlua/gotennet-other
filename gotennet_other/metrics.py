from __future__ import annotations

from typing import Dict

import torch


def compute_metrics(pred_energy: torch.Tensor, true_energy: torch.Tensor, pred_force: torch.Tensor, true_force: torch.Tensor) -> Dict[str, float]:
    e_mae = torch.mean(torch.abs(pred_energy - true_energy)).item()
    e_rmse = torch.sqrt(torch.mean((pred_energy - true_energy) ** 2)).item()
    f_mae = torch.mean(torch.abs(pred_force - true_force)).item()
    f_rmse = torch.sqrt(torch.mean((pred_force - true_force) ** 2)).item()
    return {"energy_mae": e_mae, "energy_rmse": e_rmse, "force_mae": f_mae, "force_rmse": f_rmse}
