from __future__ import annotations

import math

def finalize_metrics(total_loss, n_batches, e_abs, e_sq, e_abs_pa, e_n, f_abs, f_sq, f_n, lr, epoch):
    out = {
        "loss": total_loss / max(n_batches, 1),
        "energy_mae": e_abs / max(e_n, 1),
        "energy_rmse": math.sqrt(e_sq / max(e_n, 1)),
        "energy_mae_per_atom": e_abs_pa / max(e_n, 1),
        "force_mae": float("nan") if f_n == 0 else f_abs / f_n,
        "force_rmse": float("nan") if f_n == 0 else math.sqrt(f_sq / f_n),
        "lr": lr,
        "epoch": epoch,
    }
    return out
