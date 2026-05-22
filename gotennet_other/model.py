from __future__ import annotations

import torch
from torch import nn


class EnergyModel(nn.Module):
    """Small energy model with differentiable forces from -dE/dR."""

    def __init__(self, hidden_dim: int = 64, max_z: int = 100):
        super().__init__()
        self.embedding = nn.Embedding(max_z + 1, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        r = pos.norm(dim=-1, keepdim=True)
        h = torch.cat([self.embedding(z), r], dim=-1)
        per_atom_e = self.mlp(h).squeeze(-1)
        n_mols = int(batch.max().item()) + 1 if batch.numel() else 0
        energy = torch.zeros(n_mols, device=per_atom_e.device, dtype=per_atom_e.dtype)
        energy.index_add_(0, batch, per_atom_e)
        return energy

    def energy_and_force(self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor):
        pos = pos.clone().detach().requires_grad_(True)
        energy = self(z=z, pos=pos, batch=batch)
        force = -torch.autograd.grad(energy.sum(), pos, create_graph=True)[0]
        return energy, force
