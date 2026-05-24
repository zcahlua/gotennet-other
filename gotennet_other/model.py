from __future__ import annotations

import torch
from torch import nn


class EnergyModel(nn.Module):
    """Simple pairwise message passing model with autograd forces."""

    def __init__(self, hidden_dim: int = 64, max_z: int = 100, cutoff: float = 5.0, num_rbf: int = 16):
        super().__init__()
        self.cutoff = cutoff
        self.embedding = nn.Embedding(max_z + 1, hidden_dim)
        self.register_buffer("rbf_centers", torch.linspace(0.0, cutoff, num_rbf))
        self.rbf_gamma = 10.0 / max(cutoff, 1e-6)
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_rbf + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.atom_mlp = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 1))

    def _build_edges(self, pos: torch.Tensor, batch: torch.Tensor):
        diff = pos[:, None, :] - pos[None, :, :]
        dist = torch.linalg.norm(diff, dim=-1)
        same_mol = batch[:, None] == batch[None, :]
        not_self = ~torch.eye(pos.shape[0], device=pos.device, dtype=torch.bool)
        mask = same_mol & not_self & (dist < self.cutoff)
        src, dst = torch.where(mask)
        return src, dst, dist[src, dst]

    def _rbf(self, d: torch.Tensor):
        return torch.exp(-self.rbf_gamma * (d[:, None] - self.rbf_centers[None, :]).pow(2))

    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = self.embedding(z)
        src, dst, dist = self._build_edges(pos, batch)
        if src.numel() > 0:
            rbf = self._rbf(dist)
            cutoff = 0.5 * (torch.cos(torch.pi * dist / self.cutoff) + 1.0)
            edge_feat = torch.cat([h[src], h[dst], rbf, cutoff[:, None]], dim=-1)
            msg = self.msg_mlp(edge_feat) * cutoff[:, None]
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, msg)
            h = h + agg

        per_atom_e = self.atom_mlp(h).squeeze(-1)
        n_mols = int(batch.max().item()) + 1 if batch.numel() else 0
        energy = torch.zeros(n_mols, device=per_atom_e.device, dtype=per_atom_e.dtype)
        energy.index_add_(0, batch, per_atom_e)
        return energy

    def energy_and_force(self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor):
        pos = pos.clone().detach().requires_grad_(True)
        energy = self(z=z, pos=pos, batch=batch)
        grad = torch.autograd.grad(energy.sum(), pos, create_graph=True, allow_unused=True)[0]
        if grad is None:
            grad = torch.zeros_like(pos)
        force = -grad
        return energy, force
