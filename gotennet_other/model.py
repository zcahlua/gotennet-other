from __future__ import annotations

import torch
from torch import nn


class EnergyModel(nn.Module):
    """Local GotenNet-style fallback, not official architecture."""

    def __init__(self, hidden_dim: int = 256, num_layers: int = 6, max_z: int = 100, cutoff: float = 5.0, num_rbf: int = 64):
        super().__init__()
        self.cutoff = cutoff
        self.embedding = nn.Embedding(max_z + 1, hidden_dim)
        self.register_buffer("rbf_centers", torch.linspace(0.0, cutoff, num_rbf))
        self.rbf_gamma = 10.0 / max(cutoff, 1e-6)
        self.msg_mlps = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim * 2 + num_rbf + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)) for _ in range(num_layers)])
        self.updates = nn.ModuleList([nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)) for _ in range(num_layers)])
        self.atom_mlp = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 1))

    def _build_edges(self, pos, batch):
        diff = pos[:, None, :] - pos[None, :, :]
        dist = torch.linalg.norm(diff, dim=-1)
        mask = (batch[:, None] == batch[None, :]) & (~torch.eye(pos.shape[0], device=pos.device, dtype=torch.bool)) & (dist < self.cutoff)
        src, dst = torch.where(mask)
        return src, dst, dist[src, dst]

    def _rbf(self, d):
        return torch.exp(-self.rbf_gamma * (d[:, None] - self.rbf_centers[None, :]).pow(2))

    def forward(self, z, pos, batch):
        h = self.embedding(z)
        src, dst, dist = self._build_edges(pos, batch)
        for msg_mlp, upd in zip(self.msg_mlps, self.updates):
            if src.numel() > 0:
                rbf = self._rbf(dist)
                cutoff = 0.5 * (torch.cos(torch.pi * dist / self.cutoff) + 1.0)
                msg = msg_mlp(torch.cat([h[src], h[dst], rbf, cutoff[:, None]], dim=-1)) * cutoff[:, None]
                agg = torch.zeros_like(h)
                agg.index_add_(0, dst, msg)
                h = h + upd(agg)
        per_atom = self.atom_mlp(h).squeeze(-1)
        n_mols = int(batch.max().item()) + 1 if batch.numel() else 0
        energy = torch.zeros(n_mols, device=per_atom.device, dtype=per_atom.dtype)
        energy.index_add_(0, batch, per_atom)
        return energy + 0.0 * pos.sum()

    def energy_and_force(self, z, pos, batch, create_graph=True):
        pos = pos.clone().detach().requires_grad_(True)
        energy = self(z=z, pos=pos, batch=batch)
        grad = torch.autograd.grad(energy.sum(), pos, create_graph=create_graph, allow_unused=True)[0]
        force = -grad if grad is not None else torch.zeros_like(pos)
        return energy, force
