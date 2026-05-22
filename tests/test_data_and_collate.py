from __future__ import annotations

import pytest
import torch

from gotennet_other.data import Transition1XLoader, collate_molecules


class DummyDataset:
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        if idx == 0:
            return {"z": [1, 8], "pos": [[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]], "energy": -1.0, "force": [[0, 0, 0], [0, 0, 0]]}
        return {"z": [6], "pos": [[0.1, 0.2, 0.3]], "energy": 0.5, "force": [[0.1, -0.1, 0.0]]}


def test_collate_variable_sizes_without_padding():
    ds = Transition1XLoader(dataset=DummyDataset())
    batch = collate_molecules([ds[0], ds[1]])
    assert batch.z.shape == (3,)
    assert batch.pos.shape == (3, 3)
    assert batch.force.shape == (3, 3)
    assert torch.equal(batch.batch, torch.tensor([0, 0, 1]))


def test_missing_cache_has_actionable_message(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError, match="cache directory not found"):
        Transition1XLoader(cache_dir=str(missing))
