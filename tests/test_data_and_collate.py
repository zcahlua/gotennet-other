from __future__ import annotations

import pytest
import torch

from gotennet_other.data import OpenQDCLoader, SN2RXNLoader, Transition1XLoader, collate_molecules


class DummyDataset:
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        if idx == 0:
            return {"z": [1, 8], "pos": [[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]], "energy": -1.0, "force": [[0, 0, 0], [0, 0, 0]]}
        return {"z": [6], "pos": [[0.1, 0.2, 0.3]], "energy": 0.5, "force": [[0.1, -0.1, 0.0]]}


class FakeBunch:
    def __init__(self):
        self.atomic_numbers = torch.tensor([1, 8])
        self.positions = torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
        self.charges = torch.tensor([0.1, -0.1])
        self.formation_energies = torch.tensor([-1.2])
        self.forces = torch.zeros(2, 3, 1)
        self.name = "sample-a"
        self.subset = "train"


class FakeBunchDataset:
    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return FakeBunch()


def test_collate_variable_sizes_without_padding():
    ds = Transition1XLoader(dataset=DummyDataset())
    batch = collate_molecules([ds[0], ds[1]])
    assert batch.z.shape == (3,)
    assert batch.pos.shape == (3, 3)
    assert batch.force.shape == (3, 3)
    assert torch.equal(batch.batch, torch.tensor([0, 0, 1]))


def test_openqdc_style_bunch_sample_is_mapped():
    ds = Transition1XLoader(dataset=FakeBunchDataset(), max_samples=1)
    item = ds[0]
    assert item["z"].tolist() == [1, 8]
    assert item["pos"].shape == (2, 3)
    assert item["charge"].shape == (2,)
    assert item["energy"].shape == (1,)
    assert item["force"].shape == (2, 3)
    assert item["name"] == "sample-a"
    assert item["subset"] == "train"
    assert len(ds) == 1


def test_missing_cache_has_actionable_message(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError, match="cache directory not found"):
        Transition1XLoader(cache_dir=str(missing))


def test_sn2rxn_loader_uses_shared_openqdc_adapter():
    ds = SN2RXNLoader(dataset=DummyDataset(), max_samples=1)
    assert isinstance(ds, OpenQDCLoader)
    assert ds.dataset_name == "SN2RXN"
    assert len(ds) == 1


def test_missing_force_does_not_crash_and_defaults_to_zeros():
    class NoForce:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {"z": [1, 1], "pos": [[0, 0, 0], [1, 0, 0]], "energy": [0.0]}

    ds = SN2RXNLoader(dataset=NoForce())
    item = ds[0]
    assert item["force"].shape == (2, 3)
    assert torch.allclose(item["force"], torch.zeros(2, 3))
