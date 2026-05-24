from __future__ import annotations

import torch
from gotennet_other.data import SN2RXNLoader, Transition1XLoader, collate_molecules


class FakeOpenQDC:
    def __init__(self):
        self.data = {"subset": ["train", "val", "test"], "name": ["a", "b", "c"]}
        self.items = [
            {"atomic_numbers": [1, 8], "positions": [[0,0,0],[1,0,0]], "formation_energies": [1.0], "forces": [[[1.0],[2.0],[3.0]], [[4.0],[5.0],[6.0]]]},
            {"z": [6], "pos": [[0.1,0.2,0.3]], "formation_energies": [2.0], "forces": [[0.1,0.2,0.3]]},
            {"z": [9], "pos": [[0.0,0.0,0.0]], "energies": [3.0], "force": None},
        ]
    def __len__(self): return 3
    def __getitem__(self, idx): return self.items[idx]


def test_split_applied_inside_loader_and_force_method_axis_selected():
    ds = Transition1XLoader(dataset=FakeOpenQDC(), split='train')
    item = ds[0]
    assert len(ds) == 1
    assert item['force'].shape == (2, 3)
    assert torch.allclose(item['force'][0], torch.tensor([1.0,2.0,3.0]))


def test_sn2rxn_energy_target_energies_supported():
    ds = SN2RXNLoader(dataset=FakeOpenQDC(), split='test', energy_target='energies')
    assert ds[0]['energy'].item() == 3.0


def test_missing_force_not_fabricated_and_collate_handles_none():
    ds = SN2RXNLoader(dataset=FakeOpenQDC(), split='test', energy_target='energies')
    batch = collate_molecules([ds[0]])
    assert ds[0]['force'] is None
    assert batch.force is None
