from __future__ import annotations
import torch
from gotennet_other.data import SN2RXNLoader, collate_molecules, normalize_dataset_name

class FakeDataset:
    def __init__(self, items): self.items=items; self.data={"name":[getattr(x,'name','a') for x in items], "subset":[getattr(x,'subset','train') for x in items]}
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

class FakeSN2Bunch:
    atomic_numbers=torch.tensor([9,6,1,1,1,17]); positions=torch.randn(6,3); charges=torch.zeros(6,dtype=torch.long)
    energies=torch.tensor([-123.4,-122.0]); formation_energies=torch.tensor([-10.0,-9.5]); forces=torch.randn(6,3,2)
    name='F_CH3Cl'; subset='train'

def test_normalize_names():
    assert normalize_dataset_name('SN2RXN')=='sn2rxn'
    assert normalize_dataset_name('sn2_rxn')=='sn2rxn'
    assert normalize_dataset_name('sn2-rxn')=='sn2rxn'
    assert normalize_dataset_name('Transition1X')=='transition1x'

def test_fake_sn2_mapping_and_force_axis():
    ds=SN2RXNLoader(dataset=FakeDataset([FakeSN2Bunch()]),split='all',energy_target='energies')
    x=ds[0]
    assert x['z'].tolist()==[9,6,1,1,1,17]
    assert abs(x['energy'].item()+123.4)<1e-4
    assert x['force'].shape==(6,3)
    assert x['name']=='F_CH3Cl' and x['subset']=='train' and x['n_atoms']==6

def test_missing_force_collate_none():
    item={"z":torch.tensor([1,1]),"pos":torch.randn(2,3),"energy":torch.tensor([1.0]),"force":None}
    b=collate_molecules([item]); assert b.force is None
