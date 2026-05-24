from __future__ import annotations
import torch
from gotennet_other.model import EnergyModel
from gotennet_other.train import TrainerConfig, SyntheticSN2RXNDataset, train, evaluate
from gotennet_other.config import load_trainer_config

def test_sn2_synthetic_contains_heavy_halogens():
    ds=SyntheticSN2RXNDataset(size=16); zs=set()
    for i in range(len(ds)): zs.update(ds[i]['z'].tolist())
    assert 35 in zs and 53 in zs

def test_iodine_and_noedge_force_finite():
    m=EnergyModel(max_z=100, cutoff=0.1)
    z=torch.tensor([53,6,1,1,1,9]); pos=torch.tensor([[0.,0,0],[9.,0,0],[0,9,0],[0,0,9],[9,9,9],[5,5,5]],dtype=torch.float32); b=torch.zeros(6,dtype=torch.long)
    e,f=m.energy_and_force(z,pos,b)
    assert torch.isfinite(e).all() and torch.isfinite(f).all() and f.shape==(6,3)

def test_train_checkpoint_and_eval(tmp_path):
    cfg=TrainerConfig(dataset_name='synthetic_sn2rxn',epochs=1,batch_size=2,max_samples=8,output_dir=str(tmp_path/'out'))
    m=train(cfg)
    assert (tmp_path/'out/checkpoints/last.pt').exists() and (tmp_path/'out/checkpoints/best.pt').exists() and 'energy_mae' in m
    em=evaluate(cfg, str(tmp_path/'out/checkpoints/best.pt'), split='test'); assert 'energy_rmse' in em

def test_config_preserves_max_samples():
    cfg=load_trainer_config('configs/sn2rxn_smoke.yaml')
    assert cfg.dataset_name=='sn2rxn' and cfg.max_samples==32 and cfg.energy_target=='energies'
