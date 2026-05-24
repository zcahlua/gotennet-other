from __future__ import annotations
import yaml, torch
from pathlib import Path
from gotennet_other.train import TrainerConfig, train, evaluate
from gotennet_other.model import EnergyModel
from gotennet_other.data import OpenQDCLoader, normalize_dataset_name

class TinyDS:
    def __len__(self): return 20
    def __getitem__(self, idx):
        n=3
        pos=torch.randn(n,3)
        item={"atomic_numbers":torch.tensor([1,6,8]),"positions":pos,"formation_energies":pos.pow(2).sum().reshape(1),"name":f"grp-{idx//2}"}
        if idx%2==0: item["forces"]=-2*pos
        return item

def test_full_and_smoke_configs():
    f=yaml.safe_load(open('configs/transition1x_full.yaml')); assert f['max_samples'] is None
    s=yaml.safe_load(open('configs/transition1x_smoke.yaml')); assert s['max_samples'] is not None

def test_norm_names():
    assert normalize_dataset_name('SN2_RXN')=='sn2rxn'

def test_model_layers_and_iodine():
    m=EnergyModel(num_layers=3,max_z=100)
    z=torch.tensor([53,1]); pos=torch.randn(2,3); b=torch.tensor([0,0])
    e,f=m.energy_and_force(z,pos,b)
    assert e.shape==(1,) and f.shape==(2,3)

def test_train_eval_checkpoint(tmp_path):
    cfg=TrainerConfig(dataset_name='transition1x',device='cpu',epochs=2,batch_size=4,output_dir=str(tmp_path/'out'),max_samples=None,num_workers=0)
    res=train(cfg,dataset=TinyDS())
    ck=Path(cfg.output_dir)/'checkpoints'
    assert (ck/'last.pt').exists() and (ck/'best.pt').exists()
    out=evaluate(cfg,str(ck/'best.pt'),'val',dataset=TinyDS())
    assert 'loss' in out
    payload=torch.load(ck/'last.pt',map_location='cpu')
    for k in ['model','optimizer','scheduler','epoch','metrics','best_metric','config']:
        assert k in payload
