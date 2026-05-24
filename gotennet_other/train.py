from __future__ import annotations
import random
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from .data import OpenQDCLoader, collate_molecules, normalize_dataset_name
from .metrics import compute_metrics
from .model import EnergyModel

@dataclass
class TrainerConfig:
    dataset_name: str = "synthetic"; cache_dir: str | None = None; split: str = "train"
    batch_size: int = 8; epochs: int = 3; lr: float = 1e-3; weight_decay: float = 0.0; energy_weight: float = 1.0; force_weight: float = 10.0; device: str = "cpu"
    max_samples: int | None = None; output_dir: str = "outputs/debug"; seed: int = 42
    hidden_dim: int = 64; cutoff: float = 5.0; num_rbf: int = 16; max_z: int = 100
    grad_clip_norm: float | None = 10.0; energy_target: str | None = None; split_by_name: bool = True; read_as_zarr: bool = False; checkpoint_path: str | None = None

def seed_everything(seed: int):
    random.seed(seed); torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

class SyntheticTransition1XDataset(Dataset):
    def __init__(self, size=32, seed=42): self.size=size; self.seed=seed
    def __len__(self): return self.size
    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(self.seed+idx); n=2+(idx%3); pos=torch.randn(n,3,generator=g); z=torch.randint(1,10,(n,),generator=g)
        return {"z":z,"pos":pos,"energy":pos.pow(2).sum().reshape(1),"force":-2*pos,"n_atoms":n,"name":f"syn_{idx}","subset":"train"}

class SyntheticSN2RXNDataset(Dataset):
    def __init__(self, size=32, seed=42): self.size=size; self.seed=seed
    def __len__(self): return self.size
    def __getitem__(self, idx):
        g=torch.Generator().manual_seed(self.seed+idx); hal=torch.tensor([9,17,35,53]); n=6
        z=torch.tensor([hal[idx%4],6,1,1,1,hal[(idx+1)%4]])
        pos=torch.randn(n,3,generator=g); return {"z":z,"pos":pos,"energy":pos.pow(2).sum().reshape(1),"force":-2.0*pos,"n_atoms":n,"name":f"sn2_{idx}","subset":"train"}

def build_dataset(config: TrainerConfig, split: str | None = None):
    ds = normalize_dataset_name(config.dataset_name)
    if ds=="synthetic": return SyntheticTransition1XDataset(size=config.max_samples or 32, seed=config.seed)
    if ds=="synthetic_sn2rxn": return SyntheticSN2RXNDataset(size=config.max_samples or 32, seed=config.seed)
    return OpenQDCLoader(dataset_name=ds, split=split or config.split, cache_dir=config.cache_dir, max_samples=config.max_samples, energy_target=config.energy_target, split_by_name=config.split_by_name, seed=config.seed, read_as_zarr=config.read_as_zarr)

def build_model(config: TrainerConfig): return EnergyModel(hidden_dim=config.hidden_dim, cutoff=config.cutoff, num_rbf=config.num_rbf, max_z=config.max_z)
def build_optimizer(model, config: TrainerConfig): return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

def run_epoch(model, loader, optimizer, config: TrainerConfig, device: str):
    model.train(optimizer is not None)
    pe=[];te=[];pf=[];tf=[]
    for b in loader:
        ctx = torch.enable_grad() if optimizer is None else torch.enable_grad()
        with ctx:
            pred_e,pred_f = model.energy_and_force(b.z.to(device), b.pos.to(device), b.batch.to(device))
            true_e = b.energy.to(device); na=b.n_atoms.to(device).float(); e_loss=((pred_e/na-true_e/na)**2).mean(); loss=config.energy_weight*e_loss
            if b.force is not None and pred_f is not None:
                true_f=b.force.to(device); f_loss=((pred_f-true_f)**2).mean(); loss=loss+config.force_weight*f_loss; pf.append(pred_f.detach().cpu()); tf.append(true_f.detach().cpu())
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True); loss.backward();
                if config.grad_clip_norm is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                optimizer.step()
        pe.append(pred_e.detach().cpu()); te.append(true_e.detach().cpu())
    return compute_metrics(torch.cat(pe), torch.cat(te), torch.cat(pf) if pf else torch.zeros(0,3), torch.cat(tf) if tf else torch.zeros(0,3))

def save_checkpoint(path, model, optimizer, epoch, metrics, best_metric, config):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch,"metrics":metrics,"best_metric":best_metric,"config":asdict(config)}, path)

def train(config: TrainerConfig, resume: str|None=None):
    seed_everything(config.seed); device=config.device
    tr=DataLoader(build_dataset(config,"train"),batch_size=config.batch_size,shuffle=True,collate_fn=collate_molecules)
    va=DataLoader(build_dataset(config,"val"),batch_size=config.batch_size,shuffle=False,collate_fn=collate_molecules)
    model=build_model(config).to(device); opt=build_optimizer(model,config); start=1; best=float("inf"); bestm={}
    if resume:
        ck=torch.load(resume,map_location=device); model.load_state_dict(ck["model"]); opt.load_state_dict(ck.get("optimizer",{})); start=ck.get("epoch",0)+1
    if config.checkpoint_path:
        config.output_dir = config.checkpoint_path
    for e in range(start, config.epochs+1):
        trm=run_epoch(model,tr,opt,config,device); vam=run_epoch(model,va,None,config,device) if len(va.dataset)>0 else trm
        if len(va.dataset)==0: print("Validation split is empty; using training loss for best checkpoint selection.")
        cdir=(Path(config.checkpoint_path) if config.checkpoint_path else Path(config.output_dir)/"checkpoints")
        save_checkpoint(cdir/"last.pt",model,opt,e,vam,best,config)
        if vam["loss"]<best: best=vam["loss"]; bestm=vam; save_checkpoint(cdir/"best.pt",model,opt,e,vam,best,config)
    return bestm or trm

def evaluate(config: TrainerConfig, checkpoint: str, split: str="test"):
    ck=torch.load(checkpoint,map_location=config.device); merged={**ck.get("config",{}), **asdict(config)}; merged["output_dir"]=config.output_dir; cfg=TrainerConfig(**merged)
    model=build_model(cfg).to(cfg.device); model.load_state_dict(ck["model"])
    dl=DataLoader(build_dataset(cfg, split), batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_molecules)
    return run_epoch(model, dl, None, cfg, cfg.device)

def train_eval(config: TrainerConfig, resume: str|None=None, eval_split: str="test"):
    tr=train(config,resume=resume); ev=evaluate(config,str(Path(config.output_dir)/"checkpoints"/"best.pt"),split=eval_split); return {"train":tr,"eval":ev}

def train_for_dataset(config: TrainerConfig, dataset_name: str="transition1x", split: str="train", cache_dir: str|None=None):
    dsname = normalize_dataset_name(dataset_name)
    if cache_dir is None and dsname in {"transition1x","sn2rxn"}:
        dsname = "synthetic_sn2rxn" if dsname=="sn2rxn" else "synthetic"
    c = TrainerConfig(**{**asdict(config),"dataset_name":dsname,"split":split,"cache_dir":cache_dir or config.cache_dir})
    return train(c)


def evaluate_checkpoint(config: TrainerConfig, checkpoint_path: str, dataset_name: str, split: str, cache_dir: str | None):
    dsname = normalize_dataset_name(dataset_name)
    if cache_dir is None and dsname in {"transition1x","sn2rxn"}:
        dsname = "synthetic_sn2rxn" if dsname=="sn2rxn" else "synthetic"
    cfg = TrainerConfig(**{**asdict(config), "dataset_name": dsname, "cache_dir": cache_dir or config.cache_dir})
    return evaluate(cfg, checkpoint_path, split=split)
