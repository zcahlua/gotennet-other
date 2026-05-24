from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import torch
from torch.utils.data import DataLoader, Dataset

from .data import OpenQDCLoader, collate_molecules
from .metrics import finalize_metrics
from .model import EnergyModel

@dataclass
class TrainerConfig:
    dataset_name: str = "transition1x"
    cache_dir: str | None = None
    split: str = "train"
    batch_size: int = 32
    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-6
    energy_weight: float = 1.0
    force_weight: float = 50.0
    device: str = "cuda"
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    max_samples: int | None = None
    output_dir: str = "outputs/default"
    seed: int = 42
    hidden_dim: int = 256
    num_layers: int = 6
    cutoff: float = 5.0
    num_rbf: int = 64
    max_z: int = 100
    grad_clip_norm: float | None = 10.0
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    amp: bool = True
    ema: bool = False
    ema_decay: float = 0.999
    energy_target: str | None = None
    split_by_name: bool = True
    read_as_zarr: bool = False
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1

class SyntheticDataset(Dataset):
    def __len__(self): return 32
    def __getitem__(self, idx):
        n=3; pos=torch.randn(n,3); return {"z":torch.randint(1,10,(n,)),"pos":pos,"energy":pos.pow(2).sum().reshape(1),"force":-2*pos,"n_atoms":n}

def make_loader(cfg, split, dataset=None, shuffle=False):
    ds = OpenQDCLoader(cfg.dataset_name, split=split, cache_dir=cfg.cache_dir, dataset=dataset, max_samples=cfg.max_samples,
                       energy_target=cfg.energy_target, split_by_name=cfg.split_by_name, read_as_zarr=cfg.read_as_zarr,
                       train_fraction=cfg.train_fraction, val_fraction=cfg.val_fraction, test_fraction=cfg.test_fraction, seed=cfg.seed)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        collate_fn=collate_molecules,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
    )

def run_epoch(model, loader, cfg, optimizer=None, scaler=None, training=True):
    model.train(training)
    total_loss=e_abs=e_sq=e_abs_pa=f_abs=f_sq=0.0; e_n=f_n=n_batches=0
    for batch in loader:
        z,pos,b=batch.z.to(cfg.device),batch.pos.to(cfg.device),batch.batch.to(cfg.device)
        t_e=batch.energy.to(cfg.device); n_atoms=batch.n_atoms.to(cfg.device)
        use_amp = cfg.amp and cfg.device.startswith("cuda")
        ctx = torch.cuda.amp.autocast(enabled=use_amp)
        with ctx:
            p_e,p_f=model.energy_and_force(z,pos,b,create_graph=training)
            e_loss=torch.mean(((p_e/n_atoms)-(t_e/n_atoms))**2)
            loss=cfg.energy_weight*e_loss
            if batch.force is not None and batch.has_force.any():
                t_f=batch.force.to(cfg.device)
                f_mask=torch.cat([torch.full((int(n_atoms[i].item()),), bool(batch.has_force[i]), device=cfg.device) for i in range(len(n_atoms))])
                if f_mask.any():
                    f_loss=torch.mean((p_f[f_mask]-t_f[f_mask])**2); loss=loss+cfg.force_weight*f_loss
        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                if cfg.grad_clip_norm: torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip_norm: torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()
        total_loss += float(loss.item()); n_batches += 1
        d=(p_e-t_e); e_abs += float(d.abs().sum().item()); e_sq += float((d**2).sum().item()); e_abs_pa += float((d.abs()/n_atoms).sum().item()); e_n += len(t_e)
        if batch.force is not None and batch.has_force.any():
            dd = (p_f[f_mask] - t_f[f_mask]); f_abs += float(dd.abs().sum().item()); f_sq += float((dd**2).sum().item()); f_n += dd.numel()
    lr = optimizer.param_groups[0]["lr"] if optimizer else 0.0
    return finalize_metrics(total_loss,n_batches,e_abs,e_sq,e_abs_pa,e_n,f_abs,f_sq,f_n,lr,0)

def train(cfg: TrainerConfig, dataset=None, resume: str|None=None):
    torch.manual_seed(cfg.seed)
    model=EnergyModel(cfg.hidden_dim,cfg.num_layers,cfg.max_z,cfg.cutoff,cfg.num_rbf).to(cfg.device)
    opt=torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.scheduler == "cosine":
        if cfg.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=cfg.warmup_epochs)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs - cfg.warmup_epochs, 1))
            sched = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[cfg.warmup_epochs])
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))
    else:
        sched = None
    scaler=torch.cuda.amp.GradScaler(enabled=cfg.amp and cfg.device.startswith("cuda"))
    start=0; best=float("inf"); best_state=None
    if resume:
        ck=torch.load(resume,map_location=cfg.device); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); start=ck["epoch"]+1; best=ck.get("best_metric",best)
    train_loader=make_loader(cfg,"train",dataset=dataset,shuffle=True); val_loader=make_loader(cfg,"val",dataset=dataset)
    out=Path(cfg.output_dir)/"checkpoints"; out.mkdir(parents=True,exist_ok=True)
    for ep in range(start,cfg.epochs):
        mtr=run_epoch(model,train_loader,cfg,opt,scaler,True); mtr["epoch"]=ep
        with torch.enable_grad(): valm = run_epoch(model,val_loader,cfg,None,None,False) if len(val_loader.dataset)>0 else mtr
        metric=valm["loss"]
        state={"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict() if sched else None,"epoch":ep,"metrics":{"train":mtr,"val":valm},"best_metric":best,"config":asdict(cfg)}
        torch.save(state,out/"last.pt")
        if metric < best: best=metric; state["best_metric"]=best; torch.save(state,out/"best.pt"); best_state=copy.deepcopy(model.state_dict())
        if sched: sched.step()
    return {"train":mtr,"val":valm,"best_metric":best}

def evaluate(cfg: TrainerConfig, checkpoint: str, eval_split: str = "test", dataset=None):
    user_cfg = cfg
    ck=torch.load(checkpoint,map_location=user_cfg.device)
    cfg=TrainerConfig(**ck["config"])
    cfg.device = user_cfg.device
    cfg.cache_dir = user_cfg.cache_dir
    cfg.max_samples = user_cfg.max_samples
    cfg.read_as_zarr = user_cfg.read_as_zarr
    cfg.batch_size = user_cfg.batch_size
    cfg.num_workers = user_cfg.num_workers
    cfg.pin_memory = user_cfg.pin_memory
    cfg.persistent_workers = user_cfg.persistent_workers
    model=EnergyModel(cfg.hidden_dim,cfg.num_layers,cfg.max_z,cfg.cutoff,cfg.num_rbf).to(cfg.device)
    model.load_state_dict(ck["model"])
    loader=make_loader(cfg,eval_split,dataset=dataset)
    with torch.enable_grad():
        return run_epoch(model,loader,cfg,None,None,False)
