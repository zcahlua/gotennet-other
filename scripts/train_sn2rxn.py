from __future__ import annotations
import argparse, yaml
from gotennet_other.train import TrainerConfig, train_for_dataset

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--config', required=True); args=p.parse_args()
    cfg=yaml.safe_load(open(args.config,'r',encoding='utf-8'))
    tcfg=TrainerConfig(batch_size=cfg.get('batch_size',8), epochs=cfg.get('epochs',1), lr=cfg.get('lr',1e-3), force_weight=cfg.get('force_weight',10.0), device=cfg.get('device','cpu'), max_samples=cfg.get('max_samples'), split_seed=cfg.get('split_seed',0), checkpoint_path=cfg.get('checkpoint_path'), hidden_dim=cfg.get('hidden_dim',64))
    print(train_for_dataset(config=tcfg, dataset_name='SN2RXN', split=cfg.get('split','train'), cache_dir=cfg.get('cache_dir')))
if __name__=='__main__': main()
