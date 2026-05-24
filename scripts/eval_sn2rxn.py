from __future__ import annotations
import argparse, yaml
from gotennet_other.train import TrainerConfig, evaluate_checkpoint

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--checkpoint', required=True); p.add_argument('--split', default=None); a=p.parse_args()
    cfg=yaml.safe_load(open(a.config,'r',encoding='utf-8'))
    tcfg=TrainerConfig(batch_size=cfg.get('batch_size',8), force_weight=cfg.get('force_weight',10.0), device=cfg.get('device','cpu'), max_samples=cfg.get('max_samples'), split_seed=cfg.get('split_seed',0), hidden_dim=cfg.get('hidden_dim',64))
    print(evaluate_checkpoint(tcfg, a.checkpoint, 'SN2RXN', a.split or cfg.get('split','test'), cfg.get('cache_dir')))
if __name__=='__main__': main()
