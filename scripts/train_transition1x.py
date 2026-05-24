from __future__ import annotations
import argparse, yaml
from gotennet_other.train import TrainerConfig, train, evaluate

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--mode',choices=['train','eval','train_eval'],default='train'); p.add_argument('--checkpoint'); p.add_argument('--resume'); p.add_argument('--eval-split',default='test',choices=['train','val','test','all']); a=p.parse_args()
    cfg=TrainerConfig(**yaml.safe_load(open(a.config,'r',encoding='utf-8')))
    if a.mode=='train': print(train(cfg,resume=a.resume))
    elif a.mode=='eval':
        if not a.checkpoint: raise ValueError('--checkpoint required for --mode eval')
        print(evaluate(cfg,a.checkpoint,a.eval_split))
    else:
        train(cfg,resume=a.resume); ck=a.checkpoint or f"{cfg.output_dir}/checkpoints/best.pt"; print(evaluate(cfg,ck,a.eval_split))
if __name__=='__main__': main()
