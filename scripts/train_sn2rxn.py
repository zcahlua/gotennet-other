from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gotennet_other.config import load_trainer_config
from gotennet_other.train import train,evaluate,train_eval

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--mode',choices=['train','eval','train_eval'],default='train'); p.add_argument('--checkpoint'); p.add_argument('--resume'); p.add_argument('--eval-split',default='test'); a=p.parse_args()
    if a.mode=='eval' and not a.checkpoint: p.error('--checkpoint is required for --mode eval')
    if a.mode=='train' and a.checkpoint: p.error('--checkpoint is not valid for --mode train')
    cfg=load_trainer_config(a.config, default_dataset_name='sn2rxn')
    res = train(cfg,resume=a.resume) if a.mode=='train' else (evaluate(cfg,a.checkpoint,split=a.eval_split) if a.mode=='eval' else train_eval(cfg,resume=a.resume,eval_split=a.eval_split))
    print(res)
if __name__=='__main__': main()
