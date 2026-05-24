from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gotennet_other.data import Transition1XLoader

def main():
 p=argparse.ArgumentParser(); p.add_argument('--cache-dir',required=True); p.add_argument('--index',type=int,default=0); p.add_argument('--split',default='train'); a=p.parse_args(); ds=Transition1XLoader(split=a.split,cache_dir=a.cache_dir); x=ds[a.index]; print(x)
if __name__=='__main__': main()
