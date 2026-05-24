from __future__ import annotations
import argparse, subprocess

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--cache-dir', required=True); args=p.parse_args()
    subprocess.run(['openqdc','download','SN2RXN','--cache-dir',args.cache_dir], check=True)
if __name__=='__main__': main()
