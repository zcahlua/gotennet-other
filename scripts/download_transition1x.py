from __future__ import annotations
import argparse, shutil, subprocess, sys

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache-dir',default='data/openqdc'); p.add_argument('--as-zarr',action='store_true'); p.add_argument('--overwrite',action='store_true'); p.add_argument('--gs',action='store_true'); a=p.parse_args()
    if shutil.which('openqdc') is None:
        print('OpenQDC CLI not found. Install with:\n  pip install -e ".[openqdc]"'); raise SystemExit(1)
    cmd=['openqdc','download','Transition1X','--cache-dir',a.cache_dir]
    if a.as_zarr: cmd.append('--as-zarr')
    if a.overwrite: cmd.append('--overwrite')
    if a.gs: cmd.append('--gs')
    raise SystemExit(subprocess.run(cmd).returncode)
if __name__=='__main__': main()
