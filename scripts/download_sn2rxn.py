from __future__ import annotations
import argparse, shutil, subprocess

def _cmd(name,a):
    c=['openqdc','download',name,'--cache-dir',a.cache_dir]
    if a.as_zarr: c.append('--as-zarr')
    if a.overwrite: c.append('--overwrite')
    if a.gs: c.append('--gs')
    return c

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache-dir',default='data/openqdc'); p.add_argument('--as-zarr',action='store_true'); p.add_argument('--overwrite',action='store_true'); p.add_argument('--gs',action='store_true'); a=p.parse_args()
    if shutil.which('openqdc') is None:
        print('OpenQDC CLI not found. Install with:\n  pip install -e ".[openqdc]"'); raise SystemExit(1)
    rc=subprocess.run(_cmd('SN2RXN',a)).returncode
    if rc!=0: rc=subprocess.run(_cmd('sn2_rxn',a)).returncode
    raise SystemExit(rc)
if __name__=='__main__': main()
