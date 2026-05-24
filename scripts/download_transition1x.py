from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Transition1X via OpenQDC CLI")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--as-zarr", action="store_true")
    parser.add_argument("--gs", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["openqdc", "download", "Transition1X", "--cache-dir", str(cache_dir)]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.as_zarr:
        cmd.append("--as-zarr")
    if args.gs:
        cmd.append("--gs")

    subprocess.run(cmd, check=True)
    print(f"Downloaded Transition1X to: {cache_dir}")


if __name__ == "__main__":
    main()
