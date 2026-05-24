from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SN2RXN via OpenQDC CLI")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--as-zarr", action="store_true")
    parser.add_argument("--gs", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    primary = ["openqdc", "download", "SN2RXN", "--cache-dir", str(cache_dir)]
    alt = ["openqdc", "download", "sn2_rxn", "--cache-dir", str(cache_dir)]
    for flag in ("--overwrite", "--as-zarr", "--gs"):
        if getattr(args, flag[2:].replace("-", "_")):
            primary.append(flag)
            alt.append(flag)

    try:
        subprocess.run(primary, check=True)
    except subprocess.CalledProcessError:
        subprocess.run(alt, check=True)

    print(f"Downloaded SN2RXN to: {cache_dir}")


if __name__ == "__main__":
    main()
