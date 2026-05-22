from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()
    root = Path(args.cache_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    try:
        from openqdc.datasets import Transition1X
    except Exception as exc:
        raise RuntimeError('openqdc is required. Install with `pip install -e ".[openqdc]"`.') from exc

    kwargs = dict(cache_dir=str(root), array_format="torch", energy_unit="ev", distance_unit="ang", energy_type="formation")
    Transition1X(**kwargs)
    print(f"Transition1X initialized with cache at: {root}")


if __name__ == "__main__":
    main()
