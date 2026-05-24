from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--dataset-name", default="Transition1X")
    args = parser.parse_args()
    root = Path(args.cache_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    try:
        from openqdc import datasets as oqdc_datasets
    except Exception as exc:
        raise RuntimeError('openqdc is required. Install with `pip install -e ".[openqdc]"`.') from exc

    if not hasattr(oqdc_datasets, args.dataset_name):
        raise RuntimeError(f"OpenQDC dataset '{args.dataset_name}' is not available in installed openqdc version.")
    dataset_cls = getattr(oqdc_datasets, args.dataset_name)

    kwargs = dict(cache_dir=str(root), array_format="torch", energy_unit="ev", distance_unit="ang", energy_type="formation")
    dataset_cls(**kwargs)
    print(f"{args.dataset_name} initialized with cache at: {root}")


if __name__ == "__main__":
    main()
