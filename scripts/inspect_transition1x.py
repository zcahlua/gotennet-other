from __future__ import annotations

import argparse

from gotennet_other.data import Transition1XLoader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    ds = Transition1XLoader(split=args.split, cache_dir=args.cache_dir, max_samples=1)
    sample = ds[0]
    print({k: (v.shape if hasattr(v, "shape") else v) for k, v in sample.items()})


if __name__ == "__main__":
    main()
