from __future__ import annotations

import argparse

from gotennet_other.data import OpenQDCLoader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--max-samples", type=int, default=8)
    args = parser.parse_args()

    ds = OpenQDCLoader(
        dataset_name="transition1x",
        split=args.split,
        cache_dir=args.cache_dir,
        max_samples=args.max_samples,
    )
    print({"split": args.split, "num_samples": len(ds)})
    if len(ds) > 0:
        sample = ds[0]
        print(
            {
                "keys": sorted(sample.keys()),
                "z_shape": tuple(sample["z"].shape),
                "pos_shape": tuple(sample["pos"].shape),
                "energy_shape": tuple(sample["energy"].shape),
                "has_force": sample["force"] is not None,
            }
        )


if __name__ == "__main__":
    main()
