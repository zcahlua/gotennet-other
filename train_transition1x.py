from __future__ import annotations

import argparse

from gotennet_other.train import TrainerConfig, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = TrainerConfig(epochs=args.epochs, batch_size=args.batch_size)
    metrics = train(config=config, split="train", cache_dir=args.cache_dir)
    print(metrics)


if __name__ == "__main__":
    main()
