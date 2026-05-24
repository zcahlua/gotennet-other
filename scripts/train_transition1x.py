from __future__ import annotations

import argparse
import yaml

from gotennet_other.train import TrainerConfig, train_for_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_cfg = TrainerConfig(
        batch_size=cfg.get("batch_size", 8),
        epochs=cfg.get("epochs", 1),
        lr=cfg.get("lr", 1e-3),
        force_weight=cfg.get("force_weight", 10.0),
        device=cfg.get("device", "cpu"),
        max_samples=cfg.get("max_samples"),
        split_seed=cfg.get("split_seed", 0),
        checkpoint_path=cfg.get("checkpoint_path"),
    )
    metrics = train_for_dataset(
        config=train_cfg,
        dataset_name=cfg.get("dataset_name", "Transition1X"),
        split=cfg.get("split", "train"),
        cache_dir=cfg.get("cache_dir"),
    )
    print(metrics)


if __name__ == "__main__":
    main()
