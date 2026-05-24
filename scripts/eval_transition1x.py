from __future__ import annotations

import argparse
import torch
import yaml
from torch.utils.data import DataLoader

from gotennet_other.data import OpenQDCLoader, collate_molecules
from gotennet_other.model import EnergyModel
from gotennet_other.train import run_epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds = OpenQDCLoader(
        dataset_name=cfg.get("dataset_name", "Transition1X"),
        split=cfg.get("split", "train"),
        cache_dir=cfg.get("cache_dir"),
        max_samples=cfg.get("max_samples"),
    )
    loader = DataLoader(ds, batch_size=cfg.get("batch_size", 8), shuffle=False, collate_fn=collate_molecules)
    model = EnergyModel().to(cfg.get("device", "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=cfg.get("device", "cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = run_epoch(model, loader, optimizer=None, force_weight=cfg.get("force_weight", 10.0), device=cfg.get("device", "cpu"))
    print(metrics)


if __name__ == "__main__":
    main()
