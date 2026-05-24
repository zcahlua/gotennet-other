from __future__ import annotations
import yaml
from .data import normalize_dataset_name
from .train import TrainerConfig

def load_trainer_config(path: str, default_dataset_name: str | None = None, overrides: dict | None = None) -> TrainerConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if default_dataset_name and "dataset_name" not in data:
        data["dataset_name"] = default_dataset_name
    if overrides:
        data.update(overrides)
    data["dataset_name"] = normalize_dataset_name(data.get("dataset_name", "synthetic"))
    return TrainerConfig(**data)
