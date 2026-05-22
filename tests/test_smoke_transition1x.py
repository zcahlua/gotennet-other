from __future__ import annotations

import os
from pathlib import Path

import pytest

from gotennet_other.train import TrainerConfig, train


def test_transition1x_real_smoke_if_cache_present():
    cache_dir = os.environ.get("OPENQDC_CACHE_DIR")
    if not cache_dir:
        pytest.skip("Set OPENQDC_CACHE_DIR to run real Transition1X smoke test.")
    try:
        metrics = train(TrainerConfig(epochs=1, batch_size=2), cache_dir=cache_dir)
    except RuntimeError as exc:
        if "openqdc is required" in str(exc):
            pytest.skip("openqdc not installed in test environment.")
        raise
    assert "energy_mae" in metrics


def test_train_script_wires_max_samples_from_config():
    script = Path("scripts/train_transition1x.py").read_text(encoding="utf-8")
    assert 'max_samples=cfg.get("max_samples")' in script
