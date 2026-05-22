from __future__ import annotations

import os

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
