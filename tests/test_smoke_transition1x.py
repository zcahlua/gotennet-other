from __future__ import annotations

import os
import pytest

from gotennet_other.train import TrainerConfig, evaluate_checkpoint, train_for_dataset


def test_transition1x_real_smoke_if_cache_present():
    cache_dir = os.environ.get('OPENQDC_CACHE_DIR')
    if not cache_dir:
        pytest.skip('Set OPENQDC_CACHE_DIR to run real Transition1X smoke test.')
    metrics = train_for_dataset(TrainerConfig(epochs=1, batch_size=2), dataset_name='Transition1X', cache_dir=cache_dir)
    assert 'energy_mae' in metrics


def test_eval_uses_checkpoint_and_split(tmp_path):
    ckpt_dir = tmp_path / 'ckpt'
    cfg = TrainerConfig(epochs=1, batch_size=2, max_samples=8, checkpoint_path=str(ckpt_dir))
    train_for_dataset(cfg, dataset_name='Transition1X', split='train', cache_dir=None)
    metrics = evaluate_checkpoint(cfg, str(ckpt_dir / 'best.pt'), dataset_name='Transition1X', split='test', cache_dir=None)
    assert 'energy_rmse' in metrics


def test_sn2rxn_real_smoke_if_cache_present():
    cache_dir = os.environ.get('OPENQDC_CACHE_DIR')
    if not cache_dir:
        pytest.skip('Set OPENQDC_CACHE_DIR to run real SN2RXN smoke test.')
    metrics = train_for_dataset(TrainerConfig(epochs=1, batch_size=2), dataset_name='SN2RXN', cache_dir=cache_dir)
    assert 'energy_mae' in metrics
