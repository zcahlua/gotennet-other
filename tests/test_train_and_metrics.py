from __future__ import annotations

from gotennet_other.train import TrainerConfig, SyntheticSN2RXNDataset, train_for_dataset


def test_sn2_synthetic_contains_heavy_halogens():
    ds = SyntheticSN2RXNDataset(size=64)
    zs = set()
    for i in range(len(ds)):
        zs.update(ds[i]['z'].tolist())
    assert 35 in zs and 53 in zs


def test_checkpointing_writes_best_and_last_with_state(tmp_path):
    out = tmp_path / 'ckpts'
    cfg = TrainerConfig(epochs=2, batch_size=2, max_samples=8, checkpoint_path=str(out))
    train_for_dataset(cfg, dataset_name='Transition1X', cache_dir=None)
    import torch
    best = torch.load(out / 'best.pt', map_location='cpu')
    last = torch.load(out / 'last.pt', map_location='cpu')
    assert 'optimizer_state_dict' in best and 'epoch' in best and 'metrics' in best and 'config' in best
    assert last['epoch'] == 2
