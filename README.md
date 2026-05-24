# gotennet-other

Minimal PyTorch scaffold for OpenQDC `Transition1X` and `SN2RXN` energy/force training.

## Install
```bash
pip install -e ".[dev]"
pip install -e ".[openqdc]"
```

## Download
```bash
python scripts/download_transition1x.py --cache-dir data/openqdc
python scripts/download_sn2rxn.py --cache-dir data/openqdc
```

## Train
```bash
python scripts/train_transition1x.py --config configs/transition1x.yaml
python scripts/train_sn2rxn.py --config configs/sn2rxn.yaml
```

## Eval
```bash
python scripts/eval_transition1x.py --config configs/transition1x.yaml --checkpoint checkpoints/transition1x/best.pt --split test
python scripts/eval_sn2rxn.py --config configs/sn2rxn.yaml --checkpoint checkpoints/sn2rxn/best.pt --split test
```
