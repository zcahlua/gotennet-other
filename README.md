# gotennet-other

Minimal PyTorch codebase for GotenNet-style energy/force learning on OpenQDC `Transition1X` and `SN2RXN`.

## Install
```bash
pip install -e ".[dev]"
pip install -e ".[openqdc]"  # optional for real OpenQDC datasets
```

## Test (no OpenQDC required)
```bash
python -m pytest -q
```

## Download and inspect caches
```bash
# Transition1X
python scripts/download_transition1x.py --cache-dir data/openqdc
python scripts/inspect_transition1x.py --cache-dir data/openqdc

# SN2RXN
python scripts/download_transition1x.py --dataset-name SN2RXN --cache-dir data/openqdc
python scripts/inspect_sn2rxn.py --cache-dir data/openqdc
```

## Train
```bash
# synthetic debug run (no OpenQDC needed)
python scripts/train_transition1x.py --config configs/transition1x_debug_synthetic.yaml

# synthetic debug run for SN2RXN (no OpenQDC needed)
python scripts/train_transition1x.py --config configs/sn2rxn_debug_synthetic.yaml

# real smoke run (requires cache present)
python scripts/train_transition1x.py --config configs/transition1x_smoke.yaml

# real SN2RXN smoke run (requires cache present)
python scripts/train_transition1x.py --config configs/sn2rxn_smoke.yaml
```

If the cache path does not exist, training fails with `FileNotFoundError` before importing OpenQDC.
