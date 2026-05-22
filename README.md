# gotennet-other

Minimal PyTorch codebase for GotenNet-style energy/force learning on OpenQDC `Transition1X`.

## Install
```bash
pip install -e ".[dev]"
pip install -e ".[openqdc]"  # optional for real Transition1X
```

## Test (no OpenQDC required)
```bash
python -m pytest -q
```

## Download and inspect Transition1X cache
```bash
python scripts/download_transition1x.py --cache-dir data/openqdc
python scripts/inspect_transition1x.py --cache-dir data/openqdc
```

## Train
```bash
# synthetic debug run (no OpenQDC needed)
python scripts/train_transition1x.py --config configs/transition1x_debug_synthetic.yaml

# real smoke run (requires cache present)
python scripts/train_transition1x.py --config configs/transition1x_smoke.yaml
```

If the cache path does not exist, training fails with `FileNotFoundError` before importing OpenQDC.
