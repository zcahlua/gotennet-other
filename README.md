# gotennet-other

Minimal PyTorch codebase for GotenNet-style energy/force learning on OpenQDC `Transition1X`.

## Features
- Loads `Transition1X` through `from openqdc.datasets import Transition1X`.
- Variable-size batching without padding via concatenation + `batch` index.
- Joint energy + force training (`force = -dE/dR`).
- Energy/force MAE and RMSE metrics.
- Unit tests that run without downloading Transition1X.
- Optional real smoke train against cached Transition1X data.
- Clear failure if cache is missing.

## Install
```bash
pip install -e .[dev]
# optional for real dataset run
pip install openqdc
```

## Run tests
```bash
pytest
```

## Real Transition1X smoke train
Requires an existing cache:
```bash
export OPENQDC_CACHE_DIR=/path/to/openqdc/cache
pytest tests/test_smoke_transition1x.py
python train_transition1x.py --cache-dir "$OPENQDC_CACHE_DIR" --epochs 1 --batch-size 2
```

If cache is missing, loader raises:
- `FileNotFoundError: Transition1X cache directory not found ...`

