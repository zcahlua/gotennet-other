# gotennet-other

Local, accuracy-oriented OpenQDC training pipeline with a GotenNet-style fallback model implementation.

## Caveats
- This repository provides a local fallback implementation and training pipeline; it is **not** an official GotenNet release unless/until the official implementation is added.
- Transition1X and SN2RXN are large datasets; expect substantial download/cache size.
- Full training requires a CUDA-capable GPU and enough storage for dataset cache + checkpoints.
- Force training uses autograd through coordinates and can be memory-intensive.

## Environment setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pip install -e ".[openqdc]"
```

## Check environment
```bash
python --version
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"
python -c "import openqdc; print('openqdc import ok')"
openqdc --help
openqdc cache
nvidia-smi
```

## Download datasets
Using helper scripts:
```bash
python scripts/download_transition1x.py --cache-dir data/openqdc
python scripts/download_sn2rxn.py --cache-dir data/openqdc
```

Equivalent OpenQDC CLI:
```bash
openqdc download Transition1X --cache-dir data/openqdc
openqdc download SN2RXN --cache-dir data/openqdc
```

If SN2RXN name resolution fails:
```bash
openqdc download sn2_rxn --cache-dir data/openqdc
```

Optional flags:
```bash
openqdc download Transition1X --cache-dir data/openqdc --as-zarr
openqdc download SN2RXN --cache-dir data/openqdc --overwrite
```

## Inspect datasets
```bash
python scripts/inspect_transition1x.py --cache-dir data/openqdc --split train
python scripts/inspect_sn2rxn.py --cache-dir data/openqdc --split train
```

## Smoke vs full configs
- Smoke/debug configs (`configs/*_smoke.yaml`) are for code-path validation only.
- Full configs (`configs/transition1x_full.yaml`, `configs/sn2rxn_full.yaml`) are for accuracy-oriented training.
- For full runs, keep `max_samples` unset/empty so the full dataset is used.

## Full training
```bash
python scripts/train_transition1x.py --config configs/transition1x_full.yaml --mode train
python scripts/train_sn2rxn.py --config configs/sn2rxn_full.yaml --mode train
```

## Evaluate best checkpoint
```bash
python scripts/train_transition1x.py --config configs/transition1x_full.yaml --mode eval --checkpoint outputs/transition1x_full/checkpoints/best.pt --eval-split test
python scripts/train_sn2rxn.py --config configs/sn2rxn_full.yaml --mode eval --checkpoint outputs/sn2rxn_full/checkpoints/best.pt --eval-split test
```

## Tests
```bash
python -m pytest -q
```
