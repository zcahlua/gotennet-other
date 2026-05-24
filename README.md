# gotennet-other

Full-accuracy-oriented OpenQDC training pipeline (with smoke configs kept for quick checks).

## Smoke vs full
- Smoke/debug configs: validate code path only (`configs/*_smoke.yaml`, small epochs/samples).
- Full configs: accuracy-oriented training (`configs/transition1x_full.yaml`, `configs/sn2rxn_full.yaml`).

## Train / Eval
```bash
python scripts/train_transition1x.py --config configs/transition1x_full.yaml --mode train
python scripts/train_transition1x.py --config configs/transition1x_full.yaml --mode eval --checkpoint outputs/transition1x_full/checkpoints/best.pt --eval-split test
python scripts/train_sn2rxn.py --config configs/sn2rxn_full.yaml --mode train
python scripts/train_sn2rxn.py --config configs/sn2rxn_full.yaml --mode eval --checkpoint outputs/sn2rxn_full/checkpoints/best.pt --eval-split test
```

## Tests
```bash
python -m pytest -q
```
