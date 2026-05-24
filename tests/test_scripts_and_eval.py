from pathlib import Path


def test_required_scripts_and_config_exist():
    for f in ['scripts/train_sn2rxn.py', 'scripts/eval_sn2rxn.py', 'configs/sn2rxn.yaml']:
        assert Path(f).exists()


def test_download_scripts_use_openqdc_cli_not_dataset_init():
    t = Path('scripts/download_transition1x.py').read_text()
    s = Path('scripts/download_sn2rxn.py').read_text()
    assert 'openqdc' in t and 'download' in t and 'subprocess.run' in t
    assert 'sys.argv' not in s and 'from scripts.download_transition1x' not in s
