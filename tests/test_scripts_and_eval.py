from pathlib import Path
import subprocess, sys

def test_required_scripts_and_config_exist():
    for f in ['scripts/train_sn2rxn.py', 'configs/sn2rxn.yaml']:
        assert Path(f).exists()
    assert not Path('scripts/eval_transition1x.py').exists()

def test_cli_help_commands():
    for cmd in [
        [sys.executable,'scripts/download_transition1x.py','--help'],
        [sys.executable,'scripts/download_sn2rxn.py','--help'],
        [sys.executable,'scripts/inspect_transition1x.py','--help'],
        [sys.executable,'scripts/inspect_sn2rxn.py','--help'],
        [sys.executable,'scripts/train_transition1x.py','--help'],
        [sys.executable,'scripts/train_sn2rxn.py','--help'],
    ]:
        assert subprocess.run(cmd, capture_output=True).returncode==0
