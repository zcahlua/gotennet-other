def test_placeholder_smoke_config_exists():
    import pathlib
    assert pathlib.Path('configs/transition1x_smoke.yaml').exists()
