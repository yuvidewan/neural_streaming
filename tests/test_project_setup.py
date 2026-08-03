"""Sanity checks for the repository foundation.

These do not test compression, model, or streaming behavior (none of that
exists yet) - only that the package structure and configuration mechanism
are importable and internally consistent.
"""

from pathlib import Path

from nvc.utils.config import Config, load_default_config


def test_subpackages_import() -> None:
    import nvc
    import nvc.compression
    import nvc.data
    import nvc.evaluation
    import nvc.models
    import nvc.utils

    assert nvc.__version__


def test_config_defaults_have_expected_types() -> None:
    config = Config()

    assert isinstance(config.frame_width, int)
    assert isinstance(config.frame_height, int)
    assert isinstance(config.latent_dim, int)
    assert isinstance(config.batch_size, int)
    assert isinstance(config.learning_rate, float)
    assert isinstance(config.epochs, int)
    assert isinstance(config.raw_data_dir, Path)
    assert isinstance(config.checkpoint_dir, Path)


def test_load_default_config_reads_json_overrides() -> None:
    config = load_default_config()

    # configs/default.json currently sets these two explicitly.
    assert config.frame_width == 256
    assert config.batch_size == 8


def test_config_from_json_rejects_unknown_keys(tmp_path: Path) -> None:
    bad_config_path = tmp_path / "bad_config.json"
    bad_config_path.write_text('{"not_a_real_field": 1}', encoding="utf-8")

    try:
        Config.from_json(bad_config_path)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for unknown config key")
