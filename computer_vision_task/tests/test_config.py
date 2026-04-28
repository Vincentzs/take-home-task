from pathlib import Path

import pytest

from pipeline.config import Settings, load_settings


def test_settings_defaults_are_reasonable() -> None:
    s = Settings()
    assert s.dpi == 300
    assert 0.0 < s.ocr_confidence_threshold < 1.0
    assert s.max_workers >= 1
    assert s.page_timeout_s > 0
    assert s.dash_solid_run_min_px > 0


def test_load_settings_uses_defaults_when_no_file(tmp_path: Path) -> None:
    s = load_settings(config_path=None, env={})
    assert s.dpi == 300


def test_load_settings_reads_toml(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text("dpi = 400\nmax_workers = 2\n")
    s = load_settings(config_path=cfg, env={})
    assert s.dpi == 400
    assert s.max_workers == 2


def test_load_settings_env_overrides_toml(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text("dpi = 400\n")
    s = load_settings(config_path=cfg, env={"PIPELINE_DPI": "200"})
    assert s.dpi == 200


def test_settings_validates_dpi_range() -> None:
    with pytest.raises(ValueError, match="dpi"):
        Settings(dpi=50)


def test_settings_validates_confidence_range() -> None:
    with pytest.raises(ValueError, match="ocr_confidence_threshold"):
        Settings(ocr_confidence_threshold=1.5)
