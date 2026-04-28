"""Typed runtime settings for the pipeline.

Single immutable Settings object loaded once at startup from
config.toml + environment variables. Threaded through stages as a
function argument; never read from globals.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Settings:
    dpi: int = 300
    ocr_confidence_threshold: float = 0.4
    max_workers: int = 1
    page_timeout_s: int = 300
    dash_solid_run_min_px: int = 8
    attribute_search_radius_px: int = 200
    output_dir: Path = Path("output")
    cache_enabled: bool = True

    def __post_init__(self) -> None:
        if self.dpi < 72 or self.dpi > 1200:
            raise ValueError(f"dpi must be in [72, 1200], got {self.dpi}")
        if not 0.0 < self.ocr_confidence_threshold < 1.0:
            raise ValueError(
                f"ocr_confidence_threshold must be in (0, 1), got {self.ocr_confidence_threshold}"
            )
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {self.max_workers}")
        if self.page_timeout_s <= 0:
            raise ValueError(f"page_timeout_s must be > 0, got {self.page_timeout_s}")
        if self.dash_solid_run_min_px <= 0:
            raise ValueError(f"dash_solid_run_min_px must be > 0, got {self.dash_solid_run_min_px}")


_FIELD_TYPES: dict[str, type] = {
    "dpi": int,
    "ocr_confidence_threshold": float,
    "max_workers": int,
    "page_timeout_s": int,
    "dash_solid_run_min_px": int,
    "attribute_search_radius_px": int,
    "output_dir": Path,
    "cache_enabled": bool,
}


def _coerce(value: Any, field_type: type) -> Any:
    if field_type is bool:
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if field_type is Path:
        return Path(value)
    return field_type(value)


def load_settings(
    config_path: Path | None,
    env: dict[str, str] | None = None,
) -> Settings:
    """Load Settings from optional TOML file with environment overrides.

    Environment variables of the form PIPELINE_<FIELD> override TOML values.
    Both layers are optional; defaults from the dataclass apply otherwise.
    """
    if env is None:
        env = dict(os.environ)

    raw: dict[str, Any] = {}
    if config_path is not None and config_path.exists():
        with config_path.open("rb") as fh:
            raw = tomllib.load(fh)

    for field_def in fields(Settings):
        name = field_def.name
        env_key = f"PIPELINE_{name.upper()}"
        if env_key in env:
            raw[name] = env[env_key]

    coerced: dict[str, Any] = {}
    for name, value in raw.items():
        if name in _FIELD_TYPES:
            coerced[name] = _coerce(value, _FIELD_TYPES[name])

    return Settings(**coerced)
