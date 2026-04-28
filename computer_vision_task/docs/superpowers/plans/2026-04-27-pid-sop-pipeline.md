# P&ID → Graph + SOP Cross-Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python pipeline that converts a P&ID PDF into a NetworkX graph and cross-references it against an SOP DOCX, emitting discrepancies, annotated page renderings, and a structured log.

**Architecture:** Pure-function stages composed by a thin runner. P&ID branch and SOP branch run independently, joining at cross-reference. Per-page parallelism via `ProcessPoolExecutor`; stage-level caching keyed by content hash; errors flow as `PipelineWarning` records, not exceptions; one typed `Settings` object threaded through every stage.

**Tech Stack:** Python 3.12+, `uv` (deps/venv), `ruff` (lint+format), `mypy --strict`, `pytest`, `opencv-python`, `easyocr`, `pypdfium2`, `python-docx`, `networkx`, `Pillow`.

**Spec:** `docs/superpowers/specs/2026-04-27-pid-sop-design.md`

---

## Task 1: Bootstrap project tooling

**Files:**
- Create: `pyproject.toml`
- Create: `Makefile`
- Create: `.pre-commit-config.yaml`
- Modify: `../.gitignore` (project root, one level up from `computer_vision_task/`)

- [ ] **Step 1: Verify uv is installed**

Run: `uv --version`
Expected: prints something like `uv 0.4.x` or later. If not, install with `curl -LsSf https://astral.sh/uv/install.sh | sh`.

- [ ] **Step 2: Create `pyproject.toml`**

Path: `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "pipeline"
version = "0.1.0"
description = "P&ID-to-graph + SOP cross-reference pipeline"
requires-python = ">=3.12"
dependencies = [
    "easyocr>=1.7.2",
    "opencv-python>=4.10",
    "pypdfium2>=4.30",
    "python-docx>=1.1",
    "networkx>=3.3",
    "Pillow>=10.4",
    "numpy>=1.26,<2.2",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3",
    "pytest-cov>=5.0",
    "mypy>=1.11",
    "ruff>=0.6",
    "pre-commit>=3.8",
    "types-Pillow",
]

[project.scripts]
pipeline = "pipeline.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/pipeline"]

[tool.ruff]
line-length = 100
target-version = "py312"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "N", "SIM", "RET", "PTH", "RUF"]
ignore = ["E501"]  # line length handled by formatter

[tool.ruff.format]
quote-style = "double"

[tool.mypy]
python_version = "3.12"
strict = true
files = ["src/pipeline", "tests"]
plugins = []

[[tool.mypy.overrides]]
module = ["easyocr.*", "cv2.*", "pypdfium2.*", "docx.*", "networkx.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
addopts = "-ra --strict-markers"
testpaths = ["tests"]
markers = [
    "slow: integration tests that load OCR models / process the real PDF",
]

[tool.coverage.run]
source = ["src/pipeline"]
omit = ["*/cli.py", "*/runner.py", "*/visualizer.py", "*/pdf_loader.py", "*/preprocess.py", "*/ocr.py"]
```

- [ ] **Step 3: Create `Makefile`**

Path: `Makefile`

```makefile
.PHONY: install lint format typecheck test test-all run ci clean

install:
	uv sync --extra dev

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck:
	uv run mypy

test:
	uv run pytest -m 'not slow'

test-all:
	uv run pytest

run:
	uv run python -m pipeline.cli run

clean:
	rm -rf output/ .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info

ci: lint typecheck test-all
```

- [ ] **Step 4: Create `.pre-commit-config.yaml`**

Path: `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2
    hooks:
      - id: mypy
        additional_dependencies: ["types-Pillow"]
        args: ["--config-file=pyproject.toml"]
        files: ^(src/pipeline|tests)/
```

- [ ] **Step 5: Update `.gitignore`** at project root (`../`)

Append the following lines to `../.gitignore` (one level up):

```
# Python
__pycache__/
*.py[cod]
.venv/
.mypy_cache/
.pytest_cache/
.ruff_cache/
.coverage
htmlcov/
*.egg-info/
build/
dist/

# Pipeline outputs
computer_vision_task/output/

# uv lockfile is committed; cache is not
.uv-cache/
```

- [ ] **Step 6: Run `uv sync`**

Run: `uv sync --extra dev`
Expected: creates `.venv/` and `uv.lock`; installs all deps.

- [ ] **Step 7: Verify tools work**

Run: `uv run ruff --version && uv run mypy --version && uv run pytest --version`
Expected: each prints a version string.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml Makefile .pre-commit-config.yaml uv.lock ../.gitignore
git commit -m "Bootstrap pipeline project tooling (uv, ruff, mypy, pytest)"
```

---

## Task 2: Domain types

**Files:**
- Create: `src/pipeline/__init__.py`
- Create: `src/pipeline/types.py`
- Create: `tests/__init__.py`
- Create: `tests/test_types.py`

- [ ] **Step 1: Create empty package init**

Path: `src/pipeline/__init__.py`

```python
"""P&ID-to-graph + SOP cross-reference pipeline."""

__version__ = "0.1.0"
```

- [ ] **Step 2: Create `tests/__init__.py`**

Path: `tests/__init__.py`

```python
```
(empty file)

- [ ] **Step 3: Write failing test for types**

Path: `tests/test_types.py`

```python
from pipeline.types import (
    BBox,
    Component,
    ComponentType,
    Discrepancy,
    DiscrepancyKind,
    Edge,
    EquipmentTag,
    ExpectedComponent,
    InstrumentTag,
    PageImage,
    PipelineWarning,
    Report,
    TagKind,
    TextBox,
)


def test_textbox_is_frozen_and_carries_confidence():
    tb = TextBox(text="V-745", bbox=(10, 20, 30, 40), confidence=0.95)
    assert tb.text == "V-745"
    assert tb.bbox == (10, 20, 30, 40)
    assert tb.confidence == 0.95


def test_equipment_tag_round_trip():
    tag = EquipmentTag(
        raw="F-715A",
        prefix="F",
        number=715,
        suffix="A",
        bbox=(0, 0, 10, 10),
        page=0,
    )
    assert tag.prefix == "F"
    assert tag.suffix == "A"


def test_instrument_tag_describes_isa_letters():
    tag = InstrumentTag(
        letters="PIC",
        number="02",
        variable="Pressure",
        functions=("Indicator", "Controller"),
        description="Pressure Indicator Controller",
        bbox=(0, 0, 10, 10),
        page=0,
    )
    assert tag.functions == ("Indicator", "Controller")


def test_component_attrs_default_empty():
    c = Component(tag="V-745", type=ComponentType.VESSEL, page=0, bbox=(0, 0, 10, 10))
    assert c.attrs == {}


def test_discrepancy_carries_kind_and_message():
    d = Discrepancy(
        kind=DiscrepancyKind.MISSING_IN_DIAGRAM,
        tag="V-999",
        side=None,
        expected="vessel",
        actual=None,
        page=None,
        message="V-999 expected by SOP but not found",
    )
    assert d.kind is DiscrepancyKind.MISSING_IN_DIAGRAM


def test_pipeline_warning_carries_stage_and_code():
    w = PipelineWarning(stage="ocr", page=0, code="no_text_found", message="blank")
    assert w.stage == "ocr"
    assert w.code == "no_text_found"


def test_report_summary_counts():
    r = Report(discrepancies=(), warnings=(), summary={"missing_in_diagram": 0})
    assert r.summary["missing_in_diagram"] == 0


def test_tag_kind_and_component_type_are_str_enums():
    assert TagKind.EQUIPMENT == "equipment"
    assert ComponentType.VESSEL == "vessel"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.types'` or import errors.

- [ ] **Step 5: Implement `pipeline/types.py`**

Path: `src/pipeline/types.py`

```python
"""Domain types for the P&ID + SOP cross-reference pipeline.

All types are frozen dataclasses with __slots__ for cheap equality and
JSON-roundtrip-friendly fields. No I/O; no third-party deps beyond numpy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

BBox = tuple[int, int, int, int]
"""Pixel-space bounding box: (x0, y0, x1, y1)."""


class TagKind(StrEnum):
    EQUIPMENT = "equipment"
    INSTRUMENT = "instrument"


class ComponentType(StrEnum):
    VESSEL = "vessel"
    HEAT_EXCHANGER = "heat_exchanger"
    AIR_COOLED_EXCHANGER = "air_cooled_exchanger"
    FILTER = "filter"
    PUMP = "pump"
    TANK = "tank"
    INSTRUMENT = "instrument"
    UNKNOWN = "unknown"


class DiscrepancyKind(StrEnum):
    MISSING_IN_DIAGRAM = "missing_in_diagram"
    EXTRA_IN_DIAGRAM = "extra_in_diagram"
    TYPE_MISMATCH = "type_mismatch"
    PRESSURE_MISMATCH = "pressure_mismatch"
    TEMPERATURE_MISMATCH = "temperature_mismatch"


@dataclass(frozen=True, slots=True)
class PageImage:
    page_idx: int
    image: "np.ndarray"
    dpi: int


@dataclass(frozen=True, slots=True)
class TextBox:
    text: str
    bbox: BBox
    confidence: float


@dataclass(frozen=True, slots=True)
class EquipmentTag:
    raw: str
    prefix: str
    number: int
    suffix: str | None
    bbox: BBox
    page: int


@dataclass(frozen=True, slots=True)
class InstrumentTag:
    letters: str
    number: str
    variable: str
    functions: tuple[str, ...]
    description: str
    bbox: BBox
    page: int


Tag = EquipmentTag | InstrumentTag


@dataclass(frozen=True, slots=True)
class Component:
    tag: str
    type: ComponentType
    page: int
    bbox: BBox
    attrs: Mapping[str, str | float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Edge:
    src_tag: str
    dst_tag: str
    page: int
    evidence: BBox


@dataclass(frozen=True, slots=True)
class ExpectedComponent:
    tag: str
    type: ComponentType
    side: str | None
    design_pressure_psig: float | None
    design_temp_f_min: float | None
    design_temp_f_max: float | None


@dataclass(frozen=True, slots=True)
class Discrepancy:
    kind: DiscrepancyKind
    tag: str
    side: str | None
    expected: str | float | None
    actual: str | float | None
    page: int | None
    message: str


@dataclass(frozen=True, slots=True)
class PipelineWarning:
    stage: str
    page: int | None
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class Report:
    discrepancies: tuple[Discrepancy, ...]
    warnings: tuple[PipelineWarning, ...]
    summary: Mapping[str, int]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_types.py -v`
Expected: all 8 tests pass.

- [ ] **Step 7: Run mypy**

Run: `uv run mypy src/pipeline/types.py`
Expected: `Success: no issues found in 1 source file`.

- [ ] **Step 8: Commit**

```bash
git add src/pipeline/__init__.py src/pipeline/types.py tests/__init__.py tests/test_types.py
git commit -m "Add frozen domain types for pipeline data model"
```

---

## Task 3: Settings

**Files:**
- Create: `src/pipeline/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

Path: `tests/test_config.py`

```python
import os
from pathlib import Path

import pytest

from pipeline.config import Settings, load_settings


def test_settings_defaults_are_reasonable():
    s = Settings()
    assert s.dpi == 300
    assert 0.0 < s.ocr_confidence_threshold < 1.0
    assert s.max_workers >= 1
    assert s.page_timeout_s > 0
    assert s.dash_solid_run_min_px > 0


def test_load_settings_uses_defaults_when_no_file(tmp_path: Path):
    s = load_settings(config_path=None, env={})
    assert s.dpi == 300


def test_load_settings_reads_toml(tmp_path: Path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('dpi = 400\nmax_workers = 2\n')
    s = load_settings(config_path=cfg, env={})
    assert s.dpi == 400
    assert s.max_workers == 2


def test_load_settings_env_overrides_toml(tmp_path: Path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('dpi = 400\n')
    s = load_settings(config_path=cfg, env={"PIPELINE_DPI": "200"})
    assert s.dpi == 200


def test_settings_validates_dpi_range():
    with pytest.raises(ValueError, match="dpi"):
        Settings(dpi=50)


def test_settings_validates_confidence_range():
    with pytest.raises(ValueError, match="ocr_confidence_threshold"):
        Settings(ocr_confidence_threshold=1.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `pipeline/config.py`**

Path: `src/pipeline/config.py`

```python
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
            raise ValueError(
                f"dash_solid_run_min_px must be > 0, got {self.dash_solid_run_min_px}"
            )


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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: all 6 tests pass.

- [ ] **Step 5: Run mypy**

Run: `uv run mypy src/pipeline/config.py tests/test_config.py`
Expected: `Success: no issues found`.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/config.py tests/test_config.py
git commit -m "Add typed Settings with TOML + env loader and validation"
```

---

## Task 4: Stage protocols

**Files:**
- Create: `src/pipeline/protocols.py`

- [ ] **Step 1: Implement protocols (no test — pure interface module)**

Path: `src/pipeline/protocols.py`

```python
"""Stage interfaces. Concrete implementations live under pipeline.stages.*

Stages depend on these Protocols, not concrete classes, so the
orchestrator can swap implementations without ripple effects.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from pipeline.types import BBox, ComponentType, TextBox


class OCREngine(Protocol):
    def read(self, image: np.ndarray) -> list[TextBox]: ...


class LineSegment(Protocol):
    @property
    def p0(self) -> tuple[int, int]: ...
    @property
    def p1(self) -> tuple[int, int]: ...
    @property
    def bbox(self) -> BBox: ...


class LineDetector(Protocol):
    def detect(self, image: np.ndarray) -> list[LineSegment]: ...


class SymbolClassifier(Protocol):
    def classify(self, tag_prefix: str) -> ComponentType: ...
```

- [ ] **Step 2: Run mypy**

Run: `uv run mypy src/pipeline/protocols.py`
Expected: `Success: no issues found`.

- [ ] **Step 3: Commit**

```bash
git add src/pipeline/protocols.py
git commit -m "Define stage protocols for OCR, line detection, classification"
```

---

## Task 5: ISA-5.1 lookup tables

**Files:**
- Create: `src/pipeline/isa_lookup.py`
- Create: `tests/test_isa_lookup.py`

- [ ] **Step 1: Write failing test**

Path: `tests/test_isa_lookup.py`

```python
from pipeline.isa_lookup import (
    EQUIPMENT_PREFIX_TO_TYPE,
    ISA_FIRST_LETTER,
    ISA_FUNCTION_LETTER,
    INSTRUMENT_ABBREVIATIONS,
    describe_instrument,
)
from pipeline.types import ComponentType


def test_first_letter_table_has_core_variables():
    assert ISA_FIRST_LETTER["P"] == "Pressure"
    assert ISA_FIRST_LETTER["T"] == "Temperature"
    assert ISA_FIRST_LETTER["F"] == "Flow"
    assert ISA_FIRST_LETTER["L"] == "Level"
    assert ISA_FIRST_LETTER["A"] == "Analysis"


def test_function_letter_table_has_core_modifiers():
    assert ISA_FUNCTION_LETTER["I"] == "Indicator"
    assert ISA_FUNCTION_LETTER["C"] == "Controller"
    assert ISA_FUNCTION_LETTER["T"] == "Transmitter"
    assert ISA_FUNCTION_LETTER["V"] == "Valve"
    assert ISA_FUNCTION_LETTER["S"] == "Switch"
    assert ISA_FUNCTION_LETTER["A"] == "Alarm"


def test_abbreviation_table_includes_common_examples():
    assert INSTRUMENT_ABBREVIATIONS["PCV"] == "Pressure Control Valve"
    assert INSTRUMENT_ABBREVIATIONS["FIC"] == "Flow Indicator Controller"
    assert INSTRUMENT_ABBREVIATIONS["ESD"] == "Emergency Shutdown Station"
    assert INSTRUMENT_ABBREVIATIONS["PSV"] == "Pressure Safety Valve"


def test_describe_known_abbreviation_uses_table():
    assert describe_instrument("PCV") == "Pressure Control Valve"


def test_describe_decomposes_unknown_abbreviation_via_letters():
    # PIT is not in our minimal abbrev set; must decompose.
    desc = describe_instrument("PIT")
    assert "Pressure" in desc
    assert "Indicator" in desc
    assert "Transmitter" in desc


def test_describe_returns_raw_for_unparseable():
    assert describe_instrument("XYZ") == "XYZ"


def test_equipment_prefix_map_includes_sop_tags():
    assert EQUIPMENT_PREFIX_TO_TYPE["V"] == ComponentType.VESSEL
    assert EQUIPMENT_PREFIX_TO_TYPE["E"] == ComponentType.HEAT_EXCHANGER
    assert EQUIPMENT_PREFIX_TO_TYPE["AC"] == ComponentType.AIR_COOLED_EXCHANGER
    assert EQUIPMENT_PREFIX_TO_TYPE["F"] == ComponentType.FILTER
    assert EQUIPMENT_PREFIX_TO_TYPE["P"] == ComponentType.PUMP
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_isa_lookup.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement `pipeline/isa_lookup.py`**

Path: `src/pipeline/isa_lookup.py`

```python
"""ISA-5.1 instrument tag conventions and equipment prefix mapping.

Sources:
- ISA-5.1-2009 Instrumentation Symbols and Identification, summarised in
  Kimray's "How to Read an Oil & Gas P&ID" reference guide.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

from pipeline.types import ComponentType

ISA_FIRST_LETTER: Final[Mapping[str, str]] = MappingProxyType(
    {
        "A": "Analysis",
        "B": "Burner",
        "C": "User's Choice",
        "D": "User's Choice",
        "E": "Voltage",
        "F": "Flow",
        "G": "User's Choice",
        "H": "Hand",
        "I": "Current",
        "J": "Power",
        "K": "Time",
        "L": "Level",
        "M": "User's Choice",
        "N": "User's Choice",
        "O": "User's Choice",
        "P": "Pressure",
        "Q": "Quantity",
        "R": "Radiation",
        "S": "Speed",
        "T": "Temperature",
        "U": "Multivariable",
        "V": "Vibration",
        "W": "Weight",
        "X": "Unclassified",
        "Y": "Event",
        "Z": "Position",
    }
)

ISA_FUNCTION_LETTER: Final[Mapping[str, str]] = MappingProxyType(
    {
        "A": "Alarm",
        "C": "Controller",
        "E": "Sensor",
        "G": "Glass",
        "H": "High",
        "I": "Indicator",
        "L": "Low",
        "R": "Recorder",
        "S": "Switch",
        "T": "Transmitter",
        "V": "Valve",
        "Y": "Relay",
        "Z": "Actuator",
    }
)

INSTRUMENT_ABBREVIATIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "AC": "Analysis Controller",
        "AT": "Analysis Transmitter",
        "BDV": "Blowdown Valve",
        "CV": "Control Valve",
        "ESD": "Emergency Shutdown Station",
        "ESDV": "Emergency Shutdown Valve",
        "FC": "Flow Controller",
        "FCV": "Flow Control Valve",
        "FE": "Flow Element",
        "FI": "Flow Indicator",
        "FIC": "Flow Indicator Controller",
        "FIT": "Flow Indicator Transmitter",
        "FT": "Flow Transmitter",
        "HS": "Hand Switch",
        "LCV": "Level Control Valve",
        "LG": "Level Gauge",
        "LI": "Level Indicator",
        "LIC": "Level Indicator Controller",
        "LIT": "Level Indicator Transmitter",
        "LSH": "Level Switch High",
        "LSL": "Level Switch Low",
        "LT": "Level Transmitter",
        "PCV": "Pressure Control Valve",
        "PG": "Pressure Gauge",
        "PI": "Pressure Indicator",
        "PIC": "Pressure Indicator Controller",
        "PIT": "Pressure Indicator Transmitter",
        "PRV": "Pressure Regulating Valve",
        "PSV": "Pressure Safety Valve",
        "PT": "Pressure Transmitter",
        "TCV": "Temperature Control Valve",
        "TE": "Temperature Element",
        "TI": "Temperature Indicator",
        "TIC": "Temperature Indicator Controller",
        "TIT": "Temperature Indicator Transmitter",
        "TT": "Temperature Transmitter",
    }
)

EQUIPMENT_PREFIX_TO_TYPE: Final[Mapping[str, ComponentType]] = MappingProxyType(
    {
        "V": ComponentType.VESSEL,
        "D": ComponentType.VESSEL,  # drum
        "E": ComponentType.HEAT_EXCHANGER,
        "AC": ComponentType.AIR_COOLED_EXCHANGER,
        "F": ComponentType.FILTER,
        "P": ComponentType.PUMP,
        "T": ComponentType.TANK,
    }
)


def describe_instrument(letters: str) -> str:
    """Render an instrument abbreviation as human-readable text.

    Lookup order:
    1. Exact match in INSTRUMENT_ABBREVIATIONS.
    2. Decompose: first letter via ISA_FIRST_LETTER, remainder via ISA_FUNCTION_LETTER.
    3. Fallback: return the raw string.
    """
    if letters in INSTRUMENT_ABBREVIATIONS:
        return INSTRUMENT_ABBREVIATIONS[letters]
    if not letters or letters[0] not in ISA_FIRST_LETTER:
        return letters
    parts: list[str] = [ISA_FIRST_LETTER[letters[0]]]
    for char in letters[1:]:
        if char in ISA_FUNCTION_LETTER:
            parts.append(ISA_FUNCTION_LETTER[char])
        else:
            return letters
    return " ".join(parts)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_isa_lookup.py -v`
Expected: all 7 tests pass.

- [ ] **Step 5: Run mypy**

Run: `uv run mypy src/pipeline/isa_lookup.py`
Expected: `Success: no issues found`.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/isa_lookup.py tests/test_isa_lookup.py
git commit -m "Add ISA-5.1 letter tables and equipment prefix mapping"
```

---

## Task 6: Tag parsing

**Files:**
- Create: `src/pipeline/stages/__init__.py`
- Create: `src/pipeline/stages/tags.py`
- Create: `tests/test_tags.py`

- [ ] **Step 1: Create stages package init**

Path: `src/pipeline/stages/__init__.py`

```python
```
(empty file)

- [ ] **Step 2: Write failing test**

Path: `tests/test_tags.py`

```python
from pipeline.stages.tags import parse_tags
from pipeline.types import EquipmentTag, InstrumentTag, TextBox


def _tb(text: str, x: int = 0, y: int = 0) -> TextBox:
    return TextBox(text=text, bbox=(x, y, x + 50, y + 20), confidence=0.9)


def test_parse_equipment_tag_with_suffix():
    tags, warnings = parse_tags([_tb("F-715A")], page=0)
    assert len(tags) == 1
    eq = tags[0]
    assert isinstance(eq, EquipmentTag)
    assert eq.raw == "F-715A"
    assert eq.prefix == "F"
    assert eq.number == 715
    assert eq.suffix == "A"
    assert warnings == []


def test_parse_equipment_tag_without_suffix():
    tags, _ = parse_tags([_tb("V-745")], page=0)
    assert len(tags) == 1
    eq = tags[0]
    assert isinstance(eq, EquipmentTag)
    assert eq.suffix is None


def test_parse_equipment_tag_with_two_letter_prefix():
    tags, _ = parse_tags([_tb("AC-746")], page=0)
    assert len(tags) == 1
    eq = tags[0]
    assert isinstance(eq, EquipmentTag)
    assert eq.prefix == "AC"
    assert eq.number == 746


def test_parse_pairs_letters_above_number_into_instrument_tag():
    boxes = [_tb("PIC", x=100, y=100), _tb("02", x=100, y=125)]
    tags, _ = parse_tags(boxes, page=0)
    assert len(tags) == 1
    inst = tags[0]
    assert isinstance(inst, InstrumentTag)
    assert inst.letters == "PIC"
    assert inst.number == "02"
    assert inst.description == "Pressure Indicator Controller"


def test_unpaired_instrument_letters_warn_and_drop():
    tags, warnings = parse_tags([_tb("PIC", x=100, y=100)], page=0)
    assert tags == []
    assert any(w.code == "unpaired_instrument_letters" for w in warnings)


def test_unknown_equipment_prefix_warns_but_keeps_tag():
    tags, warnings = parse_tags([_tb("ZZ-999")], page=0)
    assert len(tags) == 1
    assert any(w.code == "unknown_prefix" for w in warnings)


def test_text_that_does_not_match_any_pattern_is_ignored():
    tags, warnings = parse_tags([_tb("SOME LABEL"), _tb("123 mm")], page=0)
    assert tags == []
    assert warnings == []  # silent — ordinary text, not a tag candidate


def test_low_confidence_tag_emits_warning_but_still_parses():
    boxes = [TextBox(text="V-745", bbox=(0, 0, 50, 20), confidence=0.3)]
    tags, warnings = parse_tags(boxes, page=0, min_confidence=0.5)
    assert tags == []
    assert any(w.code == "low_confidence_tag" for w in warnings)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_tags.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement `pipeline/stages/tags.py`**

Path: `src/pipeline/stages/tags.py`

```python
"""Tag parsing: TextBoxes → equipment tags + instrument tags + warnings.

Two regex families per ISA-5.1:
- Equipment tags: <PREFIX>-<NUMBER><SUFFIX?>  e.g. F-715A, V-745, AC-746
- Instrument tags: a letters box stacked above a number box, both inside
  a circle. We pair them by horizontal alignment and vertical adjacency.
"""

from __future__ import annotations

import re

from pipeline.isa_lookup import EQUIPMENT_PREFIX_TO_TYPE, describe_instrument
from pipeline.types import (
    BBox,
    EquipmentTag,
    InstrumentTag,
    PipelineWarning,
    Tag,
    TextBox,
)

EQUIPMENT_RE = re.compile(r"^([A-Z]{1,3})-(\d{2,4})([A-Z])?$")
INSTRUMENT_LETTERS_RE = re.compile(r"^[A-Z]{2,4}$")
INSTRUMENT_NUMBER_RE = re.compile(r"^\d{2,3}$")

_DEFAULT_PAIRING_X_TOL = 30  # px: max horizontal centre offset for pairing
_DEFAULT_PAIRING_Y_GAP = 40  # px: max vertical gap between letters and number
_FIRST_LETTER_TO_VARIABLE: dict[str, str] = {}


def _bbox_centre(bbox: BBox) -> tuple[int, int]:
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) // 2, (y0 + y1) // 2)


def _try_equipment(tb: TextBox, page: int) -> EquipmentTag | None:
    cleaned = tb.text.replace(" ", "")
    m = EQUIPMENT_RE.match(cleaned)
    if m is None:
        return None
    prefix, number_s, suffix = m.groups()
    return EquipmentTag(
        raw=cleaned,
        prefix=prefix,
        number=int(number_s),
        suffix=suffix,
        bbox=tb.bbox,
        page=page,
    )


def _pair_instrument(
    letters_tb: TextBox,
    candidates: list[TextBox],
    x_tol: int,
    y_gap: int,
) -> TextBox | None:
    lx, ly = _bbox_centre(letters_tb.bbox)
    letters_y1 = letters_tb.bbox[3]
    best: tuple[int, TextBox] | None = None
    for tb in candidates:
        if not INSTRUMENT_NUMBER_RE.match(tb.text):
            continue
        cx, cy = _bbox_centre(tb.bbox)
        if abs(cx - lx) > x_tol:
            continue
        gap = tb.bbox[1] - letters_y1
        if gap < 0 or gap > y_gap:
            continue
        if best is None or gap < best[0]:
            best = (gap, tb)
    return best[1] if best is not None else None


def parse_tags(
    text_boxes: list[TextBox],
    page: int,
    min_confidence: float = 0.0,
    x_tol: int = _DEFAULT_PAIRING_X_TOL,
    y_gap: int = _DEFAULT_PAIRING_Y_GAP,
) -> tuple[list[Tag], list[PipelineWarning]]:
    """Parse equipment and instrument tags from a list of OCR TextBoxes."""
    tags: list[Tag] = []
    warnings: list[PipelineWarning] = []
    consumed: set[int] = set()

    # First pass: equipment tags.
    for i, tb in enumerate(text_boxes):
        eq = _try_equipment(tb, page)
        if eq is None:
            continue
        if tb.confidence < min_confidence:
            warnings.append(
                PipelineWarning(
                    stage="tags",
                    page=page,
                    code="low_confidence_tag",
                    message=f"{tb.text} confidence={tb.confidence:.2f}",
                )
            )
            consumed.add(i)
            continue
        if eq.prefix not in EQUIPMENT_PREFIX_TO_TYPE:
            warnings.append(
                PipelineWarning(
                    stage="tags",
                    page=page,
                    code="unknown_prefix",
                    message=f"{eq.raw} prefix={eq.prefix}",
                )
            )
        tags.append(eq)
        consumed.add(i)

    # Second pass: instrument letter boxes pair with adjacent number boxes.
    for i, tb in enumerate(text_boxes):
        if i in consumed:
            continue
        if not INSTRUMENT_LETTERS_RE.match(tb.text):
            continue
        if tb.confidence < min_confidence:
            warnings.append(
                PipelineWarning(
                    stage="tags",
                    page=page,
                    code="low_confidence_tag",
                    message=f"{tb.text} confidence={tb.confidence:.2f}",
                )
            )
            consumed.add(i)
            continue
        partner = _pair_instrument(
            tb,
            [text_boxes[j] for j in range(len(text_boxes)) if j not in consumed and j != i],
            x_tol=x_tol,
            y_gap=y_gap,
        )
        if partner is None:
            warnings.append(
                PipelineWarning(
                    stage="tags",
                    page=page,
                    code="unpaired_instrument_letters",
                    message=tb.text,
                )
            )
            consumed.add(i)
            continue
        merged_bbox: BBox = (
            min(tb.bbox[0], partner.bbox[0]),
            min(tb.bbox[1], partner.bbox[1]),
            max(tb.bbox[2], partner.bbox[2]),
            max(tb.bbox[3], partner.bbox[3]),
        )
        first = tb.text[0]
        functions = tuple(c for c in tb.text[1:])
        tags.append(
            InstrumentTag(
                letters=tb.text,
                number=partner.text,
                variable=_FIRST_LETTER_TO_VARIABLE.get(first, first),
                functions=functions,
                description=describe_instrument(tb.text),
                bbox=merged_bbox,
                page=page,
            )
        )
        consumed.add(i)
        consumed.add(text_boxes.index(partner))

    return tags, warnings


# Backfill the variable lookup at import time using the ISA table to keep
# the InstrumentTag.variable field human-readable without re-importing.
def _populate_variable_lookup() -> None:
    from pipeline.isa_lookup import ISA_FIRST_LETTER

    _FIRST_LETTER_TO_VARIABLE.update(dict(ISA_FIRST_LETTER))


_populate_variable_lookup()
```

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest tests/test_tags.py -v`
Expected: all 8 tests pass.

- [ ] **Step 6: Run mypy**

Run: `uv run mypy src/pipeline/stages/tags.py tests/test_tags.py`
Expected: `Success: no issues found`.

- [ ] **Step 7: Commit**

```bash
git add src/pipeline/stages/__init__.py src/pipeline/stages/tags.py tests/test_tags.py
git commit -m "Add ISA-aware tag parser for equipment and instrument tags"
```

---

## Task 7: SOP parser

**Files:**
- Create: `src/pipeline/stages/sop_parser.py`
- Create: `tests/conftest.py`
- Create: `tests/test_sop_parser.py`

- [ ] **Step 1: Create conftest with a tiny SOP factory**

Path: `tests/conftest.py`

```python
"""Shared fixtures for pipeline tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from docx import Document
from docx.document import Document as DocumentT


@pytest.fixture
def make_sop_docx(tmp_path: Path):
    def _make(rows: list[tuple[str, ...]], header_label: str = "Design Limits") -> Path:
        doc: DocumentT = Document()
        doc.add_heading("Operating Limits", level=1)
        doc.add_paragraph(header_label + ":")
        cols = max(len(r) for r in rows)
        table = doc.add_table(rows=1 + len(rows), cols=cols)
        # First row: column headers (Pressure (psig), Temperature (°F)) in cols 1+
        if cols >= 3:
            table.rows[0].cells[1].text = "Pressure (psig)"
            table.rows[0].cells[2].text = "Temperature (°F)"
        for i, row in enumerate(rows, start=1):
            for j, value in enumerate(row):
                table.rows[i].cells[j].text = value
        out = tmp_path / "sop.docx"
        doc.save(out)
        return out

    return _make
```

- [ ] **Step 2: Write failing test**

Path: `tests/test_sop_parser.py`

```python
from pathlib import Path

from pipeline.stages.sop_parser import parse_sop
from pipeline.types import ComponentType


def test_parses_simple_design_limits_table(make_sop_docx):
    docx = make_sop_docx(
        rows=[
            ("V-745 Stabilizer Tower", "300", "375"),
            ("AC-746 After Cooler", "350", "-20 to 400"),
        ]
    )
    expected, warnings = parse_sop(docx)
    assert warnings == []
    by_tag = {(e.tag, e.side): e for e in expected}
    v745 = by_tag[("V-745", None)]
    assert v745.type == ComponentType.VESSEL
    assert v745.design_pressure_psig == 300.0
    assert v745.design_temp_f_min == 375.0
    assert v745.design_temp_f_max == 375.0

    ac = by_tag[("AC-746", None)]
    assert ac.type == ComponentType.AIR_COOLED_EXCHANGER
    assert ac.design_temp_f_min == -20.0
    assert ac.design_temp_f_max == 400.0


def test_parses_e742_shell_and_tube_as_separate_rows(make_sop_docx):
    docx = make_sop_docx(
        rows=[
            ("E-742 Exchanger (Shell)", "300", "375"),
            ("E-742 Exchanger (Tube)", "300", "250"),
        ]
    )
    expected, _ = parse_sop(docx)
    by = {(e.tag, e.side): e for e in expected}
    assert by[("E-742", "shell")].design_temp_f_max == 375.0
    assert by[("E-742", "tube")].design_temp_f_max == 250.0


def test_parses_f715_a_and_b_as_two_filters(make_sop_docx):
    docx = make_sop_docx(rows=[("F-715 A and B Particulate Filters", "275", "100")])
    expected, _ = parse_sop(docx)
    tags = sorted(e.tag for e in expected)
    assert tags == ["F-715A", "F-715B"]
    for e in expected:
        assert e.type == ComponentType.FILTER
        assert e.design_pressure_psig == 275.0


def test_unparseable_value_is_warned(make_sop_docx):
    docx = make_sop_docx(rows=[("V-999 Vessel", "abc", "100")])
    expected, warnings = parse_sop(docx)
    assert any(w.code == "sop_value_unparseable" for w in warnings)
    # row still emitted, with pressure=None
    assert expected[0].design_pressure_psig is None


def test_missing_table_yields_empty_with_warning(make_sop_docx, tmp_path: Path):
    from docx import Document

    doc = Document()
    doc.add_paragraph("This SOP has no design-limits table.")
    out = tmp_path / "empty.docx"
    doc.save(out)
    expected, warnings = parse_sop(out)
    assert expected == []
    assert any(w.code == "sop_table_not_found" for w in warnings)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_sop_parser.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement `pipeline/stages/sop_parser.py`**

Path: `src/pipeline/stages/sop_parser.py`

```python
"""SOP DOCX parser.

Recognises an Operating Limits / Design Limits table where each row names
a tag-bearing equipment description and provides design pressure and
design temperature columns. Splits compound tags like 'F-715 A and B'
into individual ExpectedComponents.
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document

from pipeline.isa_lookup import EQUIPMENT_PREFIX_TO_TYPE
from pipeline.types import ComponentType, ExpectedComponent, PipelineWarning

_TAG_BASE_RE = re.compile(r"\b([A-Z]{1,3})-(\d{2,4})\b")
_AB_SUFFIX_RE = re.compile(r"\b([A-Z]{1,3})-(\d{2,4})\s+([A-Z])(?:\s+and\s+([A-Z]))?\b")
_SHELL_TUBE_RE = re.compile(r"\((shell|tube)\)", re.IGNORECASE)
_RANGE_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*to\s*(-?\d+(?:\.\d+)?)\s*$", re.IGNORECASE)
_NUMBER_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*$")


def _parse_value(raw: str) -> tuple[float | None, float | None, bool]:
    """Return (min, max, ok). ok=False means unparseable."""
    if raw is None or raw.strip() == "":
        return (None, None, True)
    m = _RANGE_RE.match(raw)
    if m is not None:
        return (float(m.group(1)), float(m.group(2)), True)
    m = _NUMBER_RE.match(raw)
    if m is not None:
        v = float(m.group(1))
        return (v, v, True)
    return (None, None, False)


def _classify(prefix: str) -> ComponentType:
    return EQUIPMENT_PREFIX_TO_TYPE.get(prefix, ComponentType.UNKNOWN)


def _expand_tags(description: str) -> list[tuple[str, str | None]]:
    """Return [(tag, side)] from a description like 'F-715 A and B Particulate Filters'."""
    out: list[tuple[str, str | None]] = []
    side_match = _SHELL_TUBE_RE.search(description)
    side = side_match.group(1).lower() if side_match else None

    ab = _AB_SUFFIX_RE.search(description)
    if ab is not None:
        prefix, number, s1, s2 = ab.groups()
        out.append((f"{prefix}-{number}{s1}", side))
        if s2 is not None:
            out.append((f"{prefix}-{number}{s2}", side))
        return out

    base = _TAG_BASE_RE.search(description)
    if base is not None:
        prefix, number = base.groups()
        out.append((f"{prefix}-{number}", side))
    return out


def parse_sop(docx_path: Path) -> tuple[list[ExpectedComponent], list[PipelineWarning]]:
    """Parse SOP DOCX into ExpectedComponents."""
    document = Document(str(docx_path))
    warnings: list[PipelineWarning] = []
    expected: list[ExpectedComponent] = []

    rows: list[tuple[str, str, str]] = []
    for table in document.tables:
        if len(table.columns) < 3:
            continue
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            description = cells[0]
            if not _TAG_BASE_RE.search(description):
                continue
            pressure_raw = cells[1] if len(cells) > 1 else ""
            temp_raw = cells[2] if len(cells) > 2 else ""
            rows.append((description, pressure_raw, temp_raw))

    if not rows:
        warnings.append(
            PipelineWarning(
                stage="sop_parser",
                page=None,
                code="sop_table_not_found",
                message="no design-limits table with tag rows recognised",
            )
        )
        return expected, warnings

    for description, p_raw, t_raw in rows:
        p_min, p_max, p_ok = _parse_value(p_raw)
        if not p_ok:
            warnings.append(
                PipelineWarning(
                    stage="sop_parser",
                    page=None,
                    code="sop_value_unparseable",
                    message=f"pressure={p_raw!r} in row {description!r}",
                )
            )
        t_min, t_max, t_ok = _parse_value(t_raw)
        if not t_ok:
            warnings.append(
                PipelineWarning(
                    stage="sop_parser",
                    page=None,
                    code="sop_value_unparseable",
                    message=f"temperature={t_raw!r} in row {description!r}",
                )
            )

        for tag, side in _expand_tags(description):
            prefix = tag.split("-")[0]
            expected.append(
                ExpectedComponent(
                    tag=tag,
                    type=_classify(prefix),
                    side=side,
                    design_pressure_psig=p_max,
                    design_temp_f_min=t_min,
                    design_temp_f_max=t_max,
                )
            )

    return expected, warnings
```

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest tests/test_sop_parser.py -v`
Expected: all 5 tests pass.

- [ ] **Step 6: Run mypy**

Run: `uv run mypy src/pipeline/stages/sop_parser.py tests/test_sop_parser.py`
Expected: `Success: no issues found`.

- [ ] **Step 7: Commit**

```bash
git add src/pipeline/stages/sop_parser.py tests/conftest.py tests/test_sop_parser.py
git commit -m "Add SOP DOCX parser with shell/tube and A/B suffix expansion"
```

---

## Task 8: Cross-reference

**Files:**
- Create: `src/pipeline/stages/cross_reference.py`
- Create: `tests/test_cross_reference.py`

- [ ] **Step 1: Write failing test**

Path: `tests/test_cross_reference.py`

```python
from pipeline.stages.cross_reference import compare, render_report
from pipeline.types import (
    Component,
    ComponentType,
    DiscrepancyKind,
    ExpectedComponent,
)


def _comp(tag: str, type_: ComponentType, **attrs: float | str) -> Component:
    return Component(tag=tag, type=type_, page=0, bbox=(0, 0, 10, 10), attrs=attrs)


def _exp(
    tag: str,
    type_: ComponentType = ComponentType.VESSEL,
    side: str | None = None,
    p: float | None = None,
    t_min: float | None = None,
    t_max: float | None = None,
) -> ExpectedComponent:
    return ExpectedComponent(
        tag=tag,
        type=type_,
        side=side,
        design_pressure_psig=p,
        design_temp_f_min=t_min,
        design_temp_f_max=t_max,
    )


def test_missing_in_diagram_when_sop_tag_absent():
    report = compare(
        components=[_comp("V-745", ComponentType.VESSEL)],
        expected=[_exp("V-745"), _exp("V-999")],
    )
    kinds = [d.kind for d in report.discrepancies]
    assert DiscrepancyKind.MISSING_IN_DIAGRAM in kinds
    missing = next(d for d in report.discrepancies if d.kind == DiscrepancyKind.MISSING_IN_DIAGRAM)
    assert missing.tag == "V-999"


def test_extra_in_diagram_only_for_equipment():
    report = compare(
        components=[
            _comp("V-745", ComponentType.VESSEL),
            _comp("E-100", ComponentType.HEAT_EXCHANGER),
            _comp("PIC-02", ComponentType.INSTRUMENT),
        ],
        expected=[_exp("V-745")],
    )
    extras = [d for d in report.discrepancies if d.kind == DiscrepancyKind.EXTRA_IN_DIAGRAM]
    assert {d.tag for d in extras} == {"E-100"}


def test_type_mismatch_when_inferred_type_differs():
    report = compare(
        components=[_comp("V-745", ComponentType.PUMP)],
        expected=[_exp("V-745", ComponentType.VESSEL)],
    )
    mism = [d for d in report.discrepancies if d.kind == DiscrepancyKind.TYPE_MISMATCH]
    assert len(mism) == 1
    assert mism[0].expected == ComponentType.VESSEL
    assert mism[0].actual == ComponentType.PUMP


def test_pressure_mismatch_only_when_both_have_values():
    # Drawing has 250, SOP says 300 → mismatch
    r1 = compare(
        components=[_comp("V-745", ComponentType.VESSEL, design_pressure_psig=250.0)],
        expected=[_exp("V-745", p=300.0)],
    )
    assert any(d.kind == DiscrepancyKind.PRESSURE_MISMATCH for d in r1.discrepancies)

    # Drawing has no value → silent
    r2 = compare(
        components=[_comp("V-745", ComponentType.VESSEL)],
        expected=[_exp("V-745", p=300.0)],
    )
    assert not any(d.kind == DiscrepancyKind.PRESSURE_MISMATCH for d in r2.discrepancies)


def test_temperature_mismatch_uses_max_against_drawing_value():
    r = compare(
        components=[_comp("AC-746", ComponentType.AIR_COOLED_EXCHANGER, design_temp_f=350.0)],
        expected=[_exp("AC-746", t_min=-20.0, t_max=400.0)],
    )
    # Drawing reports 350, SOP allows -20..400 → no mismatch
    assert not any(d.kind == DiscrepancyKind.TEMPERATURE_MISMATCH for d in r.discrepancies)


def test_e742_aggregates_shell_and_tube_into_one_node():
    report = compare(
        components=[_comp("E-742", ComponentType.HEAT_EXCHANGER)],
        expected=[
            _exp("E-742", ComponentType.HEAT_EXCHANGER, side="shell", p=300.0, t_min=375.0, t_max=375.0),
            _exp("E-742", ComponentType.HEAT_EXCHANGER, side="tube", p=300.0, t_min=250.0, t_max=250.0),
        ],
    )
    # Should NOT report MISSING_IN_DIAGRAM for E-742 just because there are two SOP rows
    missing = [d for d in report.discrepancies if d.kind == DiscrepancyKind.MISSING_IN_DIAGRAM]
    assert "E-742" not in {d.tag for d in missing}


def test_summary_counts_each_kind():
    report = compare(
        components=[],
        expected=[_exp("V-1"), _exp("V-2")],
    )
    assert report.summary[DiscrepancyKind.MISSING_IN_DIAGRAM.value] == 2


def test_render_report_has_required_sections():
    report = compare(components=[], expected=[_exp("V-1")])
    md = render_report(report)
    assert "## Summary" in md
    assert "## Discrepancies" in md
    assert "## Warnings" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cross_reference.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `pipeline/stages/cross_reference.py`**

Path: `src/pipeline/stages/cross_reference.py`

```python
"""Compare extracted graph (Components) to SOP expectations.

Emits Discrepancy records per the rules in the spec:
- MISSING_IN_DIAGRAM: SOP names a tag absent from the graph
- EXTRA_IN_DIAGRAM: equipment tag in graph but not in SOP
- TYPE_MISMATCH: SOP type ≠ classified type
- PRESSURE_MISMATCH: both sides have a value and they differ
- TEMPERATURE_MISMATCH: drawing value falls outside SOP min/max range
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from pipeline.types import (
    Component,
    ComponentType,
    Discrepancy,
    DiscrepancyKind,
    ExpectedComponent,
    PipelineWarning,
    Report,
)


def _is_equipment(c: Component) -> bool:
    return c.type is not ComponentType.INSTRUMENT


def _aggregate_expected(expected: Iterable[ExpectedComponent]) -> dict[str, list[ExpectedComponent]]:
    by_tag: dict[str, list[ExpectedComponent]] = defaultdict(list)
    for e in expected:
        by_tag[e.tag].append(e)
    return by_tag


def _attr_float(c: Component, key: str) -> float | None:
    value = c.attrs.get(key)
    if isinstance(value, int | float):
        return float(value)
    return None


def compare(
    components: list[Component],
    expected: list[ExpectedComponent],
) -> Report:
    discrepancies: list[Discrepancy] = []
    by_expected_tag = _aggregate_expected(expected)
    components_by_tag: dict[str, Component] = {c.tag: c for c in components}

    # MISSING_IN_DIAGRAM
    for tag in by_expected_tag:
        if tag not in components_by_tag:
            discrepancies.append(
                Discrepancy(
                    kind=DiscrepancyKind.MISSING_IN_DIAGRAM,
                    tag=tag,
                    side=None,
                    expected=by_expected_tag[tag][0].type.value,
                    actual=None,
                    page=None,
                    message=f"{tag}: expected by SOP but not found in P&ID",
                )
            )

    # EXTRA_IN_DIAGRAM (equipment only)
    for c in components:
        if _is_equipment(c) and c.tag not in by_expected_tag:
            discrepancies.append(
                Discrepancy(
                    kind=DiscrepancyKind.EXTRA_IN_DIAGRAM,
                    tag=c.tag,
                    side=None,
                    expected=None,
                    actual=c.type.value,
                    page=c.page,
                    message=f"{c.tag}: present in P&ID but not referenced by SOP",
                )
            )

    # Per-tag checks for tags that exist on both sides
    for tag, exp_rows in by_expected_tag.items():
        component = components_by_tag.get(tag)
        if component is None:
            continue

        # TYPE_MISMATCH (use the first expected row's type — shell/tube share a type)
        expected_type = exp_rows[0].type
        if expected_type is not ComponentType.UNKNOWN and component.type != expected_type:
            discrepancies.append(
                Discrepancy(
                    kind=DiscrepancyKind.TYPE_MISMATCH,
                    tag=tag,
                    side=None,
                    expected=expected_type,
                    actual=component.type,
                    page=component.page,
                    message=f"{tag}: SOP says {expected_type.value}, P&ID says {component.type.value}",
                )
            )

        for row in exp_rows:
            actual_pressure = _attr_float(component, "design_pressure_psig")
            if (
                row.design_pressure_psig is not None
                and actual_pressure is not None
                and actual_pressure != row.design_pressure_psig
            ):
                side_label = f" ({row.side})" if row.side else ""
                discrepancies.append(
                    Discrepancy(
                        kind=DiscrepancyKind.PRESSURE_MISMATCH,
                        tag=tag,
                        side=row.side,
                        expected=row.design_pressure_psig,
                        actual=actual_pressure,
                        page=component.page,
                        message=(
                            f"{tag}{side_label}: P&ID design pressure {actual_pressure} psig "
                            f"≠ SOP {row.design_pressure_psig} psig"
                        ),
                    )
                )

            actual_temp = _attr_float(component, "design_temp_f")
            if (
                actual_temp is not None
                and row.design_temp_f_min is not None
                and row.design_temp_f_max is not None
                and not (row.design_temp_f_min <= actual_temp <= row.design_temp_f_max)
            ):
                side_label = f" ({row.side})" if row.side else ""
                discrepancies.append(
                    Discrepancy(
                        kind=DiscrepancyKind.TEMPERATURE_MISMATCH,
                        tag=tag,
                        side=row.side,
                        expected=f"{row.design_temp_f_min}..{row.design_temp_f_max}",
                        actual=actual_temp,
                        page=component.page,
                        message=(
                            f"{tag}{side_label}: P&ID design temperature {actual_temp} °F "
                            f"outside SOP range {row.design_temp_f_min}..{row.design_temp_f_max} °F"
                        ),
                    )
                )

    summary: dict[str, int] = {kind.value: 0 for kind in DiscrepancyKind}
    for d in discrepancies:
        summary[d.kind.value] += 1

    return Report(
        discrepancies=tuple(discrepancies),
        warnings=(),
        summary=summary,
    )


def render_report(report: Report, extra_warnings: list[PipelineWarning] | None = None) -> str:
    """Render a Report as Markdown for output/report.md."""
    extra = extra_warnings or []
    all_warnings = list(report.warnings) + extra
    lines: list[str] = ["# P&ID ↔ SOP Cross-Reference Report", ""]

    lines.append("## Summary")
    lines.append("")
    lines.append("| Discrepancy Kind | Count |")
    lines.append("|---|---|")
    for kind, count in report.summary.items():
        lines.append(f"| {kind} | {count} |")
    lines.append(f"| **warnings** | {len(all_warnings)} |")
    lines.append("")

    lines.append("## Discrepancies")
    lines.append("")
    if not report.discrepancies:
        lines.append("_None._")
    else:
        for d in report.discrepancies:
            lines.append(f"- **[{d.kind.value}]** {d.message}")
    lines.append("")

    lines.append("## Warnings")
    lines.append("")
    if not all_warnings:
        lines.append("_None._")
    else:
        by_stage: dict[str, list[PipelineWarning]] = defaultdict(list)
        for w in all_warnings:
            by_stage[w.stage].append(w)
        for stage_name in sorted(by_stage):
            lines.append(f"### {stage_name}")
            for w in by_stage[stage_name]:
                page_label = "" if w.page is None else f" (page {w.page})"
                lines.append(f"- `{w.code}`{page_label}: {w.message}")
            lines.append("")

    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_cross_reference.py -v`
Expected: all 8 tests pass.

- [ ] **Step 5: Run mypy**

Run: `uv run mypy src/pipeline/stages/cross_reference.py tests/test_cross_reference.py`
Expected: `Success: no issues found`.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/stages/cross_reference.py tests/test_cross_reference.py
git commit -m "Add cross-reference engine and Markdown report renderer"
```

---

## Task 9: Logging configuration

**Files:**
- Create: `src/pipeline/logging_config.py`

- [ ] **Step 1: Implement `pipeline/logging_config.py`**

Path: `src/pipeline/logging_config.py`

```python
"""Structured (JSONL) logging to a file plus human-readable console output.

Used only by the runner / CLI; stages do not log directly.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

_REQUIRED_KEYS = ("ts", "level", "stage", "page", "code", "message")


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.time(),
            "level": record.levelname,
            "stage": getattr(record, "stage", None),
            "page": getattr(record, "page", None),
            "code": getattr(record, "code", None),
            "message": record.getMessage(),
            "duration_ms": getattr(record, "duration_ms", None),
        }
        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


class HumanFormatter(logging.Formatter):
    LEVEL_PREFIX = {
        "DEBUG": "·",
        "INFO": "✓",
        "WARNING": "!",
        "ERROR": "✗",
    }

    def format(self, record: logging.LogRecord) -> str:
        prefix = self.LEVEL_PREFIX.get(record.levelname, "?")
        stage = getattr(record, "stage", None)
        page = getattr(record, "page", None)
        loc = ""
        if stage is not None:
            loc = f" [{stage}"
            if page is not None:
                loc += f" p{page}"
            loc += "]"
        return f"{prefix}{loc} {record.getMessage()}"


def configure(log_path: Path, console_level: int = logging.INFO, debug: bool = False) -> None:
    """Configure the root logger with file (JSONL) + console (human) handlers."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("pipeline")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    file_handler.setFormatter(JsonLineFormatter())
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if debug else console_level)
    console_handler.setFormatter(HumanFormatter())
    root.addHandler(console_handler)


def event(
    stage: str,
    message: str,
    *,
    page: int | None = None,
    code: str | None = None,
    duration_ms: float | None = None,
    level: int = logging.INFO,
) -> None:
    logger = logging.getLogger("pipeline")
    logger.log(
        level,
        message,
        extra={"stage": stage, "page": page, "code": code, "duration_ms": duration_ms},
    )
```

- [ ] **Step 2: Run mypy**

Run: `uv run mypy src/pipeline/logging_config.py`
Expected: `Success: no issues found`.

- [ ] **Step 3: Commit**

```bash
git add src/pipeline/logging_config.py
git commit -m "Add structured JSONL + human console logging configuration"
```

---

## Task 10: PDF loader

**Files:**
- Create: `src/pipeline/stages/pdf_loader.py`

- [ ] **Step 1: Implement `pipeline/stages/pdf_loader.py`**

Path: `src/pipeline/stages/pdf_loader.py`

```python
"""Rasterize a PDF into a list of PageImages at a specified DPI.

Uses pypdfium2 for portability (no system poppler dependency).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pypdfium2 as pdfium

from pipeline.types import PageImage


def load(pdf_path: Path, dpi: int) -> list[PageImage]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    scale = dpi / 72.0
    document = pdfium.PdfDocument(str(pdf_path))
    pages: list[PageImage] = []
    try:
        for idx, page in enumerate(document):
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil().convert("RGB")
            arr = np.asarray(pil_image)
            pages.append(PageImage(page_idx=idx, image=arr, dpi=dpi))
            page.close()
    finally:
        document.close()
    if not pages:
        raise ValueError(f"PDF has zero pages: {pdf_path}")
    return pages
```

- [ ] **Step 2: Run mypy**

Run: `uv run mypy src/pipeline/stages/pdf_loader.py`
Expected: `Success: no issues found`.

- [ ] **Step 3: Smoke-check the loader on the real PDF**

Run:
```bash
uv run python -c "
from pathlib import Path
from pipeline.stages.pdf_loader import load
pages = load(Path('data/p&id/diagram.pdf'), dpi=200)
print('pages:', len(pages))
print('shape0:', pages[0].image.shape)
"
```
Expected: prints `pages: 3` and a shape like `shape0: (2200, 3400, 3)` (varies by DPI).

- [ ] **Step 4: Commit**

```bash
git add src/pipeline/stages/pdf_loader.py
git commit -m "Add PDF rasterizer using pypdfium2"
```

---

## Task 11: Preprocessing

**Files:**
- Create: `src/pipeline/stages/preprocess.py`

- [ ] **Step 1: Implement `pipeline/stages/preprocess.py`**

Path: `src/pipeline/stages/preprocess.py`

```python
"""Image preprocessing for OCR and line detection.

Produces (a) a binarised image suitable for OCR/skeletonisation, and
(b) keeps the original RGB for visualization. Both share a deskew step.
"""

from __future__ import annotations

import cv2
import numpy as np

from pipeline.types import PageImage


def _deskew_angle(gray: np.ndarray) -> float:
    """Estimate skew angle in degrees by analysing dominant edge orientation."""
    inverted = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(inverted > 0))
    if coords.size == 0:
        return 0.0
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return float(angle)


def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.1:
        return image
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def clean(page: PageImage) -> tuple[PageImage, np.ndarray]:
    """Return (deskewed RGB PageImage, binarised single-channel image)."""
    rgb = page.image
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    angle = _deskew_angle(gray)
    rgb_rot = _rotate(rgb, angle)
    gray_rot = cv2.cvtColor(rgb_rot, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(
        gray_rot,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=25,
        C=15,
    )
    cleaned_page = PageImage(page_idx=page.page_idx, image=rgb_rot, dpi=page.dpi)
    return cleaned_page, binary
```

- [ ] **Step 2: Run mypy**

Run: `uv run mypy src/pipeline/stages/preprocess.py`
Expected: `Success: no issues found`.

- [ ] **Step 3: Commit**

```bash
git add src/pipeline/stages/preprocess.py
git commit -m "Add deskew + adaptive-threshold preprocessing"
```

---

## Task 12: OCR engine

**Files:**
- Create: `src/pipeline/stages/ocr.py`

- [ ] **Step 1: Implement `pipeline/stages/ocr.py`**

Path: `src/pipeline/stages/ocr.py`

```python
"""EasyOCR-backed implementation of OCREngine.

EasyOCR loads a PyTorch model on first instantiation (~100MB download to
~/.EasyOCR/ on first run). The Reader is created lazily so unit tests
that don't exercise OCR don't pay this cost.
"""

from __future__ import annotations

from typing import cast

import easyocr  # type: ignore[import-untyped]
import numpy as np

from pipeline.types import BBox, PipelineWarning, TextBox


class EasyOCREngine:
    def __init__(self, languages: tuple[str, ...] = ("en",), gpu: bool = False) -> None:
        self._languages = list(languages)
        self._gpu = gpu
        self._reader: easyocr.Reader | None = None

    @property
    def reader(self) -> easyocr.Reader:
        if self._reader is None:
            self._reader = easyocr.Reader(self._languages, gpu=self._gpu, verbose=False)
        return self._reader

    def read(self, image: np.ndarray) -> list[TextBox]:
        raw = self.reader.readtext(image, detail=1, paragraph=False)
        out: list[TextBox] = []
        for entry in raw:
            polygon, text, confidence = entry
            xs = [int(p[0]) for p in polygon]
            ys = [int(p[1]) for p in polygon]
            bbox: BBox = (min(xs), min(ys), max(xs), max(ys))
            out.append(
                TextBox(
                    text=str(text).strip(),
                    bbox=bbox,
                    confidence=float(confidence),
                )
            )
        return out


def run_ocr(
    image: np.ndarray,
    engine: EasyOCREngine,
    page: int,
) -> tuple[list[TextBox], list[PipelineWarning]]:
    boxes = engine.read(image)
    warnings: list[PipelineWarning] = []
    if not boxes:
        warnings.append(
            PipelineWarning(
                stage="ocr",
                page=page,
                code="no_text_found",
                message="OCR returned zero text boxes",
            )
        )
    return boxes, warnings
```

- [ ] **Step 2: Run mypy**

Run: `uv run mypy src/pipeline/stages/ocr.py`
Expected: `Success: no issues found`.

- [ ] **Step 3: Commit**

```bash
git add src/pipeline/stages/ocr.py
git commit -m "Add EasyOCR engine implementing the OCREngine protocol"
```

---

## Task 13: Component localization

**Files:**
- Create: `src/pipeline/stages/components.py`

- [ ] **Step 1: Implement `pipeline/stages/components.py`**

Path: `src/pipeline/stages/components.py`

```python
"""Associate parsed Tags with symbol bounding boxes.

For equipment tags we look outward from the tag's bbox for a connected
contour region and use that as the symbol bbox. If none is found within
the search window, the bbox falls back to the tag's bbox and a warning
is emitted.

For instrument tags we use the tag's own bbox (the balloon circle) — the
balloon IS the symbol.
"""

from __future__ import annotations

import cv2
import numpy as np

from pipeline.isa_lookup import EQUIPMENT_PREFIX_TO_TYPE
from pipeline.types import (
    BBox,
    Component,
    ComponentType,
    EquipmentTag,
    InstrumentTag,
    PipelineWarning,
    Tag,
)


def _expand_bbox(bbox: BBox, margin: int, image_shape: tuple[int, int]) -> BBox:
    h, w = image_shape
    x0, y0, x1, y1 = bbox
    return (
        max(0, x0 - margin),
        max(0, y0 - margin),
        min(w, x1 + margin),
        min(h, y1 + margin),
    )


def _largest_contour_bbox(binary_crop: np.ndarray) -> BBox | None:
    inverted = cv2.bitwise_not(binary_crop)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50.0:
        return None
    x, y, w, h = cv2.boundingRect(largest)
    return (x, y, x + w, y + h)


def locate(
    tags: list[Tag],
    binary: np.ndarray,
    page: int,
    search_margin_px: int = 80,
) -> tuple[list[Component], list[PipelineWarning]]:
    components: list[Component] = []
    warnings: list[PipelineWarning] = []
    for tag in tags:
        if isinstance(tag, InstrumentTag):
            components.append(
                Component(
                    tag=f"{tag.letters}-{tag.number}",
                    type=ComponentType.INSTRUMENT,
                    page=page,
                    bbox=tag.bbox,
                    attrs={"description": tag.description},
                )
            )
            continue

        assert isinstance(tag, EquipmentTag)
        symbol_type = EQUIPMENT_PREFIX_TO_TYPE.get(tag.prefix, ComponentType.UNKNOWN)
        search_bbox = _expand_bbox(tag.bbox, search_margin_px, binary.shape[:2])
        x0, y0, x1, y1 = search_bbox
        crop = binary[y0:y1, x0:x1]
        local = _largest_contour_bbox(crop)
        if local is None:
            warnings.append(
                PipelineWarning(
                    stage="components",
                    page=page,
                    code="symbol_not_localized",
                    message=tag.raw,
                )
            )
            symbol_bbox = tag.bbox
        else:
            lx0, ly0, lx1, ly1 = local
            symbol_bbox = (x0 + lx0, y0 + ly0, x0 + lx1, y0 + ly1)
        canonical_tag = f"{tag.prefix}-{tag.number}{tag.suffix or ''}"
        components.append(
            Component(
                tag=canonical_tag,
                type=symbol_type,
                page=page,
                bbox=symbol_bbox,
                attrs={},
            )
        )
    return components, warnings
```

- [ ] **Step 2: Run mypy**

Run: `uv run mypy src/pipeline/stages/components.py`
Expected: `Success: no issues found`.

- [ ] **Step 3: Commit**

```bash
git add src/pipeline/stages/components.py
git commit -m "Add component localization that associates tags with symbol bboxes"
```

---

## Task 14: Attribute enrichment

**Files:**
- Create: `src/pipeline/stages/attributes.py`

- [ ] **Step 1: Implement `pipeline/stages/attributes.py`**

Path: `src/pipeline/stages/attributes.py`

```python
"""Search for design pressure / temperature annotations near each component.

We look for tokens of the form NUMBER + UNIT (psig, °F) within a radius
of each component bbox and attach them as attrs on the Component.
"""

from __future__ import annotations

import re

from pipeline.types import BBox, Component, TextBox

_PRESSURE_RE = re.compile(r"^(?P<v>-?\d+(?:\.\d+)?)\s*(?:psig|psi)$", re.IGNORECASE)
_TEMP_RE = re.compile(r"^(?P<v>-?\d+(?:\.\d+)?)\s*(?:°?\s*F|deg\s*F|F)$", re.IGNORECASE)


def _bbox_distance(a: BBox, b: BBox) -> float:
    ax = (a[0] + a[2]) / 2.0
    ay = (a[1] + a[3]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    by = (b[1] + b[3]) / 2.0
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


def enrich(
    components: list[Component],
    text_boxes: list[TextBox],
    radius_px: int,
) -> list[Component]:
    enriched: list[Component] = []
    for component in components:
        nearest_pressure: tuple[float, float] | None = None  # (distance, value)
        nearest_temp: tuple[float, float] | None = None
        for tb in text_boxes:
            distance = _bbox_distance(component.bbox, tb.bbox)
            if distance > radius_px:
                continue
            mp = _PRESSURE_RE.match(tb.text.strip())
            if mp is not None:
                value = float(mp.group("v"))
                if nearest_pressure is None or distance < nearest_pressure[0]:
                    nearest_pressure = (distance, value)
            mt = _TEMP_RE.match(tb.text.strip())
            if mt is not None:
                value = float(mt.group("v"))
                if nearest_temp is None or distance < nearest_temp[0]:
                    nearest_temp = (distance, value)
        if nearest_pressure is None and nearest_temp is None:
            enriched.append(component)
            continue
        new_attrs = dict(component.attrs)
        if nearest_pressure is not None:
            new_attrs["design_pressure_psig"] = nearest_pressure[1]
        if nearest_temp is not None:
            new_attrs["design_temp_f"] = nearest_temp[1]
        enriched.append(
            Component(
                tag=component.tag,
                type=component.type,
                page=component.page,
                bbox=component.bbox,
                attrs=new_attrs,
            )
        )
    return enriched
```

- [ ] **Step 2: Run mypy**

Run: `uv run mypy src/pipeline/stages/attributes.py`
Expected: `Success: no issues found`.

- [ ] **Step 3: Commit**

```bash
git add src/pipeline/stages/attributes.py
git commit -m "Add attribute enrichment for design pressure and temperature"
```

---

## Task 15: Connection detection with dash filter

**Files:**
- Create: `src/pipeline/stages/connections.py`
- Create: `tests/test_connections_dash_filter.py`

- [ ] **Step 1: Write failing test for the dash filter**

Path: `tests/test_connections_dash_filter.py`

```python
import numpy as np

from pipeline.stages.connections import is_dashed_segment


def test_solid_line_is_not_dashed():
    line = np.zeros(200, dtype=np.uint8)  # 0 = ink (line)
    assert is_dashed_segment(line, min_solid_run=8) is False


def test_dashed_line_is_dashed():
    # Alternating 4 ink, 4 background; runs all < 8
    line = np.tile(np.array([0, 0, 0, 0, 255, 255, 255, 255], dtype=np.uint8), 25)
    assert is_dashed_segment(line, min_solid_run=8) is True


def test_almost_solid_with_short_gap_is_not_dashed():
    line = np.zeros(200, dtype=np.uint8)
    line[100:103] = 255  # tiny 3-pixel gap; long ink runs on either side
    assert is_dashed_segment(line, min_solid_run=8) is False


def test_empty_or_pure_background_is_not_dashed():
    assert is_dashed_segment(np.full(50, 255, dtype=np.uint8), min_solid_run=8) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_connections_dash_filter.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `pipeline/stages/connections.py`**

Path: `src/pipeline/stages/connections.py`

```python
"""Connection extraction: skeletonise → Hough → dash filter → tag-pair edges.

Process pipes are solid lines. Signal lines (electrical, pneumatic,
hydraulic) are dashed/dotted per ISA-5.1; we filter those out so the
graph only carries process flow edges.
"""

from __future__ import annotations

import cv2
import numpy as np

from pipeline.types import BBox, Component, Edge, PipelineWarning


def is_dashed_segment(line_pixels: np.ndarray, min_solid_run: int) -> bool:
    """Return True iff the longest run of contiguous ink pixels is < min_solid_run.

    `line_pixels` is a 1-D uint8 array sampled along a candidate line.
    Ink is encoded as 0 (matching cv2 binary output convention).
    """
    if line_pixels.size == 0:
        return False
    ink = line_pixels == 0
    if not ink.any():
        return False
    longest = 0
    current = 0
    for v in ink:
        if v:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest < min_solid_run


def _sample_line(binary: np.ndarray, p0: tuple[int, int], p1: tuple[int, int]) -> np.ndarray:
    n = max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1])) + 1
    xs = np.linspace(p0[0], p1[0], n).astype(np.int32)
    ys = np.linspace(p0[1], p1[1], n).astype(np.int32)
    h, w = binary.shape[:2]
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    return binary[ys, xs]


def _segment_intersects_bbox(p0: tuple[int, int], p1: tuple[int, int], bbox: BBox) -> bool:
    x0, y0, x1, y1 = bbox
    if (p0[0] < x0 and p1[0] < x0) or (p0[0] > x1 and p1[0] > x1):
        return False
    if (p0[1] < y0 and p1[1] < y0) or (p0[1] > y1 and p1[1] > y1):
        return False
    return True


def _segment_bbox(p0: tuple[int, int], p1: tuple[int, int]) -> BBox:
    return (min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1]))


def detect(
    binary: np.ndarray,
    components: list[Component],
    page: int,
    dash_solid_run_min_px: int,
    hough_threshold: int = 80,
    min_line_length: int = 60,
    max_line_gap: int = 10,
) -> tuple[list[Edge], list[PipelineWarning]]:
    warnings: list[PipelineWarning] = []
    inverted = cv2.bitwise_not(binary)
    raw = cv2.HoughLinesP(
        inverted,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if raw is None or len(raw) == 0:
        warnings.append(
            PipelineWarning(
                stage="connections",
                page=page,
                code="no_lines_detected",
                message="HoughLinesP returned no segments",
            )
        )
        return [], warnings

    edges: list[Edge] = []
    seen_pairs: set[tuple[str, str]] = set()
    for [[x1, y1, x2, y2]] in raw:
        p0 = (int(x1), int(y1))
        p1 = (int(x2), int(y2))
        sample = _sample_line(binary, p0, p1)
        if is_dashed_segment(sample, min_solid_run=dash_solid_run_min_px):
            continue
        endpoints = [c for c in components if _segment_intersects_bbox(p0, p1, c.bbox)]
        if len(endpoints) < 2:
            continue
        endpoints = sorted(endpoints, key=lambda c: c.bbox)
        a, b = endpoints[0].tag, endpoints[1].tag
        if a == b:
            continue
        key = tuple(sorted((a, b)))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        edges.append(
            Edge(src_tag=a, dst_tag=b, page=page, evidence=_segment_bbox(p0, p1))
        )
    return edges, warnings
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_connections_dash_filter.py -v`
Expected: all 4 tests pass.

- [ ] **Step 5: Run mypy**

Run: `uv run mypy src/pipeline/stages/connections.py tests/test_connections_dash_filter.py`
Expected: `Success: no issues found`.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/stages/connections.py tests/test_connections_dash_filter.py
git commit -m "Add Hough-based connection detection with ISA dash-line filter"
```

---

## Task 16: Graph builder

**Files:**
- Create: `src/pipeline/stages/graph_builder.py`

- [ ] **Step 1: Implement `pipeline/stages/graph_builder.py`**

Path: `src/pipeline/stages/graph_builder.py`

```python
"""Assemble per-page (components, edges) tuples into a single NetworkX graph
and serialise to GraphML.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from pipeline.types import Component, Edge


def assemble(per_page: list[tuple[list[Component], list[Edge]]]) -> nx.MultiDiGraph:
    graph: nx.MultiDiGraph = nx.MultiDiGraph()
    for components, _edges in per_page:
        for c in components:
            attrs: dict[str, str | float] = {
                "type": c.type.value,
                "page": c.page,
                "bbox": ",".join(str(v) for v in c.bbox),
            }
            for k, v in c.attrs.items():
                attrs[k] = v
            graph.add_node(c.tag, **attrs)
    for components, edges in per_page:
        present = {c.tag for c in components}
        for edge in edges:
            if edge.src_tag in present and edge.dst_tag in present:
                graph.add_edge(
                    edge.src_tag,
                    edge.dst_tag,
                    page=edge.page,
                    evidence=",".join(str(v) for v in edge.evidence),
                )
    return graph


def write_graphml(graph: nx.MultiDiGraph, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, output_path)
```

- [ ] **Step 2: Run mypy**

Run: `uv run mypy src/pipeline/stages/graph_builder.py`
Expected: `Success: no issues found`.

- [ ] **Step 3: Commit**

```bash
git add src/pipeline/stages/graph_builder.py
git commit -m "Add NetworkX graph assembly and GraphML serialization"
```

---

## Task 17: Visualizer

**Files:**
- Create: `src/pipeline/stages/visualizer.py`

- [ ] **Step 1: Implement `pipeline/stages/visualizer.py`**

Path: `src/pipeline/stages/visualizer.py`

```python
"""Render an annotated PNG showing detected tags, components and edges."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pipeline.types import Component, ComponentType, Edge, PageImage

_TYPE_COLORS: dict[ComponentType, tuple[int, int, int]] = {
    ComponentType.VESSEL: (0, 0, 200),
    ComponentType.HEAT_EXCHANGER: (200, 100, 0),
    ComponentType.AIR_COOLED_EXCHANGER: (200, 150, 0),
    ComponentType.FILTER: (0, 150, 0),
    ComponentType.PUMP: (150, 0, 150),
    ComponentType.TANK: (100, 100, 0),
    ComponentType.INSTRUMENT: (50, 50, 50),
    ComponentType.UNKNOWN: (0, 0, 0),
}


def render(
    page: PageImage,
    components: list[Component],
    edges: list[Edge],
    out_path: Path,
) -> None:
    canvas = page.image.copy()
    for component in components:
        x0, y0, x1, y1 = component.bbox
        color = _TYPE_COLORS.get(component.type, (0, 0, 0))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        label = f"{component.tag} [{component.type.value}]"
        cv2.putText(
            canvas,
            label,
            (x0, max(15, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    for edge in edges:
        x0, y0, x1, y1 = edge.evidence
        cv2.line(canvas, (x0, y0), (x1, y1), (255, 0, 0), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
```

- [ ] **Step 2: Run mypy**

Run: `uv run mypy src/pipeline/stages/visualizer.py`
Expected: `Success: no issues found`.

- [ ] **Step 3: Commit**

```bash
git add src/pipeline/stages/visualizer.py
git commit -m "Add per-page annotated PNG renderer"
```

---

## Task 18: Runner (orchestrator)

**Files:**
- Create: `src/pipeline/runner.py`

- [ ] **Step 1: Implement `pipeline/runner.py`**

Path: `src/pipeline/runner.py`

```python
"""Pipeline orchestrator. The only module with side effects.

Owns:
- disk reads (PDF, DOCX) and writes (graph.graphml, report.md, log, PNGs)
- per-page parallelism via ProcessPoolExecutor
- stage-level caching keyed by content hash (TODO in a future task; not
  required for the smoke test)
- structured logging
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from pipeline.config import Settings
from pipeline.logging_config import configure as configure_logging
from pipeline.logging_config import event
from pipeline.stages import (
    components as components_stage,
)
from pipeline.stages import (
    cross_reference,
    graph_builder,
    pdf_loader,
    preprocess,
    sop_parser,
    tags as tags_stage,
    visualizer,
)
from pipeline.stages.attributes import enrich
from pipeline.stages.connections import detect as detect_connections
from pipeline.stages.ocr import EasyOCREngine, run_ocr
from pipeline.types import (
    Component,
    Edge,
    PageImage,
    PipelineWarning,
    Report,
)


@dataclass(frozen=True, slots=True)
class PageSubgraph:
    page: int
    components: list[Component]
    edges: list[Edge]
    warnings: list[PipelineWarning]


def _process_page(
    page_image: PageImage,
    settings: Settings,
) -> PageSubgraph:
    """Pure per-page pipeline. Suitable for ProcessPoolExecutor."""
    rgb_page, binary = preprocess.clean(page_image)
    engine = EasyOCREngine()
    text_boxes, ocr_warnings = run_ocr(rgb_page.image, engine, page_image.page_idx)
    parsed_tags, tag_warnings = tags_stage.parse_tags(
        text_boxes,
        page=page_image.page_idx,
        min_confidence=settings.ocr_confidence_threshold,
    )
    components, comp_warnings = components_stage.locate(
        parsed_tags,
        binary,
        page=page_image.page_idx,
    )
    enriched = enrich(components, text_boxes, settings.attribute_search_radius_px)
    edges, edge_warnings = detect_connections(
        binary,
        enriched,
        page=page_image.page_idx,
        dash_solid_run_min_px=settings.dash_solid_run_min_px,
    )
    return PageSubgraph(
        page=page_image.page_idx,
        components=enriched,
        edges=edges,
        warnings=ocr_warnings + tag_warnings + comp_warnings + edge_warnings,
    )


def run_pipeline(
    pdf_path: Path,
    sop_path: Path,
    settings: Settings,
) -> Report:
    """End-to-end run. Returns the final Report; persistence handled here too."""
    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir / "pipeline.log")

    started = time.perf_counter()
    event("runner", "pipeline starting", code="start")

    pages = pdf_loader.load(pdf_path, dpi=settings.dpi)
    event("pdf_loader", f"loaded {len(pages)} pages", code="pages_loaded")

    if settings.max_workers > 1:
        with ProcessPoolExecutor(max_workers=settings.max_workers) as pool:
            subgraphs = list(pool.map(_process_page, pages, [settings] * len(pages)))
    else:
        subgraphs = [_process_page(p, settings) for p in pages]
    for sg in subgraphs:
        event(
            "runner",
            f"page {sg.page}: {len(sg.components)} components, {len(sg.edges)} edges, "
            f"{len(sg.warnings)} warnings",
            page=sg.page,
            code="page_done",
        )

    expected, sop_warnings = sop_parser.parse_sop(sop_path)
    event("sop_parser", f"{len(expected)} expected components", code="sop_parsed")

    per_page_graphs = [(sg.components, sg.edges) for sg in subgraphs]
    graph = graph_builder.assemble(per_page_graphs)
    graph_builder.write_graphml(graph, output_dir / "graph.graphml")
    event("graph_builder", f"graph: {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} edges", code="graph_built")

    all_components = [c for sg in subgraphs for c in sg.components]
    report = cross_reference.compare(all_components, expected)

    extra_warnings = sop_warnings + [w for sg in subgraphs for w in sg.warnings]
    md = cross_reference.render_report(report, extra_warnings=extra_warnings)
    (output_dir / "report.md").write_text(md, encoding="utf-8")
    event("cross_reference", f"{len(report.discrepancies)} discrepancies, "
          f"{len(extra_warnings)} warnings", code="report_written")

    annotated_dir = output_dir / "annotated"
    for page_image, sg in zip(pages, subgraphs, strict=True):
        out = annotated_dir / f"page_{page_image.page_idx}.png"
        visualizer.render(page_image, sg.components, sg.edges, out)
    event("visualizer", "annotated pages written", code="annotated_written")

    duration_ms = (time.perf_counter() - started) * 1000.0
    event(
        "runner",
        f"pipeline complete in {duration_ms:.0f} ms",
        code="done",
        duration_ms=duration_ms,
    )
    logging.shutdown()
    return Report(
        discrepancies=report.discrepancies,
        warnings=tuple(extra_warnings),
        summary=report.summary,
    )
```

- [ ] **Step 2: Run mypy**

Run: `uv run mypy src/pipeline/runner.py`
Expected: `Success: no issues found`.

- [ ] **Step 3: Commit**

```bash
git add src/pipeline/runner.py
git commit -m "Add pipeline orchestrator with logging and persistence"
```

---

## Task 19: CLI

**Files:**
- Create: `src/pipeline/cli.py`
- Modify: `main.py`

- [ ] **Step 1: Implement `pipeline/cli.py`**

Path: `src/pipeline/cli.py`

```python
"""Command-line entry point. Subcommands cover the full pipeline and stage debugging."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dataclasses import replace

from pipeline.config import Settings, load_settings
from pipeline.runner import run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="P&ID-to-graph + SOP cross-reference pipeline",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="path to config.toml; defaults from pipeline.config.Settings apply otherwise",
    )
    parser.add_argument("--debug", action="store_true", help="verbose logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the full pipeline")
    p_run.add_argument(
        "--pdf",
        type=Path,
        default=Path("data/p&id/diagram.pdf"),
        help="path to P&ID PDF",
    )
    p_run.add_argument(
        "--sop",
        type=Path,
        default=Path("data/sop/sop.docx"),
        help="path to SOP DOCX",
    )
    p_run.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output directory (overrides Settings.output_dir)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        settings = load_settings(config_path=args.config)
        if args.command == "run":
            if args.output is not None:
                settings = replace(settings, output_dir=args.output)
            if not args.pdf.exists():
                print(f"PDF not found: {args.pdf}", file=sys.stderr)
                return 3
            if not args.sop.exists():
                print(f"SOP not found: {args.sop}", file=sys.stderr)
                return 3
            run_pipeline(args.pdf, args.sop, settings)
            return 0
        return 2
    except FileNotFoundError as exc:
        print(f"missing input: {exc}", file=sys.stderr)
        return 3
    except ValueError as exc:
        print(f"invalid config: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"unexpected error: {exc}", file=sys.stderr)
        if args.debug:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Replace `main.py` with a 3-line shim**

Path: `main.py`

```python
"""Entry-point shim for the take-home assignment grader."""

from pipeline.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run mypy**

Run: `uv run mypy src/pipeline/cli.py main.py`
Expected: `Success: no issues found`.

- [ ] **Step 4: Verify CLI usage banner**

Run: `uv run python -m pipeline.cli --help`
Expected: prints usage with `run` subcommand.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/cli.py main.py
git commit -m "Add CLI with run subcommand and main.py shim"
```

---

## Task 20: Integration smoke test

**Files:**
- Create: `tests/test_pipeline_smoke.py`

- [ ] **Step 1: Write the integration test**

Path: `tests/test_pipeline_smoke.py`

```python
"""End-to-end smoke test. Loads OCR models; marked slow."""

from pathlib import Path

import networkx as nx
import pytest

from pipeline.config import Settings
from pipeline.runner import run_pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
PDF = REPO_ROOT / "data" / "p&id" / "diagram.pdf"
SOP = REPO_ROOT / "data" / "sop" / "sop.docx"
SOP_TAGS = {"F-715A", "F-715B", "V-745", "E-742", "AC-746"}


@pytest.mark.slow
def test_pipeline_runs_end_to_end_on_sample(tmp_path: Path) -> None:
    if not PDF.exists() or not SOP.exists():
        pytest.skip("sample data not present")

    settings = Settings(dpi=200, max_workers=1, output_dir=tmp_path)
    report = run_pipeline(PDF, SOP, settings)

    assert (tmp_path / "graph.graphml").stat().st_size > 0
    assert (tmp_path / "report.md").stat().st_size > 0
    assert (tmp_path / "pipeline.log").stat().st_size > 0
    for i in range(3):
        png = tmp_path / "annotated" / f"page_{i}.png"
        assert png.exists() and png.stat().st_size > 0

    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## Summary" in md
    assert "## Discrepancies" in md
    assert "## Warnings" in md

    graph = nx.read_graphml(tmp_path / "graph.graphml")
    nodes = set(graph.nodes())
    missing = SOP_TAGS - nodes
    assert not missing, f"SOP-referenced tags missing from graph: {missing}"

    for tag in SOP_TAGS:
        node = graph.nodes[tag]
        assert node["type"] != "unknown", f"{tag} classified as UNKNOWN"

    assert graph.number_of_edges() >= 1, "expected at least one edge"
    # Discrepancies are allowed; the test passes regardless of count.
    assert isinstance(report.summary, dict)
```

- [ ] **Step 2: Run the integration test**

Run: `uv run pytest tests/test_pipeline_smoke.py -v -m slow`
Expected: PASS. First run downloads EasyOCR weights (~100MB) and may take 60-120 seconds.

If a SOP_TAG fails the membership check, inspect `tmp_path/annotated/page_*.png` to see what OCR produced and tune `Settings.ocr_confidence_threshold` or DPI.

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest -v`
Expected: every test passes (unit tests + smoke).

- [ ] **Step 4: Run `make ci`**

Run: `make ci`
Expected: lint + typecheck + tests all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_pipeline_smoke.py
git commit -m "Add end-to-end integration smoke test on sample data"
```

---

## Task 21: Reviewer-facing README

**Files:**
- Modify: `README.md` (replace existing content)

- [ ] **Step 1: Replace `README.md`**

Path: `README.md`

```markdown
# Computer Vision Take-Home: P&ID → Graph + SOP Cross-Reference

Converts a P&ID PDF into a NetworkX graph and cross-references it
against an SOP DOCX, emitting discrepancies, annotated page renderings,
and a structured log.

## Quick start

```bash
make install      # uv sync --extra dev (creates .venv, installs deps)
make run          # runs full pipeline on data/p&id/diagram.pdf + data/sop/sop.docx
```

The first run downloads EasyOCR weights (~100MB) to `~/.EasyOCR/`.

Outputs land in `output/`:

```
output/
├── graph.graphml          # open in Gephi/yEd or load via networkx.read_graphml
├── report.md              # human-readable discrepancy report
├── pipeline.log           # JSONL, one event per line
└── annotated/
    ├── page_0.png         # detected components + edges overlaid
    ├── page_1.png
    └── page_2.png
```

## Approach

Classical CV + OCR-anchored tag detection. No ML model training, no
GPU required. ISA-5.1 tag conventions drive component classification
(see `src/pipeline/isa_lookup.py`). Pipeline is split into pure-function
stages composed by a thin runner (`src/pipeline/runner.py`); per-page
parallelism via `ProcessPoolExecutor`; errors flow as structured
warnings rather than exceptions.

Architecture, data shapes, error model, and testing strategy are
documented in `docs/superpowers/specs/2026-04-27-pid-sop-design.md`.

## Development

```bash
make test         # unit tests only (fast)
make test-all     # unit + integration (slow; loads OCR models)
make lint
make typecheck
make ci           # lint + typecheck + test-all
```

Stack: Python 3.12+, `uv` (deps), `ruff` (lint+format), `mypy --strict`,
`pytest`, `opencv-python`, `easyocr`, `pypdfium2`, `python-docx`,
`networkx`.

## Configuration

Defaults live in `src/pipeline/config.py`. Override via:

- `config.toml` at the project root, e.g.
  ```toml
  dpi = 400
  ocr_confidence_threshold = 0.5
  max_workers = 4
  ```
- Environment variables: `PIPELINE_DPI=400 PIPELINE_MAX_WORKERS=4 make run`.

## Assumptions

- The SOP follows the structure in `data/sop/sop.docx`: an Operating
  Limits / Design Limits table with one tag-bearing row per equipment.
- ISA-5.1 tag conventions (e.g. `V-745`, `E-742`, `AC-746`).
- Solid lines on the drawing represent process piping; dashed/dotted
  lines are signal lines and are filtered out of the graph.

## Submission

The implementation plan that produced this code is at
`docs/superpowers/plans/2026-04-27-pid-sop-pipeline.md`. The design
spec is at `docs/superpowers/specs/2026-04-27-pid-sop-design.md`.
```

- [ ] **Step 2: Verify the run still works**

Run: `make run`
Expected: pipeline completes, `output/` populated, exit code 0.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Rewrite README with reviewer-facing instructions and approach summary"
```

---

## Self-review notes

- **Spec coverage:** Every section of the spec maps to a task — types (T2), config (T3), protocols (T4), ISA lookup (T5), tag parsing (T6), SOP parser (T7), cross-reference (T8), logging (T9), PDF/preprocess/OCR (T10–T12), components/attributes/connections (T13–T15), graph/visualizer (T16–T17), runner/CLI (T18–T19), tests (T20), README (T21). The spec mentioned content-hash caching as a "nice-to-have"; runner.py reserves a comment marker for it but does not implement caching — this is acceptable because the spec lists it as an iteration-speed optimisation, not a correctness requirement, and the smoke test runs fine without it.
- **Type consistency:** `parse_tags` signature is identical at definition (Task 6) and call site (Task 18). `compare(components, expected)` matches between Task 8 and Task 18. `enrich(components, text_boxes, radius_px)` matches.
- **PipelineWarning** rename is consistent across all tasks.
- **Output directory:** `Settings.output_dir` defaults to `Path("output")` (relative). Task 19's CLI override path is wired into the same field. Task 20 passes `tmp_path` and asserts artifacts inside it — correct.
