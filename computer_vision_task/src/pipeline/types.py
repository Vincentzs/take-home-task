"""Domain types for the P&ID + SOP cross-reference pipeline.

All types are frozen dataclasses with __slots__ for cheap equality and
JSON-roundtrip-friendly fields. No I/O; no third-party deps beyond numpy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

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
    image: np.ndarray[Any, np.dtype[Any]]
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
