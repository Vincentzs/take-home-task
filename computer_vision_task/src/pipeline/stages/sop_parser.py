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
        _p_min, p_max, p_ok = _parse_value(p_raw)
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
