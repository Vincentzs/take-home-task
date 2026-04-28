"""Compare extracted graph (Components) to SOP expectations.

Emits Discrepancy records per the rules in the spec:
- MISSING_IN_DIAGRAM: SOP names a tag absent from the graph
- EXTRA_IN_DIAGRAM: equipment tag in graph but not in SOP
- TYPE_MISMATCH: SOP type != classified type
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


def _aggregate_expected(
    expected: Iterable[ExpectedComponent],
) -> dict[str, list[ExpectedComponent]]:
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

        # TYPE_MISMATCH (use the first expected row's type - shell/tube share a type)
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
                    message=(
                        f"{tag}: SOP says {expected_type.value}, P&ID says {component.type.value}"
                    ),
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
                            f"!= SOP {row.design_pressure_psig} psig"
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
                            f"{tag}{side_label}: P&ID design temperature {actual_temp} F "
                            f"outside SOP range "
                            f"{row.design_temp_f_min}..{row.design_temp_f_max} F"
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


def render_report(
    report: Report,
    extra_warnings: list[PipelineWarning] | None = None,
) -> str:
    """Render a Report as Markdown for output/report.md."""
    extra = extra_warnings or []
    all_warnings = list(report.warnings) + extra
    lines: list[str] = ["# P&ID / SOP Cross-Reference Report", ""]

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
