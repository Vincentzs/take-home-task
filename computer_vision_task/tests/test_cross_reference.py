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


def test_missing_in_diagram_when_sop_tag_absent() -> None:
    report = compare(
        components=[_comp("V-745", ComponentType.VESSEL)],
        expected=[_exp("V-745"), _exp("V-999")],
    )
    kinds = [d.kind for d in report.discrepancies]
    assert DiscrepancyKind.MISSING_IN_DIAGRAM in kinds
    missing = next(d for d in report.discrepancies if d.kind == DiscrepancyKind.MISSING_IN_DIAGRAM)
    assert missing.tag == "V-999"


def test_extra_in_diagram_only_for_equipment() -> None:
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


def test_type_mismatch_when_inferred_type_differs() -> None:
    report = compare(
        components=[_comp("V-745", ComponentType.PUMP)],
        expected=[_exp("V-745", ComponentType.VESSEL)],
    )
    mism = [d for d in report.discrepancies if d.kind == DiscrepancyKind.TYPE_MISMATCH]
    assert len(mism) == 1
    assert mism[0].expected == ComponentType.VESSEL
    assert mism[0].actual == ComponentType.PUMP


def test_pressure_mismatch_only_when_both_have_values() -> None:
    # Drawing has 250, SOP says 300 -> mismatch
    r1 = compare(
        components=[_comp("V-745", ComponentType.VESSEL, design_pressure_psig=250.0)],
        expected=[_exp("V-745", p=300.0)],
    )
    assert any(d.kind == DiscrepancyKind.PRESSURE_MISMATCH for d in r1.discrepancies)

    # Drawing has no value -> silent
    r2 = compare(
        components=[_comp("V-745", ComponentType.VESSEL)],
        expected=[_exp("V-745", p=300.0)],
    )
    assert not any(d.kind == DiscrepancyKind.PRESSURE_MISMATCH for d in r2.discrepancies)


def test_temperature_mismatch_uses_max_against_drawing_value() -> None:
    r = compare(
        components=[_comp("AC-746", ComponentType.AIR_COOLED_EXCHANGER, design_temp_f=350.0)],
        expected=[_exp("AC-746", t_min=-20.0, t_max=400.0)],
    )
    # Drawing reports 350, SOP allows -20..400 -> no mismatch
    assert not any(d.kind == DiscrepancyKind.TEMPERATURE_MISMATCH for d in r.discrepancies)


def test_e742_aggregates_shell_and_tube_into_one_node() -> None:
    report = compare(
        components=[_comp("E-742", ComponentType.HEAT_EXCHANGER)],
        expected=[
            _exp(
                "E-742",
                ComponentType.HEAT_EXCHANGER,
                side="shell",
                p=300.0,
                t_min=375.0,
                t_max=375.0,
            ),
            _exp(
                "E-742",
                ComponentType.HEAT_EXCHANGER,
                side="tube",
                p=300.0,
                t_min=250.0,
                t_max=250.0,
            ),
        ],
    )
    # Should NOT report MISSING_IN_DIAGRAM for E-742 just because there are two SOP rows
    missing = [d for d in report.discrepancies if d.kind == DiscrepancyKind.MISSING_IN_DIAGRAM]
    assert "E-742" not in {d.tag for d in missing}


def test_summary_counts_each_kind() -> None:
    report = compare(
        components=[],
        expected=[_exp("V-1"), _exp("V-2")],
    )
    assert report.summary[DiscrepancyKind.MISSING_IN_DIAGRAM.value] == 2


def test_render_report_has_required_sections() -> None:
    report = compare(components=[], expected=[_exp("V-1")])
    md = render_report(report)
    assert "## Summary" in md
    assert "## Discrepancies" in md
    assert "## Warnings" in md
