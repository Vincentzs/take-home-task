from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pipeline.stages.sop_parser import parse_sop
from pipeline.types import ComponentType

MakeSopDocx = Callable[..., Path]


def test_parses_simple_design_limits_table(make_sop_docx: MakeSopDocx) -> None:
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


def test_parses_e742_shell_and_tube_as_separate_rows(make_sop_docx: MakeSopDocx) -> None:
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


def test_parses_f715_a_and_b_as_two_filters(make_sop_docx: MakeSopDocx) -> None:
    docx = make_sop_docx(rows=[("F-715 A and B Particulate Filters", "275", "100")])
    expected, _ = parse_sop(docx)
    tags = sorted(e.tag for e in expected)
    assert tags == ["F-715A", "F-715B"]
    for e in expected:
        assert e.type == ComponentType.FILTER
        assert e.design_pressure_psig == 275.0


def test_unparseable_value_is_warned(make_sop_docx: MakeSopDocx) -> None:
    docx = make_sop_docx(rows=[("V-999 Vessel", "abc", "100")])
    expected, warnings = parse_sop(docx)
    assert any(w.code == "sop_value_unparseable" for w in warnings)
    # row still emitted, with pressure=None
    assert expected[0].design_pressure_psig is None


def test_missing_table_yields_empty_with_warning(
    make_sop_docx: MakeSopDocx, tmp_path: Path
) -> None:
    from docx import Document

    doc = Document()
    doc.add_paragraph("This SOP has no design-limits table.")
    out = tmp_path / "empty.docx"
    doc.save(str(out))
    expected, warnings = parse_sop(out)
    assert expected == []
    assert any(w.code == "sop_table_not_found" for w in warnings)
