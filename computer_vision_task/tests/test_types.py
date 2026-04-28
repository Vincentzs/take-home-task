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


def test_textbox_is_frozen_and_carries_confidence() -> None:
    tb = TextBox(text="V-745", bbox=(10, 20, 30, 40), confidence=0.95)
    assert tb.text == "V-745"
    assert tb.bbox == (10, 20, 30, 40)
    assert tb.confidence == 0.95


def test_equipment_tag_round_trip() -> None:
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


def test_instrument_tag_describes_isa_letters() -> None:
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


def test_component_attrs_default_empty() -> None:
    c = Component(tag="V-745", type=ComponentType.VESSEL, page=0, bbox=(0, 0, 10, 10))
    assert c.attrs == {}


def test_discrepancy_carries_kind_and_message() -> None:
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


def test_pipeline_warning_carries_stage_and_code() -> None:
    w = PipelineWarning(stage="ocr", page=0, code="no_text_found", message="blank")
    assert w.stage == "ocr"
    assert w.code == "no_text_found"


def test_report_summary_counts() -> None:
    r = Report(discrepancies=(), warnings=(), summary={"missing_in_diagram": 0})
    assert r.summary["missing_in_diagram"] == 0


def test_tag_kind_and_component_type_are_str_enums() -> None:
    # StrEnum members are equal to their string values at runtime; cast to str
    # so mypy doesn't flag the comparison as a non-overlapping literal check.
    assert str(TagKind.EQUIPMENT) == "equipment"
    assert str(ComponentType.VESSEL) == "vessel"
    assert isinstance(TagKind.EQUIPMENT, str)
    assert isinstance(ComponentType.VESSEL, str)


def test_module_exports_full_public_surface() -> None:
    # Smoke-check that every imported symbol is reachable; future tasks will
    # exercise these in depth.
    assert BBox.__origin__ is tuple  # type: ignore[attr-defined]
    assert PageImage.__name__ == "PageImage"
    assert Edge.__name__ == "Edge"
    assert ExpectedComponent.__name__ == "ExpectedComponent"
