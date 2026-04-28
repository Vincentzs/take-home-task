from pipeline.isa_lookup import (
    EQUIPMENT_PREFIX_TO_TYPE,
    INSTRUMENT_ABBREVIATIONS,
    ISA_FIRST_LETTER,
    ISA_FUNCTION_LETTER,
    describe_instrument,
)
from pipeline.types import ComponentType


def test_first_letter_table_has_core_variables() -> None:
    assert ISA_FIRST_LETTER["P"] == "Pressure"
    assert ISA_FIRST_LETTER["T"] == "Temperature"
    assert ISA_FIRST_LETTER["F"] == "Flow"
    assert ISA_FIRST_LETTER["L"] == "Level"
    assert ISA_FIRST_LETTER["A"] == "Analysis"


def test_function_letter_table_has_core_modifiers() -> None:
    assert ISA_FUNCTION_LETTER["I"] == "Indicator"
    assert ISA_FUNCTION_LETTER["C"] == "Controller"
    assert ISA_FUNCTION_LETTER["T"] == "Transmitter"
    assert ISA_FUNCTION_LETTER["V"] == "Valve"
    assert ISA_FUNCTION_LETTER["S"] == "Switch"
    assert ISA_FUNCTION_LETTER["A"] == "Alarm"


def test_abbreviation_table_includes_common_examples() -> None:
    assert INSTRUMENT_ABBREVIATIONS["PCV"] == "Pressure Control Valve"
    assert INSTRUMENT_ABBREVIATIONS["FIC"] == "Flow Indicator Controller"
    assert INSTRUMENT_ABBREVIATIONS["ESD"] == "Emergency Shutdown Station"
    assert INSTRUMENT_ABBREVIATIONS["PSV"] == "Pressure Safety Valve"


def test_describe_known_abbreviation_uses_table() -> None:
    assert describe_instrument("PCV") == "Pressure Control Valve"


def test_describe_decomposes_unknown_abbreviation_via_letters() -> None:
    # PIT is not in our minimal abbrev set; must decompose.
    desc = describe_instrument("PIT")
    assert "Pressure" in desc
    assert "Indicator" in desc
    assert "Transmitter" in desc


def test_describe_returns_raw_for_unparseable() -> None:
    assert describe_instrument("XYZ") == "XYZ"


def test_equipment_prefix_map_includes_sop_tags() -> None:
    assert EQUIPMENT_PREFIX_TO_TYPE["V"] == ComponentType.VESSEL
    assert EQUIPMENT_PREFIX_TO_TYPE["E"] == ComponentType.HEAT_EXCHANGER
    assert EQUIPMENT_PREFIX_TO_TYPE["AC"] == ComponentType.AIR_COOLED_EXCHANGER
    assert EQUIPMENT_PREFIX_TO_TYPE["F"] == ComponentType.FILTER
    assert EQUIPMENT_PREFIX_TO_TYPE["P"] == ComponentType.PUMP
