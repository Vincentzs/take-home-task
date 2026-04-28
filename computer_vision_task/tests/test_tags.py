from pipeline.stages.tags import parse_tags
from pipeline.types import EquipmentTag, InstrumentTag, TextBox


def _tb(text: str, x: int = 0, y: int = 0) -> TextBox:
    return TextBox(text=text, bbox=(x, y, x + 50, y + 20), confidence=0.9)


def test_parse_equipment_tag_with_suffix() -> None:
    tags, warnings = parse_tags([_tb("F-715A")], page=0)
    assert len(tags) == 1
    eq = tags[0]
    assert isinstance(eq, EquipmentTag)
    assert eq.raw == "F-715A"
    assert eq.prefix == "F"
    assert eq.number == 715
    assert eq.suffix == "A"
    assert warnings == []


def test_parse_equipment_tag_without_suffix() -> None:
    tags, _ = parse_tags([_tb("V-745")], page=0)
    assert len(tags) == 1
    eq = tags[0]
    assert isinstance(eq, EquipmentTag)
    assert eq.suffix is None


def test_parse_equipment_tag_with_two_letter_prefix() -> None:
    tags, _ = parse_tags([_tb("AC-746")], page=0)
    assert len(tags) == 1
    eq = tags[0]
    assert isinstance(eq, EquipmentTag)
    assert eq.prefix == "AC"
    assert eq.number == 746


def test_parse_pairs_letters_above_number_into_instrument_tag() -> None:
    boxes = [_tb("PIC", x=100, y=100), _tb("02", x=100, y=125)]
    tags, _ = parse_tags(boxes, page=0)
    assert len(tags) == 1
    inst = tags[0]
    assert isinstance(inst, InstrumentTag)
    assert inst.letters == "PIC"
    assert inst.number == "02"
    assert inst.description == "Pressure Indicator Controller"


def test_unpaired_instrument_letters_warn_and_drop() -> None:
    tags, warnings = parse_tags([_tb("PIC", x=100, y=100)], page=0)
    assert tags == []
    assert any(w.code == "unpaired_instrument_letters" for w in warnings)


def test_unknown_equipment_prefix_warns_but_keeps_tag() -> None:
    tags, warnings = parse_tags([_tb("ZZ-999")], page=0)
    assert len(tags) == 1
    assert any(w.code == "unknown_prefix" for w in warnings)


def test_text_that_does_not_match_any_pattern_is_ignored() -> None:
    tags, warnings = parse_tags([_tb("SOME LABEL"), _tb("123 mm")], page=0)
    assert tags == []
    assert warnings == []  # silent — ordinary text, not a tag candidate


def test_low_confidence_tag_emits_warning_but_still_parses() -> None:
    boxes = [TextBox(text="V-745", bbox=(0, 0, 50, 20), confidence=0.3)]
    tags, warnings = parse_tags(boxes, page=0, min_confidence=0.5)
    assert tags == []
    assert any(w.code == "low_confidence_tag" for w in warnings)


def test_underscore_in_equipment_tag_is_normalized() -> None:
    tags, _ = parse_tags([_tb("AC_746")], page=0)
    assert len(tags) == 1
    eq = tags[0]
    assert isinstance(eq, EquipmentTag)
    assert eq.raw == "AC-746"
    assert eq.prefix == "AC"


def test_lowercase_equipment_tag_is_normalized() -> None:
    tags, _ = parse_tags([_tb("e-742")], page=0)
    assert len(tags) == 1
    eq = tags[0]
    assert isinstance(eq, EquipmentTag)
    assert eq.raw == "E-742"


def test_split_prefix_and_number_are_merged() -> None:
    boxes = [
        TextBox(text="F", bbox=(10, 100, 30, 120), confidence=0.9),
        TextBox(text="715A", bbox=(50, 102, 110, 122), confidence=0.85),
    ]
    tags, _ = parse_tags(boxes, page=0)
    assert len(tags) == 1
    eq = tags[0]
    assert isinstance(eq, EquipmentTag)
    assert eq.raw == "F-715A"
    assert eq.prefix == "F"
    assert eq.number == 715
    assert eq.suffix == "A"


def test_split_prefix_does_not_merge_too_far_apart() -> None:
    boxes = [
        TextBox(text="F", bbox=(10, 100, 30, 120), confidence=0.9),
        TextBox(text="715A", bbox=(200, 102, 260, 122), confidence=0.85),  # >80px gap
    ]
    tags, _ = parse_tags(boxes, page=0)
    # Should NOT merge — gap is too large; orphan number is dropped silently
    assert tags == []
