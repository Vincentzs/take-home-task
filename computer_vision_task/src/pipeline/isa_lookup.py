"""ISA-5.1 instrument tag conventions and equipment prefix mapping.

Sources:
- ISA-5.1-2009 Instrumentation Symbols and Identification, summarised in
  Kimray's "How to Read an Oil & Gas P&ID" reference guide.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

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
    variable = ISA_FIRST_LETTER[letters[0]]
    if variable in {"User's Choice", "Unclassified"}:
        return letters
    parts: list[str] = [variable]
    for char in letters[1:]:
        if char in ISA_FUNCTION_LETTER:
            parts.append(ISA_FUNCTION_LETTER[char])
        else:
            return letters
    return " ".join(parts)
