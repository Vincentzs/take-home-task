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
