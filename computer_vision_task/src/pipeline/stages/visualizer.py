"""Render an annotated PNG showing detected tags, components and edges."""

from __future__ import annotations

from pathlib import Path

import cv2

from pipeline.types import Component, ComponentType, Edge, PageImage

_TYPE_COLORS: dict[ComponentType, tuple[int, int, int]] = {
    ComponentType.VESSEL: (0, 0, 200),
    ComponentType.HEAT_EXCHANGER: (200, 100, 0),
    ComponentType.AIR_COOLED_EXCHANGER: (200, 150, 0),
    ComponentType.FILTER: (0, 150, 0),
    ComponentType.PUMP: (150, 0, 150),
    ComponentType.TANK: (100, 100, 0),
    ComponentType.INSTRUMENT: (50, 50, 50),
    ComponentType.UNKNOWN: (0, 0, 0),
}


def render(
    page: PageImage,
    components: list[Component],
    edges: list[Edge],
    out_path: Path,
) -> None:
    canvas = page.image.copy()
    for component in components:
        x0, y0, x1, y1 = component.bbox
        color = _TYPE_COLORS.get(component.type, (0, 0, 0))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        label = f"{component.tag} [{component.type.value}]"
        cv2.putText(
            canvas,
            label,
            (x0, max(15, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    for edge in edges:
        x0, y0, x1, y1 = edge.evidence
        cv2.line(canvas, (x0, y0), (x1, y1), (255, 0, 0), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
