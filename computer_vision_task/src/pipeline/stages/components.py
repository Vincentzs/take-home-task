"""Associate parsed Tags with symbol bounding boxes.

For equipment tags we look outward from the tag's bbox for a connected
contour region and use that as the symbol bbox. If none is found within
the search window, the bbox falls back to the tag's bbox and a warning
is emitted.

For instrument tags we use the tag's own bbox (the balloon circle) — the
balloon IS the symbol.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from pipeline.isa_lookup import EQUIPMENT_PREFIX_TO_TYPE
from pipeline.types import (
    BBox,
    Component,
    ComponentType,
    EquipmentTag,
    InstrumentTag,
    PipelineWarning,
    Tag,
)


def _expand_bbox(bbox: BBox, margin: int, image_shape: tuple[int, ...]) -> BBox:
    h, w = image_shape[0], image_shape[1]
    x0, y0, x1, y1 = bbox
    return (
        max(0, x0 - margin),
        max(0, y0 - margin),
        min(w, x1 + margin),
        min(h, y1 + margin),
    )


def _largest_contour_bbox(binary_crop: np.ndarray[Any, np.dtype[Any]]) -> BBox | None:
    inverted = cv2.bitwise_not(binary_crop)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50.0:
        return None
    x, y, w, h = cv2.boundingRect(largest)
    return (x, y, x + w, y + h)


def locate(
    tags: list[Tag],
    binary: np.ndarray[Any, np.dtype[Any]],
    page: int,
    search_margin_px: int = 80,
) -> tuple[list[Component], list[PipelineWarning]]:
    components: list[Component] = []
    warnings: list[PipelineWarning] = []
    for tag in tags:
        if isinstance(tag, InstrumentTag):
            components.append(
                Component(
                    tag=f"{tag.letters}-{tag.number}",
                    type=ComponentType.INSTRUMENT,
                    page=page,
                    bbox=tag.bbox,
                    attrs={"description": tag.description},
                )
            )
            continue

        assert isinstance(tag, EquipmentTag)
        symbol_type = EQUIPMENT_PREFIX_TO_TYPE.get(tag.prefix, ComponentType.UNKNOWN)
        search_bbox = _expand_bbox(tag.bbox, search_margin_px, binary.shape[:2])
        x0, y0, x1, y1 = search_bbox
        crop = binary[y0:y1, x0:x1]
        local = _largest_contour_bbox(crop)
        if local is None:
            warnings.append(
                PipelineWarning(
                    stage="components",
                    page=page,
                    code="symbol_not_localized",
                    message=tag.raw,
                )
            )
            symbol_bbox = tag.bbox
        else:
            lx0, ly0, lx1, ly1 = local
            symbol_bbox = (x0 + lx0, y0 + ly0, x0 + lx1, y0 + ly1)
        canonical_tag = f"{tag.prefix}-{tag.number}{tag.suffix or ''}"
        components.append(
            Component(
                tag=canonical_tag,
                type=symbol_type,
                page=page,
                bbox=symbol_bbox,
                attrs={},
            )
        )
    return components, warnings
