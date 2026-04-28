"""Connection extraction: skeletonise -> Hough -> dash filter -> tag-pair edges.

Process pipes are solid lines. Signal lines (electrical, pneumatic,
hydraulic) are dashed/dotted per ISA-5.1; we filter those out so the
graph only carries process flow edges.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from pipeline.types import BBox, Component, Edge, PipelineWarning


def is_dashed_segment(line_pixels: np.ndarray[Any, np.dtype[Any]], min_solid_run: int) -> bool:
    """Return True iff the longest run of contiguous ink pixels is < min_solid_run.

    `line_pixels` is a 1-D uint8 array sampled along a candidate line.
    Ink is encoded as 0 (matching cv2 binary output convention).
    """
    if line_pixels.size == 0:
        return False
    ink = line_pixels == 0
    if not ink.any():
        return False
    longest = 0
    current = 0
    for v in ink:
        if v:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest < min_solid_run


def _sample_line(
    binary: np.ndarray[Any, np.dtype[Any]],
    p0: tuple[int, int],
    p1: tuple[int, int],
) -> np.ndarray[Any, np.dtype[Any]]:
    n = max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1])) + 1
    xs = np.linspace(p0[0], p1[0], n).astype(np.int32)
    ys = np.linspace(p0[1], p1[1], n).astype(np.int32)
    h, w = binary.shape[:2]
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    return binary[ys, xs]


def _segment_intersects_bbox(p0: tuple[int, int], p1: tuple[int, int], bbox: BBox) -> bool:
    x0, y0, x1, y1 = bbox
    if (p0[0] < x0 and p1[0] < x0) or (p0[0] > x1 and p1[0] > x1):
        return False
    return not ((p0[1] < y0 and p1[1] < y0) or (p0[1] > y1 and p1[1] > y1))


def _segment_bbox(p0: tuple[int, int], p1: tuple[int, int]) -> BBox:
    return (
        min(p0[0], p1[0]),
        min(p0[1], p1[1]),
        max(p0[0], p1[0]),
        max(p0[1], p1[1]),
    )


def detect(
    binary: np.ndarray[Any, np.dtype[Any]],
    components: list[Component],
    page: int,
    dash_solid_run_min_px: int,
    hough_threshold: int = 80,
    min_line_length: int = 60,
    max_line_gap: int = 10,
) -> tuple[list[Edge], list[PipelineWarning]]:
    warnings: list[PipelineWarning] = []
    inverted = cv2.bitwise_not(binary)
    raw = cv2.HoughLinesP(
        inverted,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if raw is None or len(raw) == 0:
        warnings.append(
            PipelineWarning(
                stage="connections",
                page=page,
                code="no_lines_detected",
                message="HoughLinesP returned no segments",
            )
        )
        return [], warnings

    edges: list[Edge] = []
    seen_pairs: set[tuple[str, str]] = set()
    for entry in raw:
        x1, y1, x2, y2 = entry[0]
        p0 = (int(x1), int(y1))
        p1 = (int(x2), int(y2))
        sample = _sample_line(binary, p0, p1)
        if is_dashed_segment(sample, min_solid_run=dash_solid_run_min_px):
            continue
        endpoints = [c for c in components if _segment_intersects_bbox(p0, p1, c.bbox)]
        if len(endpoints) < 2:
            continue
        endpoints = sorted(endpoints, key=lambda c: c.bbox)
        a, b = endpoints[0].tag, endpoints[1].tag
        if a == b:
            continue
        key: tuple[str, str] = (a, b) if a < b else (b, a)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        edges.append(Edge(src_tag=a, dst_tag=b, page=page, evidence=_segment_bbox(p0, p1)))
    return edges, warnings
