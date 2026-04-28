"""Stage interfaces. Concrete implementations live under pipeline.stages.*

Stages depend on these Protocols, not concrete classes, so the
orchestrator can swap implementations without ripple effects.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from pipeline.types import BBox, ComponentType, TextBox


class OCREngine(Protocol):
    def read(self, image: np.ndarray[Any, np.dtype[Any]]) -> list[TextBox]: ...


class LineSegment(Protocol):
    @property
    def p0(self) -> tuple[int, int]: ...
    @property
    def p1(self) -> tuple[int, int]: ...
    @property
    def bbox(self) -> BBox: ...


class LineDetector(Protocol):
    def detect(self, image: np.ndarray[Any, np.dtype[Any]]) -> list[LineSegment]: ...


class SymbolClassifier(Protocol):
    def classify(self, tag_prefix: str) -> ComponentType: ...
