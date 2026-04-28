"""EasyOCR-backed implementation of OCREngine.

EasyOCR loads a PyTorch model on first instantiation (~100MB download to
~/.EasyOCR/ on first run). The Reader is created lazily so unit tests
that don't exercise OCR don't pay this cost.
"""

from __future__ import annotations

from typing import Any

import easyocr
import numpy as np

from pipeline.types import BBox, PipelineWarning, TextBox


class EasyOCREngine:
    def __init__(self, languages: tuple[str, ...] = ("en",), gpu: bool = False) -> None:
        self._languages = list(languages)
        self._gpu = gpu
        self._reader: easyocr.Reader | None = None

    @property
    def reader(self) -> easyocr.Reader:
        if self._reader is None:
            self._reader = easyocr.Reader(self._languages, gpu=self._gpu, verbose=False)
        return self._reader

    def read(self, image: np.ndarray[Any, np.dtype[Any]]) -> list[TextBox]:
        raw = self.reader.readtext(image, detail=1, paragraph=False)
        out: list[TextBox] = []
        for entry in raw:
            polygon, text, confidence = entry
            xs = [int(p[0]) for p in polygon]
            ys = [int(p[1]) for p in polygon]
            bbox: BBox = (min(xs), min(ys), max(xs), max(ys))
            out.append(
                TextBox(
                    text=str(text).strip(),
                    bbox=bbox,
                    confidence=float(confidence),
                )
            )
        return out


def run_ocr(
    image: np.ndarray[Any, np.dtype[Any]],
    engine: EasyOCREngine,
    page: int,
) -> tuple[list[TextBox], list[PipelineWarning]]:
    boxes = engine.read(image)
    warnings: list[PipelineWarning] = []
    if not boxes:
        warnings.append(
            PipelineWarning(
                stage="ocr",
                page=page,
                code="no_text_found",
                message="OCR returned zero text boxes",
            )
        )
    return boxes, warnings
