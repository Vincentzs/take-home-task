"""Rasterize a PDF into a list of PageImages at a specified DPI.

Uses pypdfium2 for portability (no system poppler dependency).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pypdfium2 as pdfium

from pipeline.types import PageImage


def load(pdf_path: Path, dpi: int) -> list[PageImage]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    scale = dpi / 72.0
    document = pdfium.PdfDocument(str(pdf_path))
    pages: list[PageImage] = []
    try:
        for idx, page in enumerate(document):
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil().convert("RGB")
            arr = np.asarray(pil_image)
            pages.append(PageImage(page_idx=idx, image=arr, dpi=dpi))
            page.close()
    finally:
        document.close()
    if not pages:
        raise ValueError(f"PDF has zero pages: {pdf_path}")
    return pages
