"""Image preprocessing for OCR and line detection.

Produces (a) a binarised image suitable for OCR/skeletonisation, and
(b) keeps the original RGB for visualization. Both share a deskew step.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from pipeline.types import PageImage


def _deskew_angle(gray: np.ndarray[Any, np.dtype[Any]]) -> float:
    """Estimate skew angle in degrees by analysing dominant edge orientation."""
    inverted = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(inverted > 0))
    if coords.size == 0:
        return 0.0
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    return float(angle)


def _rotate(image: np.ndarray[Any, np.dtype[Any]], angle: float) -> np.ndarray[Any, np.dtype[Any]]:
    if abs(angle) < 0.1:
        return image
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def clean(page: PageImage) -> tuple[PageImage, np.ndarray[Any, np.dtype[Any]]]:
    """Return (deskewed RGB PageImage, binarised single-channel image)."""
    rgb = page.image
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    angle = _deskew_angle(gray)
    rgb_rot = _rotate(rgb, angle)
    gray_rot = cv2.cvtColor(rgb_rot, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(
        gray_rot,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=25,
        C=15,
    )
    cleaned_page = PageImage(page_idx=page.page_idx, image=rgb_rot, dpi=page.dpi)
    return cleaned_page, binary
