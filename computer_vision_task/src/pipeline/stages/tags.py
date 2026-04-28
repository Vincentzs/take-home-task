"""Tag parsing: TextBoxes -> equipment tags + instrument tags + warnings.

Two regex families per ISA-5.1:
- Equipment tags: <PREFIX>-<NUMBER><SUFFIX?>  e.g. F-715A, V-745, AC-746
- Instrument tags: a letters box stacked above a number box, both inside
  a circle. We pair them by horizontal alignment and vertical adjacency.
"""

from __future__ import annotations

import re

from pipeline.isa_lookup import (
    EQUIPMENT_PREFIX_TO_TYPE,
    ISA_FIRST_LETTER,
    describe_instrument,
)
from pipeline.types import (
    BBox,
    EquipmentTag,
    InstrumentTag,
    PipelineWarning,
    Tag,
    TextBox,
)

EQUIPMENT_RE = re.compile(r"^([A-Z]{1,3})-(\d{2,4})([A-Z])?$")
INSTRUMENT_LETTERS_RE = re.compile(r"^[A-Z]{2,4}$")
INSTRUMENT_NUMBER_RE = re.compile(r"^\d{2,3}$")
ORPHAN_NUMBER_RE = re.compile(r"^(\d{2,4})([A-Z])?$")
ORPHAN_PREFIX_RE = re.compile(r"^[A-Z]{1,3}$")
# Loose equipment regex tolerant of common OCR digit confusions (Z->7, O->0,
# I/L->1, S->5, B->8) inside the digit slot. Used as a fallback when the
# strict regex fails. Non-greedy so a trailing A/B is preferred as the suffix
# rather than being absorbed into the digit slot.
EQUIPMENT_RE_LOOSE = re.compile(r"^([A-Z]{1,3})-([0-9OIZSBL]{2,4}?)([A-Z])?$")
# Same fallback but for tokens whose prefix letter was eaten by OCR (e.g.
# "_Z15B" -> "-Z15B" after underscore normalization). The prefix is recovered
# later by matching against equipment tags already detected on the same page.
EQUIPMENT_RE_NO_PREFIX = re.compile(r"^-([0-9OIZSBL]{2,4}?)([A-Z])?$")
_DIGIT_FIX = str.maketrans({"O": "0", "I": "1", "L": "1", "Z": "7", "S": "5", "B": "8"})

_DEFAULT_PAIRING_X_TOL = 30  # px: max horizontal centre offset for pairing
_DEFAULT_PAIRING_Y_GAP = 40  # px: max vertical gap between letters and number


def _bbox_centre(bbox: BBox) -> tuple[int, int]:
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) // 2, (y0 + y1) // 2)


def _try_equipment(tb: TextBox, page: int) -> tuple[EquipmentTag, bool] | None:
    """Parse a single TextBox into an EquipmentTag.

    Returns ``(tag, is_loose)`` where ``is_loose`` is True when the tag was
    recovered via the digit-confusion fallback (caller may want to keep it
    even if confidence is low, since OCR-corrupt tokens are inherently noisy).
    """
    cleaned = tb.text.replace(" ", "").replace("_", "-").upper()
    m = EQUIPMENT_RE.match(cleaned)
    if m is not None:
        prefix, number_s, suffix = m.groups()
        return (
            EquipmentTag(
                raw=cleaned,
                prefix=prefix,
                number=int(number_s),
                suffix=suffix,
                bbox=tb.bbox,
                page=page,
            ),
            False,
        )
    # Fallback: tolerate OCR digit confusion in the number slot (e.g. "F-Z15A"
    # produced by EasyOCR misreading "F-715A").
    m_loose = EQUIPMENT_RE_LOOSE.match(cleaned)
    if m_loose is None:
        return None
    prefix, number_s_raw, suffix = m_loose.groups()
    number_s_fixed = number_s_raw.translate(_DIGIT_FIX)
    if not number_s_fixed.isdigit():
        return None
    raw_fixed = f"{prefix}-{number_s_fixed}{suffix or ''}"
    return (
        EquipmentTag(
            raw=raw_fixed,
            prefix=prefix,
            number=int(number_s_fixed),
            suffix=suffix,
            bbox=tb.bbox,
            page=page,
        ),
        True,
    )


def _pair_instrument(
    letters_tb: TextBox,
    candidates: list[TextBox],
    x_tol: int,
    y_gap: int,
) -> TextBox | None:
    lx, _ly = _bbox_centre(letters_tb.bbox)
    letters_y1 = letters_tb.bbox[3]
    best: tuple[int, TextBox] | None = None
    for tb in candidates:
        if not INSTRUMENT_NUMBER_RE.match(tb.text):
            continue
        cx, _cy = _bbox_centre(tb.bbox)
        if abs(cx - lx) > x_tol:
            continue
        gap = tb.bbox[1] - letters_y1
        if gap < 0 or gap > y_gap:
            continue
        if best is None or gap < best[0]:
            best = (gap, tb)
    return best[1] if best is not None else None


def parse_tags(
    text_boxes: list[TextBox],
    page: int,
    min_confidence: float = 0.0,
    x_tol: int = _DEFAULT_PAIRING_X_TOL,
    y_gap: int = _DEFAULT_PAIRING_Y_GAP,
) -> tuple[list[Tag], list[PipelineWarning]]:
    """Parse equipment and instrument tags from a list of OCR TextBoxes."""
    tags: list[Tag] = []
    warnings: list[PipelineWarning] = []
    consumed: set[int] = set()

    # First pass: equipment tags.
    for i, tb in enumerate(text_boxes):
        result = _try_equipment(tb, page)
        if result is None:
            continue
        eq, is_loose = result
        # Loose-recovery tags (digit-confusion fallback) are inherently noisy,
        # so we keep them even when below the confidence threshold but emit a
        # warning. Strict matches still get dropped on low confidence.
        if tb.confidence < min_confidence:
            warnings.append(
                PipelineWarning(
                    stage="tags",
                    page=page,
                    code="low_confidence_tag",
                    message=f"{tb.text} confidence={tb.confidence:.2f}",
                )
            )
            if not is_loose:
                consumed.add(i)
                continue
        if eq.prefix not in EQUIPMENT_PREFIX_TO_TYPE:
            warnings.append(
                PipelineWarning(
                    stage="tags",
                    page=page,
                    code="unknown_prefix",
                    message=f"{eq.raw} prefix={eq.prefix}",
                )
            )
        tags.append(eq)
        consumed.add(i)

    # Second pass: instrument letter boxes pair with adjacent number boxes.
    for i, tb in enumerate(text_boxes):
        if i in consumed:
            continue
        if not INSTRUMENT_LETTERS_RE.match(tb.text):
            continue
        if tb.confidence < min_confidence:
            warnings.append(
                PipelineWarning(
                    stage="tags",
                    page=page,
                    code="low_confidence_tag",
                    message=f"{tb.text} confidence={tb.confidence:.2f}",
                )
            )
            consumed.add(i)
            continue
        partner = _pair_instrument(
            tb,
            [text_boxes[j] for j in range(len(text_boxes)) if j not in consumed and j != i],
            x_tol=x_tol,
            y_gap=y_gap,
        )
        if partner is None:
            warnings.append(
                PipelineWarning(
                    stage="tags",
                    page=page,
                    code="unpaired_instrument_letters",
                    message=tb.text,
                )
            )
            consumed.add(i)
            continue
        merged_bbox: BBox = (
            min(tb.bbox[0], partner.bbox[0]),
            min(tb.bbox[1], partner.bbox[1]),
            max(tb.bbox[2], partner.bbox[2]),
            max(tb.bbox[3], partner.bbox[3]),
        )
        first = tb.text[0]
        functions = tuple(tb.text[1:])
        tags.append(
            InstrumentTag(
                letters=tb.text,
                number=partner.text,
                variable=ISA_FIRST_LETTER.get(first, first),
                functions=functions,
                description=describe_instrument(tb.text),
                bbox=merged_bbox,
                page=page,
            )
        )
        consumed.add(i)
        consumed.add(text_boxes.index(partner))

    # Third pass: orphan-number + adjacent-prefix recovery
    for i, tb in enumerate(text_boxes):
        if i in consumed:
            continue
        cleaned = tb.text.replace(" ", "").replace("_", "-").upper()
        m_num = ORPHAN_NUMBER_RE.match(cleaned)
        if m_num is None:
            continue
        number_s, suffix = m_num.groups()

        # Find a prefix box to the left, vertically aligned
        candidates: list[tuple[int, int, TextBox]] = []  # (distance, j, tb)
        for j, prefix_tb in enumerate(text_boxes):
            if j in consumed or j == i:
                continue
            prefix_clean = prefix_tb.text.replace(" ", "").replace("_", "-").upper()
            if not ORPHAN_PREFIX_RE.match(prefix_clean):
                continue
            # left of and vertically near
            if prefix_tb.bbox[2] >= tb.bbox[0]:
                continue  # not to the left
            horiz_gap = tb.bbox[0] - prefix_tb.bbox[2]
            if horiz_gap > 80:
                continue
            prefix_cy = (prefix_tb.bbox[1] + prefix_tb.bbox[3]) // 2
            number_cy = (tb.bbox[1] + tb.bbox[3]) // 2
            if abs(prefix_cy - number_cy) > 25:
                continue
            candidates.append((horiz_gap, j, prefix_tb))

        if not candidates:
            continue
        candidates.sort()
        _, j, prefix_tb = candidates[0]
        prefix_clean = prefix_tb.text.replace(" ", "").replace("_", "-").upper()
        min_conf = min(tb.confidence, prefix_tb.confidence)
        if min_conf < min_confidence:
            warnings.append(
                PipelineWarning(
                    stage="tags",
                    page=page,
                    code="low_confidence_tag",
                    message=f"{prefix_clean}-{number_s}{suffix or ''} confidence={min_conf:.2f}",
                )
            )
            consumed.add(i)
            consumed.add(j)
            continue
        raw = f"{prefix_clean}-{number_s}{suffix or ''}"
        if prefix_clean not in EQUIPMENT_PREFIX_TO_TYPE:
            warnings.append(
                PipelineWarning(
                    stage="tags",
                    page=page,
                    code="unknown_prefix",
                    message=f"{raw} prefix={prefix_clean}",
                )
            )
        merged_eq_bbox: BBox = (
            min(prefix_tb.bbox[0], tb.bbox[0]),
            min(prefix_tb.bbox[1], tb.bbox[1]),
            max(prefix_tb.bbox[2], tb.bbox[2]),
            max(prefix_tb.bbox[3], tb.bbox[3]),
        )
        tags.append(
            EquipmentTag(
                raw=raw,
                prefix=prefix_clean,
                number=int(number_s),
                suffix=suffix,
                bbox=merged_eq_bbox,
                page=page,
            )
        )
        consumed.add(i)
        consumed.add(j)

    # Fourth pass: prefix-eaten tags (e.g. EasyOCR returned "_Z15B" because the
    # leading "F" was lost). Recover the prefix by matching the digit stem
    # against equipment tags already detected on the same page; when multiple
    # candidates share the number, prefer the one most vertically aligned with
    # the orphan token (vessel A/B variants typically sit on the same row).
    for i, tb in enumerate(text_boxes):
        if i in consumed:
            continue
        cleaned = tb.text.replace(" ", "").replace("_", "-").upper()
        m_np = EQUIPMENT_RE_NO_PREFIX.match(cleaned)
        if m_np is None:
            continue
        number_s_raw, suffix = m_np.groups()
        number_s_fixed = number_s_raw.translate(_DIGIT_FIX)
        if not number_s_fixed.isdigit():
            continue
        number_int = int(number_s_fixed)
        siblings = [
            t
            for t in tags
            if isinstance(t, EquipmentTag) and t.page == page and t.number == number_int
        ]
        if not siblings:
            continue
        orphan_cy = (tb.bbox[1] + tb.bbox[3]) // 2
        siblings.sort(key=lambda t: abs((t.bbox[1] + t.bbox[3]) // 2 - orphan_cy))
        prefix_inferred = siblings[0].prefix
        raw = f"{prefix_inferred}-{number_s_fixed}{suffix or ''}"
        if tb.confidence < min_confidence:
            warnings.append(
                PipelineWarning(
                    stage="tags",
                    page=page,
                    code="low_confidence_tag",
                    message=f"{raw} confidence={tb.confidence:.2f}",
                )
            )
            consumed.add(i)
            continue
        tags.append(
            EquipmentTag(
                raw=raw,
                prefix=prefix_inferred,
                number=number_int,
                suffix=suffix,
                bbox=tb.bbox,
                page=page,
            )
        )
        consumed.add(i)

    return tags, warnings
