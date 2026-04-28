import numpy as np

from pipeline.stages.connections import is_dashed_segment


def test_solid_line_is_not_dashed() -> None:
    line = np.zeros(200, dtype=np.uint8)  # 0 = ink (line)
    assert is_dashed_segment(line, min_solid_run=8) is False


def test_dashed_line_is_dashed() -> None:
    # Alternating 4 ink, 4 background; runs all < 8
    line = np.tile(np.array([0, 0, 0, 0, 255, 255, 255, 255], dtype=np.uint8), 25)
    assert is_dashed_segment(line, min_solid_run=8) is True


def test_almost_solid_with_short_gap_is_not_dashed() -> None:
    line = np.zeros(200, dtype=np.uint8)
    line[100:103] = 255  # tiny 3-pixel gap; long ink runs on either side
    assert is_dashed_segment(line, min_solid_run=8) is False


def test_empty_or_pure_background_is_not_dashed() -> None:
    assert is_dashed_segment(np.full(50, 255, dtype=np.uint8), min_solid_run=8) is False
