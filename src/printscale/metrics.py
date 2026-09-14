"""Cheap numerical checks. Each catches a specific failure.

* :func:`seam_ratio` — a bad tile blend
* :func:`clipping` — a crushed tonal range
* :func:`radial_psd` — quantifies what the model actually added
* :func:`ppi_table` — keeps the print-size decision on numbers
"""

from __future__ import annotations

import logging

import numpy as np

from printscale._types import Array

LOG = logging.getLogger(__name__)

__all__ = ["clipping", "detect_crop", "ppi_table", "radial_psd", "seam_ratio"]


MM_PER_INCH = 25.4


def seam_ratio(gray: Array, period: int, neighbourhood: int = 8) -> tuple[float, float]:
    """Peak gradient energy at the tile pitch, relative to its surroundings.

    Returns (vertical, horizontal). 1.0 means no detectable seams; anything
    above roughly 1.5 means the blend is wrong.

    Args:
        period: expected seam spacing in output pixels (``step * scale``).
    """
    values = gray.astype(np.float32)
    columns = np.abs(np.diff(values, axis=1)).mean(axis=0)
    rows = np.abs(np.diff(values, axis=0)).mean(axis=1)

    def score(energy: Array) -> float:
        if period <= 0 or period >= len(energy):
            return 1.0
        indices = np.arange(period, len(energy), period)
        if not len(indices):
            return 1.0
        local = energy[indices]
        base = np.array(
            [energy[max(0, i - neighbourhood) : i + neighbourhood].mean() for i in indices]
        )
        return float(np.max(local / np.maximum(base, 1e-6)))

    return score(columns), score(rows)


def clipping(image: Array, low: int = 2, high: int = 253) -> tuple[float, float]:
    """Fraction of pixels at each end of the range. Returns (shadow, highlight).

    Above roughly 0.005 either way, pull the contrast back.
    """
    eight_bit = image if image.dtype == np.uint8 else (image / 257.0).astype(np.uint8)
    return float((eight_bit <= low).mean()), float((eight_bit >= high).mean())


def radial_psd(gray: Array, size: int = 1024, windows: int = 5, seed: int = 0) -> Array:
    """Radially averaged power spectral density, averaged over several windows.

    Hann-tapered to stop edge discontinuities leaking broadband energy into the
    result. Index i corresponds to i/size cycles per pixel.

    Compare a model output against a Lanczos enlargement of the same source:
    everything above the source Nyquist (0.25 cyc/px after a 2x enlargement) is
    what the model synthesised. Expect 15-20 dB.
    """
    height, width = gray.shape
    if height < size or width < size:
        size = min(height, width, size)
    taper = np.outer(np.hanning(size), np.hanning(size)).astype(np.float32)

    rng = np.random.default_rng(seed)
    yy, xx = np.indices((size, size))
    radius = np.hypot(yy - size // 2, xx - size // 2).astype(int)
    counts = np.bincount(radius.ravel())

    total = np.zeros(counts.shape, dtype=np.float64)
    for _ in range(windows):
        y = int(rng.integers(0, height - size + 1))
        x = int(rng.integers(0, width - size + 1))
        block = gray[y : y + size, x : x + size].astype(np.float32)
        block = (block - block.mean()) * taper
        power = np.abs(np.fft.fftshift(np.fft.fft2(block))) ** 2
        total += np.bincount(radius.ravel(), power.ravel()) / np.maximum(counts, 1)

    return (total / windows)[: size // 2]


def ppi_table(
    pixel_width: int, widths_mm: tuple[float, ...] = (594, 841, 1000, 1200, 1500)
) -> list[tuple[float, float, int]]:
    """(width_mm, height_mm_unknown_placeholder, ppi) for a set of print widths.

    Returns a list of (width_mm, ppi, verdict_index) where verdict_index is
    0 comfortable, 1 fine, 2 mural, 3 soft — so callers can render their own
    wording. Thresholds assume a print viewed at arm's length.
    """
    rows: list[tuple[float, float, int]] = []
    for width_mm in widths_mm:
        ppi = pixel_width / (width_mm / MM_PER_INCH)
        if ppi >= 150:
            verdict = 0
        elif ppi >= 100:
            verdict = 1
        elif ppi >= 80:
            verdict = 2
        else:
            verdict = 3
        rows.append((width_mm, ppi, verdict))
    return rows


def detect_crop(gray: Array, threshold: int = 200, margin: int = 3) -> tuple[int, int, int, int]:
    """Find the photo rectangle inside a scan that has a paper border.

    Scans of press prints carry white borders, credit lines and logos. Returns
    (x0, y0, x1, y1) of the largest contiguous darker-than-`threshold` run
    through the centre of the frame.

    This is a convenience, not a guarantee — check the result before relying
    on it for anything irreversible.
    """
    height, width = gray.shape
    band_rows = gray[height // 6 : 5 * height // 6, :]
    band_cols = gray[:, width // 6 : 5 * width // 6]
    col_mean = band_rows.mean(axis=0)
    row_mean = band_cols.mean(axis=1)

    def run(profile: Array, centre: int) -> tuple[int, int]:
        inside = set(np.where(profile < threshold)[0].tolist())
        if centre not in inside:
            return 0, len(profile) - 1
        start = end = centre
        while start - 1 in inside:
            start -= 1
        while end + 1 in inside:
            end += 1
        return start, end

    x0, x1 = run(col_mean, width // 2)
    y0, y1 = run(row_mean, height // 2)
    return x0 + margin, y0 + margin, x1 - margin, y1 - margin
