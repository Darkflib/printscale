"""The verification helpers."""

from __future__ import annotations

import numpy as np
import pytest

from printscale import Array
from printscale.metrics import clipping, detect_crop, ppi_table, radial_psd, seam_ratio


def test_seam_ratio_is_flat_on_a_clean_image(smooth_rgb: Array) -> None:
    gray = (smooth_rgb[..., 0] * 255).astype(np.uint8)
    vertical, horizontal = seam_ratio(gray, period=64)
    assert vertical < 1.5
    assert horizontal < 1.5


def test_seam_ratio_detects_planted_seams() -> None:
    gray = np.full((256, 256), 120, dtype=np.uint8)
    gray[:, 64::64] = 200  # a bright line at exactly the tile pitch
    vertical, _ = seam_ratio(gray, period=64)
    assert vertical > 2.0


def test_seam_ratio_handles_degenerate_period(smooth_rgb: Array) -> None:
    gray = (smooth_rgb[..., 0] * 255).astype(np.uint8)
    assert seam_ratio(gray, period=0) == (1.0, 1.0)
    assert seam_ratio(gray, period=10_000) == (1.0, 1.0)


def test_clipping_counts_both_ends() -> None:
    image = np.full((100, 100), 128, dtype=np.uint8)
    image[:10, :] = 0
    image[-5:, :] = 255
    shadow, highlight = clipping(image)
    assert shadow == pytest.approx(0.10, abs=1e-6)
    assert highlight == pytest.approx(0.05, abs=1e-6)


def test_clipping_accepts_sixteen_bit() -> None:
    image = np.full((50, 50), 30_000, dtype=np.uint16)
    shadow, highlight = clipping(image)
    assert shadow == 0.0
    assert highlight == 0.0


def test_radial_psd_shows_more_energy_in_a_noisier_image() -> None:
    rng = np.random.default_rng(4)
    size = 256
    base = np.full((size, size), 0.5, dtype=np.float32)
    quiet = base + rng.standard_normal((size, size)).astype(np.float32) * 0.001
    loud = base + rng.standard_normal((size, size)).astype(np.float32) * 0.02
    high_band = slice(size // 8, size // 4)
    assert (
        radial_psd(loud, size=size, windows=1)[high_band].mean()
        > radial_psd(quiet, size=size, windows=1)[high_band].mean()
    )


def test_ppi_table_maths() -> None:
    rows = ppi_table(5436, widths_mm=(800.0, 1500.0))
    assert rows[0][1] == pytest.approx(172.5, abs=0.5)
    assert rows[1][1] == pytest.approx(92.0, abs=0.5)
    assert rows[0][2] == 0  # comfortable
    assert rows[1][2] == 2  # mural territory


def test_detect_crop_finds_a_bordered_photo() -> None:
    canvas = np.full((400, 500), 250, dtype=np.uint8)  # paper border
    canvas[60:340, 80:420] = 90  # the photo
    x0, y0, x1, y1 = detect_crop(canvas)
    assert 75 <= x0 <= 90
    assert 55 <= y0 <= 70
    assert 410 <= x1 <= 425
    assert 330 <= y1 <= 345


def test_detect_crop_on_a_borderless_image() -> None:
    canvas = np.full((200, 200), 90, dtype=np.uint8)
    x0, y0, x1, y1 = detect_crop(canvas)
    assert (x1 - x0, y1 - y0) == (193, 193)
