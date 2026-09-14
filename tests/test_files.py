"""Writing the deliverables, and the DPI arithmetic that decides print size."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from printscale.files import dpi_for_width, load_rgb, save_jpeg, save_master


def test_dpi_for_width() -> None:
    assert dpi_for_width(5436, 800.0) == 173
    assert dpi_for_width(5436, 1200.0) == 115
    with pytest.raises(ValueError, match="positive"):
        dpi_for_width(1000, 0.0)


def test_master_is_sixteen_bit_and_stamped(tmp_path: Path) -> None:
    gray = np.linspace(0, 1, 256 * 128, dtype=np.float32).reshape(128, 256)
    path = save_master(gray, tmp_path / "master.tif", width_mm=100.0)
    with Image.open(path) as image:
        assert image.mode == "I;16"
        assert image.size == (256, 128)
        assert image.info["dpi"][0] == pytest.approx(dpi_for_width(256, 100.0), abs=1)


def test_master_compression_is_lossless(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    gray = rng.random((64, 96), dtype=np.float32)
    path = save_master(gray, tmp_path / "m.tif", width_mm=50.0, compress=True)
    with Image.open(path) as image:
        reloaded = np.array(image)
    expected = (np.clip(gray, 0, 1) * 65535).round().astype(np.uint16)
    np.testing.assert_array_equal(reloaded, expected)


def test_master_rejects_colour(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="single-channel"):
        save_master(np.zeros((8, 8, 3), dtype=np.float32), tmp_path / "x.tif", 100.0)


def test_jpeg_preview_round_trips(tmp_path: Path) -> None:
    gray = np.full((64, 64), 0.5, dtype=np.float32)
    path = save_jpeg(gray, tmp_path / "preview.jpg", width_mm=100.0)
    assert path.stat().st_size > 0
    loaded = load_rgb(path)
    assert loaded.shape == (64, 64, 3)
    assert abs(float(loaded.mean()) - 0.5) < 0.02


def test_load_rgb_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="could not read"):
        load_rgb(tmp_path / "nope.png")
