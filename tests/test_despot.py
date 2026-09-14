"""Dust removal, and the regression that matters: it must not eat fine detail.

The first version of this detector flagged 16,768 blobs on a real press scan
and started removing eyelashes, because an eyelash is also a bright structure
smaller than the structuring element. The local-detail gate is what fixed it.
These tests pin both halves: dust on flat tone goes, fine structure in busy
regions stays.
"""

from __future__ import annotations

import numpy as np
import pytest

from printscale.despot import despeckle, find_specks, local_detail


def _field_with_dust(seed: int = 0) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """A dark flat field with small bright specks at known positions."""
    rng = np.random.default_rng(seed)
    image = np.full((256, 256), 40, dtype=np.uint8)
    image = np.clip(image + rng.integers(-3, 4, image.shape), 0, 255).astype(np.uint8)
    positions = [(40, 50), (120, 80), (200, 190), (70, 210)]
    for y, x in positions:
        image[y - 1 : y + 2, x - 1 : x + 2] = 235
    return image, positions


def _hairs(seed: int = 1) -> np.ndarray:
    """A busy dark region full of thin bright strokes — a stand-in for hair."""
    rng = np.random.default_rng(seed)
    image = np.full((256, 256), 30, dtype=np.uint8)
    for _ in range(90):
        y = int(rng.integers(10, 246))
        x = int(rng.integers(10, 230))
        length = int(rng.integers(8, 22))
        image[y : y + 2, x : x + length] = 210
    return image


def test_local_detail_separates_flat_from_busy() -> None:
    flat, _ = _field_with_dust()
    busy = _hairs()
    assert float(np.median(local_detail(flat))) < float(np.median(local_detail(busy)))


def test_dust_on_flat_tone_is_removed() -> None:
    image, positions = _field_with_dust()
    repaired, mask = despeckle(image)
    assert mask.any(), "expected the detector to find the planted specks"
    for y, x in positions:
        assert int(repaired[y, x]) < 120, f"speck at ({x},{y}) survived"
    assert float((mask > 0).mean()) < 0.01


def test_fine_structure_in_busy_regions_survives() -> None:
    """The regression test. Without the detail gate this removes most strokes."""
    image = _hairs()
    repaired, mask = despeckle(image)
    touched = float((mask > 0).mean())
    assert touched < 0.002, f"detector touched {touched:.4%} of a hair field"
    bright_before = int((image > 150).sum())
    bright_after = int((repaired > 150).sum())
    assert bright_after > 0.95 * bright_before


def test_detail_gate_is_what_does_the_work() -> None:
    """Without the gate the detector fires inside the hair region too.

    A composite: dust on flat tone in the left half, hair in the right. The
    gate should keep the specks and drop everything on the right.
    """
    dusty, _ = _field_with_dust()
    hairy = _hairs()
    composite = np.hstack([dusty, hairy])

    gated = find_specks(composite, detail_limit=5.0, max_aspect=6.0)
    ungated = find_specks(composite, detail_limit=1e9, max_aspect=6.0)

    right = slice(dusty.shape[1], None)
    assert int((ungated[:, right] > 0).sum()) > int((gated[:, right] > 0).sum())
    assert int((gated[:, : dusty.shape[1]] > 0).sum()) > 0, "specks on flat tone were lost"


def test_sixteen_bit_path_preserves_dtype() -> None:
    image, positions = _field_with_dust()
    as_uint16 = image.astype(np.uint16) * 257
    repaired, mask = despeckle(as_uint16)
    assert repaired.dtype == np.uint16
    assert mask.any()
    for y, x in positions:
        assert int(repaired[y, x]) < 120 * 257


def test_no_specks_leaves_image_untouched() -> None:
    flat = np.full((128, 128), 128, dtype=np.uint8)
    repaired, mask = despeckle(flat)
    np.testing.assert_array_equal(repaired, flat)
    assert not mask.any()


def test_rejects_colour_input() -> None:
    with pytest.raises(ValueError, match="single-channel"):
        despeckle(np.zeros((16, 16, 3), dtype=np.uint8))


def test_rejects_bad_polarity() -> None:
    image, _ = _field_with_dust()
    with pytest.raises(ValueError, match="polarity"):
        find_specks(image, polarity="purple")
