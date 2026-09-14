"""Grain measurement and the unsharp stages."""

from __future__ import annotations

import numpy as np
import pytest

from printscale import Array
from printscale.finishing import (
    add_grain,
    local_contrast,
    measure_grain,
    print_sharpen,
    to_neutral_gray,
)


def test_measure_grain_recovers_known_sigma(grainy_gray: tuple[Array, float]) -> None:
    image, sigma = grainy_gray
    estimate = measure_grain(image)
    # The high-pass keeps most but not all of the noise power, so the estimate
    # sits a little under the true sigma. Pin the ratio rather than the value.
    assert 0.45 * sigma < estimate < 1.2 * sigma


def test_measure_grain_ignores_detail() -> None:
    """A busy region must not inflate the estimate — the point of using the tail.

    The estimator needs *some* flat area to find. A picture that is edges
    everywhere has no grain-only region to measure and will read high; that is
    a real limitation, not a bug, so this test gives it a realistic mixture.
    """
    rng = np.random.default_rng(11)
    height = width = 320
    sigma = 0.008
    image = np.full((height, width), 0.5, dtype=np.float32)
    image += rng.standard_normal(image.shape).astype(np.float32) * sigma
    busy = image.copy()
    busy[:, 200:] = 0.5
    busy[:, 200::4] = 0.0  # a hard-edged region occupying a third of the frame

    quiet_estimate = measure_grain(image)
    busy_estimate = measure_grain(busy)
    assert 0.4 * sigma < quiet_estimate < 1.3 * sigma
    assert busy_estimate < 2.0 * quiet_estimate, (
        "the flat-tail selection should have ignored the hard-edged third"
    )


def test_measure_grain_on_tiny_image_returns_zero() -> None:
    assert measure_grain(np.zeros((8, 8), dtype=np.float32)) == 0.0


def test_measure_grain_rejects_colour() -> None:
    with pytest.raises(ValueError, match="2-D"):
        measure_grain(np.zeros((16, 16, 3), dtype=np.float32))


def test_add_grain_is_deterministic(smooth_rgb: Array) -> None:
    first = add_grain(smooth_rgb, 0.01, 2.0, seed=7)
    second = add_grain(smooth_rgb, 0.01, 2.0, seed=7)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(add_grain(smooth_rgb, 0.01, 2.0, seed=8), first)


def test_add_grain_is_luminance_modulated() -> None:
    """Grain must be strongest at mid-grey and suppressed at the extremes."""
    height = width = 128
    for level, expect_more in ((0.5, True), (0.02, False), (0.98, False)):
        field = np.full((height, width, 3), level, dtype=np.float32)
        grained = add_grain(field, 0.02, 2.0, strength=1.0, seed=3)
        deviation = float(np.abs(grained - field).mean())
        if expect_more:
            mid = deviation
        else:
            assert deviation < mid * 0.5, f"grain at {level} should be well below mid-grey"


def test_add_grain_disabled_is_a_no_op(smooth_rgb: Array) -> None:
    np.testing.assert_array_equal(add_grain(smooth_rgb, 0.01, 2.0, strength=0.0), smooth_rgb)
    np.testing.assert_array_equal(add_grain(smooth_rgb, 0.0, 2.0), smooth_rgb)


def test_neutral_gray_removes_cast() -> None:
    coloured = np.zeros((8, 8, 3), dtype=np.float32)
    coloured[..., 0], coloured[..., 1], coloured[..., 2] = 0.6, 0.5, 0.4
    grey = to_neutral_gray(coloured)
    assert float(grey.std(axis=2).max()) == pytest.approx(0.0, abs=1e-7)


def test_sharpen_gate_leaves_flat_tone_alone() -> None:
    """The threshold gate exists so restored grain is not amplified."""
    field = np.full((64, 64, 3), 0.5, dtype=np.float32)
    rng = np.random.default_rng(1)
    field = field + rng.standard_normal(field.shape).astype(np.float32) * 0.003
    sharpened = print_sharpen(field, 1.1, 0.42, threshold=0.012)
    assert float(np.abs(sharpened - field).max()) < 0.004


def test_sharpen_boosts_a_real_edge() -> None:
    """A step with headroom either side, so the USM overshoot is not clipped away."""
    field = np.full((64, 64, 3), 0.3, dtype=np.float32)
    field[:, 32:] = 0.7
    sharpened = print_sharpen(field, 1.1, 0.6, threshold=0.012)
    # Overshoot on both sides of the step is the signature of a working USM.
    assert sharpened[32, 30, 0] < field[32, 30, 0] - 1e-4
    assert sharpened[32, 34, 0] > field[32, 34, 0] + 1e-4
    assert float(np.abs(sharpened - field).max()) > 0.01


def test_local_contrast_preserves_mean(smooth_rgb: Array) -> None:
    lifted = local_contrast(smooth_rgb, 18.0, 0.16)
    assert float(abs(lifted.mean() - smooth_rgb.mean())) < 0.01
    assert lifted.min() >= 0.0
    assert lifted.max() <= 1.0
