"""The blend arithmetic.

This is the highest-value test in the suite. An overlap-blended tiler that
averages instead of dividing by the accumulated weight produces an image that
is subtly too bright in the overlap zones — no exception, no obvious artefact,
just a worse print. Running an identity callable through the tiler and
demanding exact reconstruction pins that behaviour permanently.
"""

from __future__ import annotations

import numpy as np
import pytest

from printscale import Array
from printscale.tiling import blend_window, tiled_apply


def test_identity_reconstructs_exactly(smooth_rgb: Array) -> None:
    """scale=1 with an identity callable must return the input unchanged."""
    out = tiled_apply(smooth_rgb, lambda patch: patch, 1, tile=64, overlap=8, progress_every=0)
    assert out.shape == smooth_rgb.shape
    np.testing.assert_allclose(out, smooth_rgb, atol=1e-5)


@pytest.mark.parametrize(("tile", "overlap"), [(48, 4), (64, 16), (96, 24), (128, 32)])
def test_identity_holds_across_geometries(smooth_rgb: Array, tile: int, overlap: int) -> None:
    """Whatever the windows sum to, the divide must still reconstruct."""
    out = tiled_apply(smooth_rgb, lambda p: p, 1, tile=tile, overlap=overlap, progress_every=0)
    np.testing.assert_allclose(out, smooth_rgb, atol=1e-5)


def test_windows_are_not_a_partition_of_unity() -> None:
    """Documents the reason the divide exists, so nobody 'simplifies' it away.

    With step = tile - 2*overlap the ramps never line up, and the accumulated
    weight overshoots badly. A naive sum would be ~49% too bright in places.
    """
    tile, overlap = 320, 32
    step = tile - 2 * overlap
    length = 3 * step + tile
    accumulated = np.zeros(length, dtype=np.float32)
    for start in range(0, 4 * step, step):
        window = blend_window(1, tile, overlap)[0]
        accumulated[start : start + tile] += window
    interior = accumulated[overlap:-overlap]
    assert interior.max() > 1.4, "expected a large overshoot; geometry may have changed"
    assert interior.min() >= 1.0 - 1e-5


def test_nearest_upsample_is_reconstructed(smooth_rgb: Array) -> None:
    """A deterministic x2 callable must survive tiling without seams."""

    def double(patch: Array) -> Array:
        return np.repeat(np.repeat(patch, 2, axis=0), 2, axis=1)

    reference = double(smooth_rgb)
    out = tiled_apply(smooth_rgb, double, 2, tile=64, overlap=8, progress_every=0)
    assert out.shape == reference.shape
    # Blending two identical upsamplings must not shift anything.
    assert float(np.abs(out - reference).max()) < 2e-3


def test_rejects_bad_geometry(smooth_rgb: Array) -> None:
    with pytest.raises(ValueError, match="must exceed"):
        tiled_apply(smooth_rgb, lambda p: p, 1, tile=32, overlap=16, progress_every=0)
    with pytest.raises(ValueError, match="HxWxC"):
        tiled_apply(smooth_rgb[..., 0], lambda p: p, 1, progress_every=0)


def test_rejects_callable_with_wrong_scale(smooth_rgb: Array) -> None:
    """A callable that lies about its scale must fail loudly, not blend garbage."""
    with pytest.raises(ValueError, match="expected"):
        tiled_apply(smooth_rgb, lambda p: p, 2, tile=64, overlap=8, progress_every=0)


def test_blend_window_shoulders() -> None:
    window = blend_window(40, 40, 8)
    assert window.shape == (40, 40)
    assert window[20, 20] == pytest.approx(1.0)
    assert window[0, 0] < 0.05
    # C1 continuity: the second difference along a shoulder stays small.
    row = window[20, :12]
    assert np.abs(np.diff(row, n=2)).max() < 0.1
