"""End-to-end finishing, driven by a stand-in model so no weights are needed."""

from __future__ import annotations

import numpy as np

from printscale.pipeline import FinishParams, finish
from printscale.tiling import tiled_apply


def _fake_sr(patch: np.ndarray) -> np.ndarray:
    """A deterministic, slightly smoothing x2 — stands in for the real model."""
    doubled = np.repeat(np.repeat(patch, 2, axis=0), 2, axis=1)
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    for axis in (0, 1):
        doubled = np.apply_along_axis(
            lambda line: np.convolve(line, kernel, mode="same"), axis, doubled
        )
    return doubled.astype(np.float32)


def test_finish_produces_a_printable_frame(
    smooth_rgb: np.ndarray, grainy_gray: tuple[np.ndarray, float]
) -> None:
    source_gray, _sigma = grainy_gray
    small = smooth_rgb[:96, :96]
    upscaled = tiled_apply(small, _fake_sr, 2, tile=48, overlap=8, progress_every=0)

    out = finish(upscaled, 2.0, FinishParams(despeckle=False), source_gray)

    assert out.shape == (192, 192)
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    # Grain was restored, so the result is not the flat field the model produced.
    assert float(out.std()) > float(upscaled[..., 0].std())


def test_finish_with_grain_disabled_is_smoother(
    smooth_rgb: np.ndarray, grainy_gray: tuple[np.ndarray, float]
) -> None:
    source_gray, _ = grainy_gray
    small = smooth_rgb[:96, :96]
    upscaled = tiled_apply(small, _fake_sr, 2, tile=48, overlap=8, progress_every=0)

    grained = finish(upscaled, 2.0, FinishParams(despeckle=False, grain=0.9), source_gray)
    plain = finish(upscaled, 2.0, FinishParams(despeckle=False, grain=0.0), source_gray)
    assert float(grained.std()) > float(plain.std())


def test_finish_is_reproducible(
    smooth_rgb: np.ndarray, grainy_gray: tuple[np.ndarray, float]
) -> None:
    source_gray, _ = grainy_gray
    small = smooth_rgb[:64, :64]
    upscaled = tiled_apply(small, _fake_sr, 2, tile=32, overlap=8, progress_every=0)
    params = FinishParams(despeckle=False)
    np.testing.assert_array_equal(
        finish(upscaled, 2.0, params, source_gray),
        finish(upscaled, 2.0, params, source_gray),
    )
