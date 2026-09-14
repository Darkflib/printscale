"""Shared fixtures. Synthetic images only — no test depends on a real photo."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260914)


@pytest.fixture
def smooth_rgb(rng: np.random.Generator) -> np.ndarray:
    """A gently varying HxWx3 float32 image in [0, 1], no hard edges."""
    height, width = 200, 260
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    base = 0.5 + 0.2 * np.sin(xx / 40.0) + 0.15 * np.cos(yy / 55.0)
    base = np.clip(base, 0.05, 0.95).astype(np.float32)
    return np.repeat(base[..., None], 3, axis=2)


@pytest.fixture
def grainy_gray(rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """A flat-ish field with grain of a known sigma. Returns (image, sigma)."""
    sigma = 0.012
    height, width = 320, 320
    _yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    base = 0.45 + 0.05 * np.sin(xx / 90.0)
    noisy = base + rng.standard_normal((height, width)) * sigma
    return np.clip(noisy, 0.0, 1.0).astype(np.float32), sigma
