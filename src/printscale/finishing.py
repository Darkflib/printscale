"""The finishing chain. Order is load-bearing.

1. neutral greyscale  — so colour management finds no cast to correct
2. grain restoration  — GAN output is locally smooth and reads as plastic
3. local contrast     — large-radius unsharp, adds presence
4. print sharpen      — edge-radius unsharp, threshold-gated
5. dust removal       — see :mod:`printscale.despot`, runs last on 16-bit

Stages 3 and 4 are the same operation; only the radius differs, which is what
decides the spatial frequency band being boosted. Stage 4 must be gated because
stage 2 deliberately put grain back and an ungated sharpen would amplify it.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from printscale._types import Array, as_float32

LOG = logging.getLogger(__name__)

__all__ = [
    "add_grain",
    "local_contrast",
    "measure_grain",
    "print_sharpen",
    "to_neutral_gray",
]


# Rec. 709 luma coefficients, applied to linear-ish sRGB values. Good enough for
# collapsing a monochrome scan's paper cast; not a colorimetric conversion.
_LUMA = (0.2126, 0.7152, 0.0722)


def to_neutral_gray(rgb: Array) -> Array:
    """Collapse a 3-channel image to neutral grey, returned still 3-channel."""
    luma = _LUMA[0] * rgb[..., 0] + _LUMA[1] * rgb[..., 1] + _LUMA[2] * rgb[..., 2]
    return as_float32(np.repeat(luma[..., None], 3, axis=2))


def measure_grain(gray: Array, *, samples: int = 320, patch: int = 24, seed: int = 0) -> float:
    """Estimate grain sigma from the flattest patches in the image.

    High-pass each sampled patch, take its standard deviation, then use the
    *lower tail* of that distribution. Sampling the flattest regions is what
    separates grain from detail — the median over all patches would mostly
    measure edges.

    Args:
        gray: HxW float32 in [0, 1].

    Returns:
        Estimated sigma in the same [0, 1] units. 0.0 if the image is too small.
    """
    if gray.ndim != 2:
        raise ValueError(f"expected a 2-D single-channel image, got {gray.shape}")
    height, width = gray.shape
    if height < patch or width < patch:
        LOG.warning("image %dx%d smaller than patch %d; returning 0", width, height, patch)
        return 0.0

    rng = np.random.default_rng(seed)
    sigmas: list[float] = []
    for _ in range(samples):
        y = int(rng.integers(0, height - patch + 1))
        x = int(rng.integers(0, width - patch + 1))
        block = gray[y : y + patch, x : x + patch]
        high_pass = block - as_float32(cv2.GaussianBlur(block, (0, 0), 2.0))
        sigmas.append(float(high_pass.std()))

    sigmas.sort()
    # Discard the very bottom (dead-flat scanner areas read as ~0) and anything
    # above the median (that is detail, not grain).
    tail = sigmas[len(sigmas) // 8 : len(sigmas) // 2]
    return float(np.median(tail)) if tail else 0.0


def add_grain(
    rgb: Array,
    sigma: float,
    scale: float,
    *,
    strength: float = 0.55,
    seed: int = 1,
) -> Array:
    """Re-introduce monochrome grain the model smoothed away.

    Modulated by luminance on a triangular weight peaking at mid-grey, because
    photographic granularity genuinely peaks around mid-density and falls away
    in deep shadow and blown highlight. Flat additive noise reads as digital
    hiss precisely because it ignores this.

    Args:
        sigma: grain amplitude from :func:`measure_grain`.
        scale: the enlargement factor, used to widen the grain clumps to match.
        strength: 0 disables; 0.5-0.7 is usually right; above 1.0 is visible.
    """
    if sigma <= 0 or strength <= 0:
        return rgb
    height, width, _ = rgb.shape
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((height, width), dtype=np.float32)
    noise = as_float32(cv2.GaussianBlur(noise, (0, 0), max(0.5, 0.45 * scale)))
    noise /= max(float(noise.std()), 1e-6)

    luma = rgb.mean(axis=2)
    weight = np.clip(1.0 - np.abs(luma - 0.5) * 2.0, 0.0, 1.0) ** 0.6
    return as_float32(np.clip(rgb + (noise * weight * (sigma * strength))[..., None], 0.0, 1.0))


def local_contrast(rgb: Array, radius: float = 18.0, amount: float = 0.16) -> Array:
    """Large-radius, low-amount unsharp. Adds presence without haloing edges."""
    if amount <= 0 or radius <= 0:
        return rgb
    blurred = as_float32(cv2.GaussianBlur(rgb, (0, 0), radius))
    return as_float32(np.clip(rgb + (rgb - blurred) * amount, 0.0, 1.0))


def print_sharpen(
    rgb: Array,
    radius: float = 1.1,
    amount: float = 0.42,
    threshold: float = 0.012,
) -> Array:
    """Threshold-gated unsharp at the edge radius.

    Only differences above `threshold` are boosted, so the grain added in
    :func:`add_grain` is left alone. The gate mask is itself feathered, because
    a hard on/off boundary along an edge leaves a ragged transition.
    """
    if amount <= 0 or radius <= 0:
        return rgb
    blurred = as_float32(cv2.GaussianBlur(rgb, (0, 0), radius))
    detail = rgb - blurred
    gate = (np.abs(detail) > threshold).astype(np.float32)
    gate = as_float32(cv2.GaussianBlur(gate, (0, 0), radius * 0.7))
    return as_float32(np.clip(rgb + detail * amount * gate, 0.0, 1.0))
