"""Overlap-blended tiled inference.

Peak memory for a convolutional network scales with activation volume — 64
feature maps at full spatial resolution through 23 blocks. Tiling trades a
little redundant compute for a bounded footprint.

The blend is the part people get wrong. Two rules:

1. Cross-fade with a raised cosine spanning the full 0 -> 1, not a linear
   ramp. A linear ramp has a discontinuous derivative at each end and leaves
   a faint visible crease. Tiles against the image border are not feathered
   on that side at all.
2. Divide by the accumulated weight rather than averaging. With
   ``step = tile - 2 * overlap`` the windows do **not** form a partition of
   unity — consecutive windows are offset by ``step`` while each ramp is only
   ``overlap`` wide, so the accumulated weight can reach about 1.5. The divide
   is what makes the blend exact whatever the windows happen to sum to.

``tests/test_tiling.py`` pins rule 2 by running an identity callable through
the tiler and asserting exact reconstruction.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import numpy as np

LOG = logging.getLogger(__name__)

__all__ = ["blend_window", "tiled_apply"]

Array = np.ndarray


def _cosine_ramp(length: int, feather: int, *, lead: bool = True, trail: bool = True) -> Array:
    """Flat-topped window with raised-cosine shoulders of `feather` samples.

    The shoulder runs the full 0 -> 1 of a half cosine. An earlier version
    sampled only ``linspace(0, pi, 2*feather)[:feather]``, which tops out around
    0.45 and then jumps to 1.0 — a discontinuity that undermines the whole point
    of using a cosine rather than a linear ramp.

    `lead` and `trail` select which ends are feathered. A tile sitting against
    the edge of the image has no neighbour to cross-fade with on that side, so
    its window must stay at 1.0 there; otherwise the accumulated weight falls to
    zero at the border and the divide produces a black edge line.
    """
    window = np.ones(length, dtype=np.float32)
    edge = min(feather, length // 2)
    if edge <= 0:
        return window
    # Sample shoulder midpoints so the ramp spans the full 0..1 range.
    steps = (np.arange(edge, dtype=np.float32) + 0.5) / edge
    shoulder = (1.0 - np.cos(np.pi * steps)) / 2.0
    if lead:
        window[:edge] = shoulder
    if trail:
        window[-edge:] = shoulder[::-1]
    return window


def blend_window(
    height: int,
    width: int,
    feather: int,
    edges: tuple[bool, bool, bool, bool] = (False, False, False, False),
) -> Array:
    """Separable 2-D raised-cosine window, shape (height, width).

    Args:
        edges: (top, left, bottom, right) — True means "this side is an image
            border", so that side is left unfeathered.
    """
    top, left, bottom, right = edges
    rows = _cosine_ramp(height, feather, lead=not top, trail=not bottom)
    cols = _cosine_ramp(width, feather, lead=not left, trail=not right)
    return np.outer(rows, cols).astype(np.float32)


def tiled_apply(
    image: Array,
    fn: Callable[[Array], Array],
    scale: int,
    *,
    tile: int = 320,
    overlap: int = 32,
    progress_every: int = 5,
) -> Array:
    """Apply `fn` tile-by-tile and blend the results.

    Args:
        image: HxWxC float32 in [0, 1].
        fn: maps an HxWxC tile to a (H*scale)x(W*scale)xC tile. Any callable —
            tests pass an identity so the blend arithmetic can be verified
            without a model.
        scale: the factor `fn` applies.
        tile: tile side in input pixels. Raise it until memory complains;
            larger tiles mean less redundant overlap compute.
        overlap: feather width in input pixels, on every edge.

    Raises:
        ValueError: on a non-3-channel image or a tile too small for the overlap.
    """
    if image.ndim != 3:
        raise ValueError(f"expected HxWxC, got shape {image.shape}")
    if tile <= 2 * overlap:
        raise ValueError(f"tile ({tile}) must exceed 2*overlap ({2 * overlap})")
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")

    height, width, channels = image.shape
    step = tile - 2 * overlap
    out = np.zeros((height * scale, width * scale, channels), dtype=np.float32)
    weight = np.zeros((height * scale, width * scale, 1), dtype=np.float32)

    ys = list(range(0, max(height - 2 * overlap, 1), step))
    xs = list(range(0, max(width - 2 * overlap, 1), step))
    total = len(ys) * len(xs)
    LOG.info(
        "tiling %dx%d into %d tiles of %d (overlap %d), scale x%d",
        width,
        height,
        total,
        tile,
        overlap,
        scale,
    )

    done = 0
    started = time.monotonic()
    for y_start in ys:
        for x_start in xs:
            # Clamp to the far edge so the last tile is full-size rather than a
            # thin sliver, which would otherwise get a lopsided window.
            y_end, x_end = min(y_start + tile, height), min(x_start + tile, width)
            y0, x0 = max(0, y_end - tile), max(0, x_end - tile)
            patch = image[y0:y_end, x0:x_end, :]

            result = fn(patch)
            expected = (patch.shape[0] * scale, patch.shape[1] * scale)
            if result.shape[:2] != expected:
                raise ValueError(
                    f"fn returned {result.shape[:2]} for tile at ({x0},{y0}); "
                    f"expected {expected} at scale x{scale}"
                )

            out_h, out_w = result.shape[:2]
            # Do not feather against the image border — there is no neighbour
            # there, and a zero accumulated weight would black out the edge.
            at_edges = (y0 == 0, x0 == 0, y_end == height, x_end == width)
            win = blend_window(out_h, out_w, overlap * scale, at_edges)[..., None]
            oy, ox = y0 * scale, x0 * scale
            out[oy : oy + out_h, ox : ox + out_w, :] += result * win
            weight[oy : oy + out_h, ox : ox + out_w, :] += win

            done += 1
            if progress_every and (done % progress_every == 0 or done == total):
                rate = (time.monotonic() - started) / done
                LOG.info(
                    "  tile %d/%d — %.1fs/tile, eta %.0fs", done, total, rate, rate * (total - done)
                )

    # The division, not an average: see the module docstring.
    np.divide(out, np.maximum(weight, 1e-6), out=out)
    return np.clip(out, 0.0, 1.0)
