"""Dust and speck removal for scans of photographic prints.

A morphological top-hat (image minus its opening) isolates bright structures
smaller than the structuring element, which is the textbook detector for white
dust on a dark print. Run naively it is also a very good eyelash detector — an
eyelash is equally a bright structure smaller than a 9-pixel disc.

The fix is context, not a better detector: accept a speck only where the
surrounding neighbourhood is flat. Dust sits on smooth tone; eyelashes sit in
hair. On a real 5436x3000 press scan this took the candidate count from 16,768
to 163.

Always render the mask and look at it before applying. :func:`despeckle`
returns it for exactly that reason.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from printscale._types import Array, as_float32

LOG = logging.getLogger(__name__)

__all__ = ["despeckle", "find_specks", "local_detail", "ring_detail"]


def local_detail(gray: Array, window: int = 21) -> Array:
    """Local standard deviation — a cheap proxy for "is there real texture here".

    Computed from box-filtered first and second moments, so it costs two passes
    regardless of window size.

    This is a diagnostic, not the gate. A speck inflates the reading of its own
    neighbourhood, so screening candidates on it silently drops the
    highest-contrast dust. Use it to choose a `detail_limit` by eye; the gate
    itself is :func:`ring_detail`, which excludes the candidate from its own
    surroundings.
    """
    values = gray.astype(np.float32)
    mean = cv2.boxFilter(values, -1, (window, window), normalize=True)
    mean_sq = cv2.boxFilter(values * values, -1, (window, window), normalize=True)
    return as_float32(np.sqrt(np.maximum(mean_sq - mean * mean, 0.0)))


def ring_detail(gray: Array, blob: Array, box: tuple[int, int, int, int], pad: int) -> float:
    """Standard deviation of the annulus around a candidate, excluding the candidate.

    This is the honest form of "dust sits on smooth tone, eyelashes sit in
    hair". Two cheaper formulations both fail:

    * a plain local standard deviation includes the speck, so a high-contrast
      speck inflates its own neighbourhood and the gate rejects it;
    * median-filtering first removes the speck, but also erases the thin
      strokes that make a hair region busy, so hair starts reading as flat.

    Args:
        blob: boolean mask of the candidate, same shape as `gray`.
        box: (x, y, width, height) of the candidate's bounding box.
        pad: how far beyond the box to look.
    """
    x, y, width, height = box
    height_total, width_total = gray.shape
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(width_total, x + width + pad), min(height_total, y + height + pad)

    patch = gray[y0:y1, x0:x1].astype(np.float32)
    occupied = blob[y0:y1, x0:x1]
    surroundings = patch[~occupied]
    if surroundings.size < 16:
        return float("inf")  # too little context to judge; refuse the candidate
    return float(surroundings.std())


def find_specks(
    gray: Array,
    *,
    polarity: str = "white",
    disk: int = 9,
    contrast: int = 40,
    max_area: int = 45,
    detail_limit: float = 5.0,
    max_aspect: float = 3.0,
) -> Array:
    """Mask of speck candidates of one polarity. uint8, 0 or 255.

    Args:
        gray: HxW uint8.
        polarity: "white" for bright specks on dark tone, "black" for the inverse.
        disk: structuring element diameter; specks must be smaller than this.
        contrast: minimum top-hat response, in 8-bit levels.
        max_area: reject anything larger, in pixels.
        detail_limit: reject a candidate whose surrounding annulus is busier
            than this. The single most important parameter — without it the
            detector eats hair and eyelashes.
        max_aspect: reject elongated shapes, which are scratches or real detail.
    """
    if polarity not in {"white", "black"}:
        raise ValueError(f"polarity must be 'white' or 'black', got {polarity!r}")

    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (disk, disk))
    operation = cv2.MORPH_TOPHAT if polarity == "white" else cv2.MORPH_BLACKHAT
    response = cv2.morphologyEx(gray, operation, element)

    # No global detail pre-screen here. A local standard deviation includes the
    # speck, so screening on it silently drops the highest-contrast dust — the
    # very thing most worth removing. The per-blob ring test below is cheap
    # enough (a few thousand small windows) to be the only gate.
    candidates = (response > contrast).astype(np.uint8)
    # cv2-stubs types this overload as integer|floating and rejects uint8, which
    # is the documented input type for a binary mask.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)  # type: ignore[call-overload]

    keep = np.zeros_like(candidates)
    kept = 0
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        box_w = int(stats[index, cv2.CC_STAT_WIDTH])
        box_h = int(stats[index, cv2.CC_STAT_HEIGHT])
        if area > max_area or max(box_w, box_h) > disk:
            continue
        if max(box_w, box_h) / max(min(box_w, box_h), 1) > max_aspect:
            continue
        blob = labels == index
        box = (
            int(stats[index, cv2.CC_STAT_LEFT]),
            int(stats[index, cv2.CC_STAT_TOP]),
            box_w,
            box_h,
        )
        if ring_detail(gray, blob, box, pad=disk) > detail_limit:
            continue
        keep[blob] = 255
        kept += 1

    LOG.info("%s specks: %d candidates, %d kept", polarity, count - 1, kept)
    return keep


def despeckle(
    gray: Array,
    *,
    disk: int = 9,
    contrast: int = 40,
    max_area: int = 45,
    detail_limit: float = 5.0,
    detail_window: int = 21,
) -> tuple[Array, Array]:
    """Remove dust specks. Returns (repaired image, mask that was applied).

    Works on 8-bit or 16-bit single-channel input. The 16-bit path does not use
    a wide median: OpenCV's ``medianBlur`` only accepts ksize <= 5 above 8-bit
    depth, so the fill is an iterated 5x5 median combined with a greyscale
    opening — which is the right operator for bright specks on dark tone anyway.
    """
    if gray.ndim != 2:
        raise ValueError(f"expected a single-channel image, got shape {gray.shape}")

    eight_bit = gray if gray.dtype == np.uint8 else (gray / 257.0).astype(np.uint8)
    LOG.debug(
        "median local detail %.2f (diagnostic; the gate is per-blob)",
        float(np.median(local_detail(eight_bit, detail_window))),
    )

    white = find_specks(
        eight_bit,
        polarity="white",
        disk=disk,
        contrast=contrast,
        max_area=max_area,
        detail_limit=detail_limit,
    )
    black = find_specks(
        eight_bit,
        polarity="black",
        disk=disk,
        contrast=contrast,
        max_area=max_area,
        detail_limit=detail_limit,
    )
    mask = cv2.dilate(cv2.bitwise_or(white, black), np.ones((3, 3), np.uint8), iterations=1)

    if not mask.any():
        LOG.info("no specks matched; image unchanged")
        return gray.copy(), mask

    if gray.dtype == np.uint8:
        repaired = cv2.inpaint(gray, mask, 4, cv2.INPAINT_TELEA)
    else:
        median = gray
        for _ in range(3):
            median = cv2.medianBlur(median, 5)
        opened = cv2.morphologyEx(
            gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (disk, disk))
        )
        fill = np.minimum(median, opened)
        repaired = np.where(mask > 0, fill, gray).astype(gray.dtype)

    touched = float((mask > 0).mean())
    LOG.info("repaired %d speck pixels (%.4f%% of frame)", int((mask > 0).sum()), 100 * touched)
    if touched > 0.002:
        LOG.warning(
            "dust removal touched %.3f%% of pixels — over ~0.2%% it is "
            "probably eating real detail; inspect the mask",
            100 * touched,
        )
    return repaired, mask
