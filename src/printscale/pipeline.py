"""Orchestration: the crop bake-off, the model pass, and the finishing chain."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from printscale import finishing
from printscale._types import Array
from printscale.despot import despeckle
from printscale.models import WEIGHTS, load_model
from printscale.tiling import tiled_apply

LOG = logging.getLogger(__name__)

__all__ = ["BakeResult", "FinishParams", "bake_off", "finish", "upscale"]


@dataclass
class FinishParams:
    """Every knob in the finishing chain, with defaults tuned on a film scan."""

    grain: float = 0.55
    lc_radius: float = 18.0
    lc_amount: float = 0.16
    sharp_radius: float = 1.1
    sharp_amount: float = 0.42
    sharp_threshold: float = 0.012
    despeckle: bool = True
    detail_limit: float = 5.0
    seed: int = 1


@dataclass
class BakeResult:
    """One candidate's showing on the test crop."""

    name: str
    seconds: float
    megapixels: float
    image: Array = field(repr=False)

    @property
    def seconds_per_megapixel(self) -> float:
        return self.seconds / max(self.megapixels, 1e-9)

    def projected_minutes(self, full_megapixels: float) -> float:
        return self.seconds_per_megapixel * full_megapixels / 60.0


def _torch_upscaler(model: torch.nn.Module):  # type: ignore[no-untyped-def]
    """Wrap a model as the plain array-to-array callable :func:`tiled_apply` wants."""

    def run(patch: Array) -> Array:
        with torch.inference_mode():
            tensor = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0)
            out = model(tensor).clamp_(0.0, 1.0)
            array: Array = out.squeeze(0).numpy().transpose(1, 2, 0)
            return array

    return run


def upscale(
    rgb: Array,
    model_key: str = "x2plus",
    *,
    tile: int = 320,
    overlap: int = 32,
    threads: int = 0,
    cache_dir: Path | None = None,
) -> tuple[Array, int]:
    """Run the model over a whole image. Returns (result, scale).

    Args:
        threads: torch intra-op threads. 0 leaves torch's own default alone,
            which is what you want on a GPU box or a many-core machine.
    """
    if threads > 0:
        torch.set_num_threads(threads)
    model, scale = load_model(model_key, cache_dir)
    started = time.monotonic()
    result = tiled_apply(rgb, _torch_upscaler(model), scale, tile=tile, overlap=overlap)
    LOG.info("super-resolution finished in %.1f min", (time.monotonic() - started) / 60)
    return result, scale


def bake_off(
    rgb: Array,
    *,
    crop: tuple[int, int, int] | None = None,
    models: tuple[str, ...] = ("x2plus", "x4plus", "general-x4v3"),
    threads: int = 0,
    cache_dir: Path | None = None,
) -> list[BakeResult]:
    """Run every candidate on one small crop and report cost per megapixel.

    Never start a full-frame run on an unproven model. Pick a crop containing
    the hardest content you have — a face, fine hair, small text.

    Args:
        crop: (x, y, size). Defaults to a centred square.
    """
    if crop is None:
        side = min(384, rgb.shape[0], rgb.shape[1])
        crop = ((rgb.shape[1] - side) // 2, (rgb.shape[0] - side) // 2, side)
    x, y, side = crop
    if x < 0 or y < 0 or y + side > rgb.shape[0] or x + side > rgb.shape[1]:
        raise ValueError(f"crop {crop} does not fit inside {rgb.shape[1]}x{rgb.shape[0]}")
    patch = rgb[y : y + side, x : x + side, :]
    megapixels = (side * side) / 1e6

    if threads > 0:
        torch.set_num_threads(threads)

    results: list[BakeResult] = []

    # Baseline first. On simple images it sometimes wins, and it costs nothing
    # to find that out.
    started = time.monotonic()
    as_bgr = (patch[..., ::-1] * 255).astype(np.uint8)
    lanczos = cv2.resize(as_bgr, (side * 2, side * 2), interpolation=cv2.INTER_LANCZOS4)
    blurred = cv2.GaussianBlur(lanczos, (0, 0), 1.6)
    sharpened = cv2.addWeighted(lanczos, 1.55, blurred, -0.55, 0)
    results.append(
        BakeResult(
            "lanczos+usm",
            time.monotonic() - started,
            megapixels,
            sharpened[..., ::-1].astype(np.float32) / 255.0,
        )
    )

    for key in models:
        if key not in WEIGHTS:
            LOG.error("skipping unknown model %r", key)
            continue
        try:
            model, scale = load_model(key, cache_dir)
        except (OSError, ValueError, KeyError):
            LOG.exception("could not load %s; skipping", key)
            continue
        started = time.monotonic()
        out = tiled_apply(
            patch, _torch_upscaler(model), scale, tile=192, overlap=24, progress_every=0
        )
        elapsed = time.monotonic() - started
        if scale > 2:  # normalise every candidate to x2 so they are comparable
            target = (side * 2, side * 2)
            out = cv2.resize(out, target, interpolation=cv2.INTER_AREA)
        results.append(BakeResult(key, elapsed, megapixels, out))
        LOG.info(
            "%-14s %6.1fs for %.2f MP  ->  %6.1f s/MP",
            key,
            elapsed,
            megapixels,
            elapsed / megapixels,
        )

    return results


def contact_sheet(results: list[BakeResult], region: tuple[int, int, int] | None = None) -> Array:
    """Tile the bake-off outputs side by side at 1:1 so they can be compared.

    Returns BGR uint8, ready for ``cv2.imwrite``. Looking at this is the point
    of the bake-off; the timings alone will not tell you about waxy skin.
    """
    if not results:
        raise ValueError("no bake-off results to lay out")
    side = results[0].image.shape[0]
    if region is None:
        size = min(340, side)
        region = ((side - size) // 2, (side - size) // 2, size)
    rx, ry, size = region

    tiles: list[Array] = []
    for result in results:
        crop = (result.image[ry : ry + size, rx : rx + size, ::-1] * 255).astype(np.uint8).copy()
        cv2.rectangle(crop, (0, 0), (size - 1, 24), (0, 0, 0), -1)
        label = f"{result.name}  {result.seconds_per_megapixel:.0f}s/MP"
        cv2.putText(
            crop, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )
        tiles.append(crop)

    columns = min(3, len(tiles))
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + columns]) for i in range(0, len(tiles), columns)]
    return np.vstack(rows)


def finish(rgb: Array, scale: float, params: FinishParams, source_gray: Array) -> Array:
    """Run the finishing chain. Returns HxW float32 greyscale in [0, 1].

    Args:
        source_gray: the ORIGINAL single-channel image, not the upscaled one.
            Grain must be measured before the model smoothed it away.
    """
    sigma = finishing.measure_grain(source_gray)
    LOG.info("measured source grain sigma %.5f (%.2f/255)", sigma, sigma * 255)

    out = finishing.to_neutral_gray(rgb)
    out = finishing.add_grain(out, sigma, scale, strength=params.grain, seed=params.seed)
    out = finishing.local_contrast(out, params.lc_radius, params.lc_amount)
    out = finishing.print_sharpen(
        out, params.sharp_radius, params.sharp_amount, params.sharp_threshold
    )
    gray = out[..., 0]

    if params.despeckle:
        as_uint16 = (np.clip(gray, 0, 1) * 65535).round().astype(np.uint16)
        repaired, _mask = despeckle(as_uint16, detail_limit=params.detail_limit)
        gray = repaired.astype(np.float32) / 65535.0

    return gray
