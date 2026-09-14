"""Reading and writing, including the 16-bit master and DPI stamping.

The pixel count is fixed once the model has run; print size is only a matter of
what you tell the printer. Stamping DPI metadata is how you say which size you
meant without resampling anything.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from printscale.metrics import MM_PER_INCH

LOG = logging.getLogger(__name__)

__all__ = ["dpi_for_width", "load_rgb", "save_jpeg", "save_master", "stamp_dpi"]

Array = np.ndarray


def load_rgb(path: Path | str) -> Array:
    """Read an image as HxWx3 float32 RGB in [0, 1]."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"could not read an image from {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def stamp_dpi(path: Path, dpi: int, **save_options: object) -> None:
    """Write DPI metadata in place. Non-fatal if the format will not carry it.

    Pillow has no way to edit a JPEG's metadata without re-encoding it, and it
    defaults to quality 75 when re-saving. Callers writing a JPEG must pass the
    same quality and subsampling they wrote it with, or the stamp quietly
    throws away most of the file.
    """
    try:
        with Image.open(path) as image:
            image.load()
            image.save(path, dpi=(dpi, dpi), **save_options)  # type: ignore[arg-type]
    except (OSError, ValueError):
        LOG.warning("could not stamp DPI on %s", path.name, exc_info=True)


def dpi_for_width(pixel_width: int, width_mm: float) -> int:
    """Effective DPI if `pixel_width` pixels are printed `width_mm` wide."""
    if width_mm <= 0:
        raise ValueError(f"width_mm must be positive, got {width_mm}")
    return round(pixel_width / (width_mm / MM_PER_INCH))


def save_master(gray: Array, path: Path, width_mm: float, *, compress: bool = True) -> Path:
    """Write the 16-bit greyscale TIFF the printer should receive.

    Compression is Deflate and lossless; an uncompressed 16-bit master of any
    size routinely exceeds upload and email limits for no benefit.
    """
    if gray.ndim != 2:
        raise ValueError(f"expected a single-channel image, got {gray.shape}")
    as_uint16 = (
        gray if gray.dtype == np.uint16 else (np.clip(gray, 0, 1) * 65535).round().astype(np.uint16)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    dpi = dpi_for_width(as_uint16.shape[1], width_mm)

    if compress:
        # No explicit mode: Pillow infers I;16 from uint16, and passing the
        # mode is deprecated from Pillow 13.
        Image.fromarray(as_uint16).save(path, compression="tiff_deflate", dpi=(dpi, dpi))
    else:
        cv2.imwrite(str(path), as_uint16)
        stamp_dpi(path, dpi)

    LOG.info(
        "wrote %s — %d x %d, %.0f mm at %d dpi, %.1f MB",
        path.name,
        as_uint16.shape[1],
        as_uint16.shape[0],
        width_mm,
        dpi,
        path.stat().st_size / 1e6,
    )
    return path


def save_jpeg(gray: Array, path: Path, width_mm: float, quality: int = 97) -> Path:
    """Write the 8-bit preview. For looking at and ordering from, not printing."""
    as_uint8 = (
        gray
        if gray.dtype == np.uint8
        else ((gray / 257.0) if gray.dtype == np.uint16 else np.clip(gray, 0, 1) * 255)
        .round()
        .astype(np.uint8)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # One write, through Pillow, so quality and DPI are set together. Writing
    # with cv2 and then stamping would re-encode at Pillow's default quality.
    Image.fromarray(as_uint8, mode="L").save(
        path,
        format="JPEG",
        quality=quality,
        subsampling=0,
        dpi=(dpi_for_width(as_uint8.shape[1], width_mm),) * 2,
    )
    LOG.info("wrote %s — %.1f MB", path.name, path.stat().st_size / 1e6)
    return path
