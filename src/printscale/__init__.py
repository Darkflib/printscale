"""printscale — upscale scanned photographs for large-format print.

The model is one stage of six. Most of the perceived quality comes from the
finishing chain, and most of the wasted time comes from committing compute
before a cheap crop test says it will pay.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "add_grain",
    "despeckle",
    "load_model",
    "measure_grain",
    "ppi_table",
    "tiled_apply",
]

from printscale.despot import despeckle
from printscale.finishing import add_grain, measure_grain
from printscale.metrics import ppi_table
from printscale.models import load_model
from printscale.tiling import tiled_apply
