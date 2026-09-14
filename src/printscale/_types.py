"""Shared type aliases.

`np.ndarray` is generic. Under mypy's strict mode an unparameterised alias is a
``type-arg`` error — but only against numpy stubs old enough to require the
arguments, which is what resolves on Python 3.10. Parameterising once here
keeps every module consistent across the support matrix.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

__all__ = ["Array", "as_float32"]

#: A numpy array of unspecified dtype. Every stage in this package works in
#: float32 [0, 1] unless its docstring says otherwise; the alias does not
#: enforce that, because the 8-bit and 16-bit paths share these signatures.
Array: TypeAlias = npt.NDArray[Any]


def as_float32(array: Any) -> Array:
    """Pin a result to float32.

    OpenCV returns a loosely typed array whose dtype follows its input, and on
    older numpy stubs that surfaces as ``Any`` — which strict mode rejects at a
    return boundary. Making the dtype explicit satisfies the checker and, more
    usefully, keeps a stray float64 from silently doubling memory on a
    16-megapixel frame.
    """
    return np.asarray(array, dtype=np.float32)
