"""Weight registry, verified download, and architecture inference.

Security posture: a ``.pth`` is a pickle. Two controls apply here and both
matter — ``torch.load(..., weights_only=True)`` refuses arbitrary object
construction during unpickling, and every downloaded file is verified against a
SHA-256 digest pinned in :data:`WEIGHTS` before it is opened at all. Neither is
sufficient alone: the digest stops a substituted file, ``weights_only`` stops a
malicious tensor container that still matches nothing we pinned (a local file
passed by path, say).
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from torch import nn

from printscale.arch import RRDBNet, SRVGGNetCompact

LOG = logging.getLogger(__name__)

__all__ = ["WEIGHTS", "ModelSpec", "default_cache_dir", "ensure_weights", "load_model"]

_RELEASES: Final = "https://github.com/xinntao/Real-ESRGAN/releases/download"
_CHUNK: Final = 1 << 20


@dataclass(frozen=True)
class ModelSpec:
    """One published checkpoint: where it lives and what it must hash to."""

    name: str
    url: str
    sha256: str
    size: int
    scale: int
    note: str


WEIGHTS: Final[dict[str, ModelSpec]] = {
    "x2plus": ModelSpec(
        name="RealESRGAN_x2plus.pth",
        url=f"{_RELEASES}/v0.2.1/RealESRGAN_x2plus.pth",
        sha256="49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
        size=67061725,
        scale=2,
        note="Default. Best photographic detail at 2x, ~1/4 the cost of x4plus.",
    ),
    "x4plus": ModelSpec(
        name="RealESRGAN_x4plus.pth",
        url=f"{_RELEASES}/v0.1.0/RealESRGAN_x4plus.pth",
        sha256="4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        size=67040989,
        scale=4,
        note="Same body as x2plus. Only worth it when you genuinely need 4x.",
    ),
    "general-x4v3": ModelSpec(
        name="realesr-general-x4v3.pth",
        url=f"{_RELEASES}/v0.2.5.0/realesr-general-x4v3.pth",
        sha256="8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
        size=4885111,
        scale=4,
        note="Fast and small. Denoises film grain into plastic — check before using.",
    ),
    "general-wdn-x4v3": ModelSpec(
        name="realesr-general-wdn-x4v3.pth",
        url=f"{_RELEASES}/v0.2.5.0/realesr-general-wdn-x4v3.pth",
        sha256="1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca",
        size=4885111,
        scale=4,
        note="Weaker-denoise counterpart of general-x4v3; blend the two to taste.",
    ),
}


def default_cache_dir() -> Path:
    """Where weights live unless told otherwise. Honours PRINTSCALE_CACHE."""
    env = os.environ.get("PRINTSCALE_CACHE")
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".cache"
    return root / "printscale" / "weights"


def _digest(path: Path) -> str:
    """Stream a SHA-256 so a 67 MB checkpoint does not land in memory twice."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def ensure_weights(key: str, cache_dir: Path | None = None, *, force: bool = False) -> Path:
    """Return a local path to a verified checkpoint, downloading if needed.

    Raises:
        KeyError: unknown model key.
        OSError: the download failed or could not be written.
        ValueError: the downloaded file did not match its pinned digest.
    """
    try:
        spec = WEIGHTS[key]
    except KeyError:
        known = ", ".join(sorted(WEIGHTS))
        raise KeyError(f"unknown model {key!r}; known models: {known}") from None

    cache = cache_dir or default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / spec.name

    if dest.is_file() and not force:
        actual = _digest(dest)
        if actual == spec.sha256:
            LOG.debug("cached %s verified", spec.name)
            return dest
        LOG.warning(
            "cached %s has digest %s, expected %s — refetching",
            spec.name,
            actual[:12],
            spec.sha256[:12],
        )

    LOG.info("downloading %s (%.1f MB)", spec.name, spec.size / 1e6)
    # Download to a temp file in the same directory so the rename is atomic and
    # a partial transfer can never be mistaken for a good cache entry.
    fd, tmp_name = tempfile.mkstemp(dir=cache, suffix=".part")
    tmp = Path(tmp_name)
    try:
        os.close(fd)
        if not spec.url.startswith("https://"):  # defence in depth; the table is ours
            raise ValueError(f"refusing non-HTTPS weight URL: {spec.url}")
        with urllib.request.urlopen(spec.url, timeout=120) as resp, tmp.open("wb") as out:  # noqa: S310
            shutil.copyfileobj(resp, out, _CHUNK)
        actual = _digest(tmp)
        if actual != spec.sha256:
            raise ValueError(
                f"{spec.name} failed verification: got {actual}, expected {spec.sha256}. "
                "Refusing to load it."
            )
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    LOG.info("verified %s", spec.name)
    return dest


def _infer_rrdbnet(state: dict[str, torch.Tensor]) -> tuple[nn.Module, int, str]:
    """Recover scale and depth of an RRDBNet from its own tensors.

    ``conv_first`` input channels give the scale, because the x2 and x1 variants
    pixel-unshuffle their input by 2 and 4 respectively.
    """
    in_ch = int(state["conv_first.weight"].shape[1])
    scale = {3: 4, 12: 2, 48: 1}.get(in_ch)
    if scale is None:
        raise ValueError(f"cannot infer scale from conv_first in_channels={in_ch}")
    blocks = 1 + max(int(k.split(".")[1]) for k in state if k.startswith("body."))
    feat = int(state["conv_first.weight"].shape[0])
    return (
        RRDBNet(scale=scale, num_block=blocks, num_feat=feat),
        scale,
        (f"RRDBNet(blocks={blocks}, feat={feat}, scale={scale})"),
    )


def _infer_srvgg(state: dict[str, torch.Tensor]) -> tuple[nn.Module, int, str]:
    """Recover convolution count and scale of an SRVGGNetCompact."""
    last = max(int(k.split(".")[1]) for k in state if k.startswith("body."))
    num_conv = (last - 2) // 2
    out_ch = int(state[f"body.{last}.weight"].shape[0])
    scale = math.isqrt(out_ch // 3)
    if scale * scale * 3 != out_ch:
        raise ValueError(f"cannot infer scale from tail out_channels={out_ch}")
    feat = int(state["body.0.weight"].shape[0])
    return (
        SRVGGNetCompact(num_conv=num_conv, upscale=scale, num_feat=feat),
        scale,
        (f"SRVGGNetCompact(convs={num_conv}, feat={feat}, scale={scale})"),
    )


def load_model(source: str | Path, cache_dir: Path | None = None) -> tuple[nn.Module, int]:
    """Load a checkpoint by registry key or path. Returns (eval model, scale).

    The architecture is read out of the checkpoint rather than passed in, and
    the state dict is loaded with ``strict=True`` so any structural mismatch
    fails immediately instead of silently producing a worse image.
    """
    path = Path(source)
    if not path.exists():
        path = ensure_weights(str(source), cache_dir)

    blob = torch.load(path, map_location="cpu", weights_only=True)
    state: dict[str, torch.Tensor]
    if isinstance(blob, dict) and ("params_ema" in blob or "params" in blob):
        state = blob.get("params_ema") or blob["params"]
    else:
        state = blob

    keys = set(state)
    if "conv_first.weight" in keys:
        model, scale, desc = _infer_rrdbnet(state)
    elif "body.0.weight" in keys:
        model, scale, desc = _infer_srvgg(state)
    else:
        raise ValueError(f"unrecognised checkpoint layout in {path}")

    model.load_state_dict(state, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    LOG.info("loaded %s as %s", path.name, desc)
    return model, scale
