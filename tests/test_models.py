"""Weight fetching. The security-relevant part of the package."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import pytest

from printscale.models import WEIGHTS, ModelSpec, default_cache_dir, ensure_weights


def _install_fake(monkeypatch: pytest.MonkeyPatch, payload: bytes, declared: str) -> None:
    """Register a fake model whose declared digest may or may not match."""
    spec = ModelSpec(
        name="fake.pth",
        url="https://example.invalid/fake.pth",
        sha256=declared,
        size=len(payload),
        scale=2,
        note="test fixture",
    )
    monkeypatch.setitem(WEIGHTS, "fake", spec)

    class _Response(io.BytesIO):
        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *exc: Any) -> None:
            self.close()

    monkeypatch.setattr(
        "printscale.models.urllib.request.urlopen",
        lambda url, timeout=0: _Response(payload),
    )


def test_download_is_verified_and_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"not really a checkpoint, but it hashes"
    _install_fake(monkeypatch, payload, hashlib.sha256(payload).hexdigest())

    path = ensure_weights("fake", tmp_path)
    assert path.read_bytes() == payload
    # A second call must hit the cache, not the network.
    monkeypatch.setattr(
        "printscale.models.urllib.request.urlopen",
        lambda *a, **k: pytest.fail("should have used the cache"),
    )
    assert ensure_weights("fake", tmp_path) == path


def test_digest_mismatch_is_refused_and_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A substituted file must never reach torch.load, and must not be cached."""
    _install_fake(monkeypatch, b"tampered", "0" * 64)
    with pytest.raises(ValueError, match="failed verification"):
        ensure_weights("fake", tmp_path)
    assert not (tmp_path / "fake.pth").exists()
    assert not list(tmp_path.glob("*.part")), "a partial download was left behind"


def test_corrupt_cache_entry_is_refetched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"the real thing"
    _install_fake(monkeypatch, payload, hashlib.sha256(payload).hexdigest())
    stale = tmp_path / "fake.pth"
    stale.write_bytes(b"something else entirely")

    path = ensure_weights("fake", tmp_path)
    assert path.read_bytes() == payload


def test_unknown_model_lists_the_known_ones() -> None:
    with pytest.raises(KeyError, match="x2plus"):
        ensure_weights("no-such-model", Path("does-not-matter"))


def test_registry_digests_are_well_formed() -> None:
    for key, spec in WEIGHTS.items():
        assert len(spec.sha256) == 64, f"{key} digest is not a SHA-256"
        assert int(spec.sha256, 16) >= 0, f"{key} digest is not hex"
        assert spec.url.startswith("https://"), f"{key} is not fetched over HTTPS"
        assert spec.scale in (1, 2, 4)


def test_cache_dir_honours_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PRINTSCALE_CACHE", str(tmp_path / "elsewhere"))
    assert default_cache_dir() == tmp_path / "elsewhere"
