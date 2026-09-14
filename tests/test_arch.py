"""Architecture construction and inference from a checkpoint's own tensors."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from printscale.arch import RRDBNet, SRVGGNetCompact
from printscale.models import load_model


@pytest.mark.parametrize(("scale", "expected_in_ch"), [(4, 3), (2, 12), (1, 48)])
def test_rrdbnet_input_channels_encode_scale(scale: int, expected_in_ch: int) -> None:
    """This relationship is what load_model uses to recover the scale."""
    model = RRDBNet(scale=scale, num_block=2)
    assert model.conv_first.in_channels == expected_in_ch


@pytest.mark.parametrize("scale", [1, 2, 4])
def test_rrdbnet_output_shape(scale: int) -> None:
    model = RRDBNet(scale=scale, num_block=2).eval()
    with torch.inference_mode():
        out = model(torch.rand(1, 3, 32, 32))
    assert out.shape == (1, 3, 32 * scale, 32 * scale)


def test_srvgg_output_shape() -> None:
    model = SRVGGNetCompact(num_conv=4, upscale=4).eval()
    with torch.inference_mode():
        out = model(torch.rand(1, 3, 16, 16))
    assert out.shape == (1, 3, 64, 64)


def test_residual_scaling_is_applied() -> None:
    """Guards the 0.2 constants, which fail silently rather than raising.

    Zero every convolution's weights and biases. Each inner RDB then returns
    ``0 * 0.2 + x``, i.e. the identity, so the RRDB's own residual gives
    ``x * 0.2 + x`` — exactly 1.2x. Any drift in either 0.2 constant moves this
    number, and nothing else in the codebase would raise.
    """
    from printscale.arch import RRDB

    block = RRDB(num_feat=8, num_grow_ch=4).eval()
    for param in block.parameters():
        torch.nn.init.zeros_(param)
    probe = torch.rand(1, 8, 12, 12)
    with torch.inference_mode():
        out = block(probe)
    torch.testing.assert_close(out, probe * 1.2)


@pytest.mark.parametrize(("scale", "blocks"), [(2, 3), (4, 2)])
def test_load_model_round_trip_rrdbnet(tmp_path: Path, scale: int, blocks: int) -> None:
    """Save a known model, reload it by path, and check what was inferred."""
    original = RRDBNet(scale=scale, num_block=blocks)
    checkpoint = tmp_path / "fake.pth"
    torch.save({"params_ema": original.state_dict()}, checkpoint)

    loaded, inferred_scale = load_model(checkpoint)
    assert inferred_scale == scale
    assert isinstance(loaded, RRDBNet)
    assert len(list(loaded.body)) == blocks
    assert not loaded.training
    assert all(not p.requires_grad for p in loaded.parameters())


def test_load_model_round_trip_srvgg(tmp_path: Path) -> None:
    original = SRVGGNetCompact(num_conv=6, upscale=4)
    checkpoint = tmp_path / "compact.pth"
    torch.save({"params": original.state_dict()}, checkpoint)

    loaded, scale = load_model(checkpoint)
    assert scale == 4
    assert isinstance(loaded, SRVGGNetCompact)


def test_load_model_rejects_unknown_layout(tmp_path: Path) -> None:
    checkpoint = tmp_path / "nonsense.pth"
    torch.save({"totally.unrelated.weight": torch.zeros(2, 2)}, checkpoint)
    with pytest.raises(ValueError, match="unrecognised checkpoint"):
        load_model(checkpoint)


def test_load_model_is_strict(tmp_path: Path) -> None:
    """A checkpoint missing a tensor must fail, not load a half-initialised net."""
    state = RRDBNet(scale=2, num_block=2).state_dict()
    del state["conv_last.weight"]
    checkpoint = tmp_path / "truncated.pth"
    torch.save({"params": state}, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_model(checkpoint)
