"""Network architectures for the official Real-ESRGAN checkpoints.

These follow BasicSR's definitions closely enough to load its published weights
with ``strict=True``. They are reimplemented here rather than imported because
``basicsr`` and ``realesrgan`` pin ``torchvision.transforms.functional_tensor``,
which was removed in torchvision 0.17 — installing them alongside a current
torch is a dependency conflict with no clean resolution.

Two numbers in here are load-bearing and produce no error if wrong:

* the ``0.2`` residual scaling in both :class:`ResidualDenseBlock` and
  :class:`RRDB` — get it wrong and output quality degrades silently;
* the ``0.2`` LeakyReLU negative slope.

:func:`printscale.models.load_model` guards structure via ``strict=True`` and
guards numerics via a golden-output smoke test; see ``tests/test_arch.py``.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

__all__ = ["RRDB", "RRDBNet", "ResidualDenseBlock", "SRVGGNetCompact"]

# Empirical residual scaling from the ESRGAN paper. Keeps activations bounded
# so a 23-block stack trains without exploding.
RESIDUAL_SCALE = 0.2
LRELU_SLOPE = 0.2


class ResidualDenseBlock(nn.Module):
    """Five convolutions with dense concatenation, scaled residual output.

    Each convolution sees the block input concatenated with every earlier
    activation, which gives short gradient paths and heavy feature reuse.
    """

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=LRELU_SLOPE, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return cast(Tensor, x5 * RESIDUAL_SCALE + x)


class RRDB(nn.Module):
    """Residual in residual dense block: three RDBs plus an identity path."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: Tensor) -> Tensor:
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return cast(Tensor, out * RESIDUAL_SCALE + x)


class RRDBNet(nn.Module):
    """The x2plus / x4plus generator. 16.7M parameters.

    The x2 and x4 variants share an identical body. For ``scale=2`` the input
    is passed through ``pixel_unshuffle`` first, folding each 2x2 spatial block
    into the channel dimension, so ``conv_first`` takes 12 channels and the 23
    RRDBs run at a quarter of the pixel count. The tail always upsamples by 4.

    That is why x2plus costs roughly a quarter of x4plus per input megapixel —
    a useful sanity check on any new setup.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch *= 4
        elif scale == 1:
            num_in_ch *= 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=LRELU_SLOPE, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        if self.scale == 2:
            feat = F.pixel_unshuffle(x, downscale_factor=2)
        elif self.scale == 1:
            feat = F.pixel_unshuffle(x, downscale_factor=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return cast(Tensor, self.conv_last(self.lrelu(self.conv_hr(feat))))


class SRVGGNetCompact(nn.Module):
    """The realesr-general-x4v3 generator. ~1.2M parameters.

    A plain convolution stack with a pixel-shuffle tail and a nearest-neighbour
    long skip. Fast, but trained against degradations weighted towards
    compressed web imagery, so it treats photographic grain as noise to remove.
    Rarely the right choice for a film scan.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 16,
        upscale: int = 4,
    ) -> None:
        super().__init__()
        self.upscale = upscale
        body: list[nn.Module] = [
            nn.Conv2d(num_in_ch, num_feat, 3, 1, 1),
            nn.PReLU(num_parameters=num_feat),
        ]
        for _ in range(num_conv):
            body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            body.append(nn.PReLU(num_parameters=num_feat))
        body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return cast(Tensor, out + F.interpolate(x, scale_factor=self.upscale, mode="nearest"))
