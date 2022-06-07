import logging
from typing import List
import torch
import torch.nn as nn
from unet.nn.blocks import (
    DoubleConvolution,
    DownScaleBlock,
    UpScaleBlock,
    UpSamplingOptions,
    PointWiseConvolution,
)


class DyUnetModel2D(nn.Module):
    """Dynamic u-net architecture using 2D convolutions"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filters: List[int],
        sampling_mode: UpSamplingOptions,
    ) -> None:
        super().__init__()

        (
            self.header_conv,
            self.down_block,
            self.up_block,
            self.bottom_conv,
        ) = self._build_model(in_channels, out_channels, filters, sampling_mode)

    def _build_model(
        self,
        in_channels: int,
        out_channels: int,
        filters: List[int],
        sample_mode: UpSamplingOptions,
    ):
        """build u-net main blocks (header conv, downsampler, upsampler, last conv)"""

        header_conv = DoubleConvolution(
            in_channels=in_channels, out_channels=filters[0]
        )

        # build down-sampling block
        down_sampler_out_channels = filters[1:] + [filters[-1] * 2]  # 512 -> 1024
        down_sampler_block = [
            DownScaleBlock(in_channels=i, out_channels=o)
            for i, o in zip(filters, down_sampler_out_channels)
        ]

        # build up-sampling block
        up_sampler_in_channels = [filters[-1] * 2] + list(reversed(filters))[:-1]
        up_sampler_block = [
            UpScaleBlock(in_channels=i, out_channels=o, mode=sample_mode)
            for i, o in zip(up_sampler_in_channels, reversed(filters))
        ]

        last_conv = PointWiseConvolution(
            in_channels=filters[0], out_channels=out_channels
        )

        return header_conv, down_sampler_block, up_sampler_block, last_conv

    def forward(self, x):
        x = self.header_conv(x)

        down_samples_ctx = []
        for idx in range(len(self.down_block)):

            if len(down_samples_ctx) == 0:
                down_samples_ctx.append(self.down_block[idx](x))

            else:
                last_ctx = down_samples_ctx[-1]
                down_samples_ctx.append(self.down_block[idx](last_ctx))


        ground_ctx = self.up_block[0](down_samples_ctx[-1], down_samples_ctx[-2])

        up_block_loop = self.up_block[1:]
        for idx, ctx in enumerate(reversed(down_samples_ctx[:-2])):
            ground_ctx = up_block_loop[idx](ground_ctx, ctx)

        # last up-scale block
        ground_ctx = up_block_loop[-1](ground_ctx, x)
        logits = self.bottom_conv(ground_ctx)
        return logits