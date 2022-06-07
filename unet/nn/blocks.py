from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum


class DoubleConvolution(nn.Module):
    """[convolution + batch normalization + relu] * two"""

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: Union[None, int] = None
    ) -> None:
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels
        self.double_model = nn.Sequential(
            # first block
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # second block
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_model(x)


class DownScaleBlock(nn.Module):
    """down-scale using max-pool followed by double-convolution"""

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.max_pool_block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvolution(in_channels=in_channels, out_channels=out_channels),
        )

    def forward(self, x):
        return self.max_pool_block(x)


class UpSamplingOptions(Enum):
    bilinear = "bilinear"
    transpose = "transpose"
    nearest = "nearest"


class UpScaleBlock(nn.Module):
    """up-sampling followed by double-convolution"""

    def __init__(
        self, in_channels, out_channels, mode: UpSamplingOptions = "bilinear"
    ) -> None:
        super().__init__()

        # up-sampling using transposed-convolutions
        if mode == UpSamplingOptions.transpose:

            self.up = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=2,
                stride=2,
            )
            self.conv = DoubleConvolution(
                in_channels=in_channels, out_channels=out_channels
            )

        else:
            self.up = nn.Upsample(scale_factor=2, mode=mode.value, align_corners=True)
            self.conv = DoubleConvolution(
                in_channels=in_channels, out_channels=out_channels
            )

    def forward(self, x_first, x_second):
        x_first = self.up(x_first)
        # input shape is (batch, channels, width, height)
        diffY = x_first.size()[2] - x_second.size()[2]  # diff of width
        diffX = x_first.size()[3] - x_second.size()[3]  # diff of height

        # padding inputs in order to use skip-connections
        x_second = F.pad(
            x_second, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2))
        )

        x_first = torch.cat([x_first, x_second], dim=1)
        return self.conv(x_first)


class PointWiseConvolution(nn.Module):
    """single convolution block with kernel size of 1"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        return self.conv(x)
