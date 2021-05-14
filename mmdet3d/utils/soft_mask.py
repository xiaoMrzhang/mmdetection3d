"""Soft mask"""

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np

from mmdet3d.utils.residual_block import ResidualBlock


class SoftMask(nn.Module):
    """Soft mask encoder 1.

    Args:
        input_channels (int): Input channels.
        output_channels (int): Output channels.
    """

    # input size is 248*216
    def __init__(self, in_channels=128, out_channels=[128, 128, 256], size1=(248, 216), \
                 size2=(124, 108), size3=(62, 54)):
        super(SoftMask, self).__init__()

        self.first_residual_blocks = ResidualBlock(in_channels, out_channels[0])
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 124*108
        self.residual1_blocks = ResidualBlock(in_channels, out_channels[1])

        self.skip1_connection_residual_block = ResidualBlock(out_channels[1], out_channels[1])

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 62*54
        self.residual2_blocks = ResidualBlock(out_channels[1], out_channels[2])

        self.skip2_connection_residual_block = ResidualBlock(out_channels[2], out_channels[2])

        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 31*27
        self.residual3_blocks = nn.Sequential(
            ResidualBlock(out_channels[2], out_channels[2] * 2),
            ResidualBlock(out_channels[2] * 2, out_channels[2])
        )

        self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)
        self.residual4_blocks = ResidualBlock(out_channels[2], out_channels[1])
        # 62*54

        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)
        self.residual5_blocks = ResidualBlock(out_channels[1], out_channels[0])
        # 124*108

        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.residual6_blocks = nn.Sequential(
            nn.Conv2d(out_channels[2], out_channels[2], kernel_size=1, stride=1, bias = False),
            nn.BatchNorm2d(out_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels[2], out_channels[2], kernel_size=1, stride=1, bias = False),
            # nn.Sigmoid()
        )

        self.residual7_blocks = nn.Sequential(
            nn.Conv2d(out_channels[1], out_channels[1], kernel_size=1, stride=1, bias = False),
            nn.BatchNorm2d(out_channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels[1], out_channels[1], kernel_size=1, stride=1, bias = False),
            # nn.Sigmoid()
        )

        self.residual8_blocks = nn.Sequential(
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=1, stride=1, bias = False),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=1, stride=1, bias = False),
            # nn.Sigmoid()
        )

    def forward(self, x):
        """Forward function.
        
        Args:
            x (torch.Tensor): Input with shape (N, c, H, W)

        Returns:
            torch.Tensor: mask output.
        """
        # 248*216
        y = self.first_residual_blocks(x)

        out_mpool1 = self.mpool1(x)
        # 124*108
        out_residual1 = self.residual1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_residual1)
        out_mpool2 = self.mpool2(out_residual1)
        # 62*54
        out_residual2 = self.residual2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_residual2)
        out_mpool3 = self.mpool3(out_residual2)
        # 31*27
        out_residual3 = self.residual3_blocks(out_mpool3)
        out = self.interpolation3(out_residual3) + out_skip2_connection        
        # 62*54
        mask3 = self.residual6_blocks(out)

        out_residual4 = self.residual4_blocks(out)
        out = self.interpolation2(out_residual4) + out_skip1_connection        
        # 124*108
        mask2 = self.residual7_blocks(out)

        out_residual5 = self.residual5_blocks(out)
        out = self.interpolation1(out_residual5) + y
        # 248*216
        mask1 = self.residual8_blocks(out)

        masks = []
        masks.append(mask1)
        masks.append(mask2)
        masks.append(mask3)

        return tuple(masks)


class SoftMaskEncoder1(nn.Module):
    """Soft mask encoder 1.

    Args:
        input_channels (int): Input channels.
        output_channels (int): Output channels.
    """

    # input size is 248*216
    def __init__(self, in_channels=128, out_channels=128, size1=(248, 216), \
                 size2=(124, 108), size3=(62, 54)):
        super(SoftMaskEncoder1, self).__init__()

        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 124*108
        self.residual1_blocks = ResidualBlock(in_channels, out_channels)

        self.skip1_connection_residual_block = ResidualBlock(out_channels, out_channels)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 62*54
        self.residual2_blocks = ResidualBlock(out_channels, out_channels)

        self.skip2_connection_residual_block = ResidualBlock(out_channels, out_channels)

        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 31*27
        self.residual3_blocks = nn.Sequential(
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels)
        )

        self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)
        self.residual4_blocks = ResidualBlock(out_channels, out_channels)
        # 62*54

        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)
        self.residual5_blocks = ResidualBlock(out_channels, out_channels)
        # 124*108

        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.residual6_blocks = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Forward function.
        
        Args:
            x (torch.Tensor): Input with shape (N, c, H, W)

        Returns:
            torch.Tensor: mask output.
        """
        # 248*216
        y = self.first_residual_blocks(x)

        out_mpool1 = self.mpool1(x)
        # 124*108
        out_residual1 = self.residual1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_residual1)
        out_mpool2 = self.mpool2(out_residual1)
        # 62*54
        out_residual2 = self.residual2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_residual2)
        out_mpool3 = self.mpool3(out_residual2)
        # 31*27
        out_residual3 = self.residual3_blocks(out_mpool3)
        out = self.interpolation3(out_residual3) + out_skip2_connection        
        # 62*54

        out_residual4 = self.residual4_blocks(out)
        out = self.interpolation2(out_residual4) + out_skip1_connection        
        # 124*108

        out_residual5 = self.residual5_blocks(out)
        out = self.interpolation1(out_residual5) + y
        # 248*216
        out_last = self.residual6_blocks(out)  

        return out_last


class SoftMaskEncoder2(nn.Module):
    """Soft mask encoder 2.

    Args:
        input_channels (int): Input channels.
        output_channels (int): Output channels.
    """

    # input size is 124*108
    def __init__(self, in_channels=128, out_channels=128, size1=(124, 108), size2=(62, 54)):
        super(SoftMaskEncoder2, self).__init__()

        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 62*54
        self.residual1_blocks = ResidualBlock(in_channels, out_channels)

        self.skip1_connection_residual_block = ResidualBlock(out_channels, out_channels)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 31*27
        self.residual2_blocks = nn.Sequential(
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels)
        )

        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)
        self.residual3_blocks = ResidualBlock(out_channels, out_channels)
        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.residual4_blocks = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias = False),
            nn.Sigmoid()
        )


    def forward(self, x):
        """Forward function.
        
        Args:
            x (torch.Tensor): Input with shape (N, c, H, W)

        Returns:
            torch.Tensor: mask output.
        """
        # 124*108
        y = self.first_residual_blocks(x)

        out_mpool1 = self.mpool1(x)
        # 62*54
        out_residual1 = self.residual1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_residual1)
        out_mpool2 = self.mpool2(out_residual1)
        # 31*27
        out_residual2 = self.residual2_blocks(out_mpool2)
        out = self.interpolation2(out_residual2) + out_skip1_connection
        # 62*54

        out_residual3 = self.residual3_blocks(out)
        out = self.interpolation1(out_residual3) + y
        # 124*108

        out_last = self.residual4_blocks(out)

        return out_last


class SoftMaskEncoder3(nn.Module):
    """Soft mask encoder 3.

    Args:
        input_channels (int): Input channels.
        output_channels (int): Output channels.
    """
    # input size is 124*108
    def __init__(self, in_channels=256, out_channels=256, size1=(62, 54)):
        super(SoftMaskEncoder3, self).__init__()

        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 31*27
        self.residual1_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(out_channels, out_channels)
        )

        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.residual2_blocks = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias = False),
            nn.Sigmoid()
        )


    def forward(self, x):
        """Forward function.
        
        Args:
            x (torch.Tensor): Input with shape (N, c, H, W)

        Returns:
            torch.Tensor: mask output.
        """
        # 62*54
        y = self.first_residual_blocks(x)

        out_mpool1 = self.mpool1(x)
        # 31*27
        out_residual1 = self.residual1_blocks(out_mpool1)
        out = self.interpolation1(out_residual1) + y
        # 62*54

        out_last = self.residual2_blocks(out)

        return out_last


def test():
    """test."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    x = torch.randn(3, 128, 248, 216).float().to(device)
    print(x.shape)
    mask1 = SoftMaskEncoder1(128, 128).to(device)
    m1 = mask1(x)
    print(m1.shape)

    y = torch.randn(3, 64, 124, 108).float().to(device)
    mask2 = SoftMaskEncoder2(64, 128).to(device)
    m2 = mask2(y)
    print(m2.shape)

    z = torch.randn(3, 128, 62, 54).float().to(device)
    mask3 = SoftMaskEncoder3(128, 256).to(device)
    m3 = mask3(z)
    print(m3.shape)

    f = torch.randn(3, 128, 248, 216).float().to(device)
    soft_mask = SoftMask(128, [128, 128, 256]).to(device)
    masks = soft_mask(f)
    print(masks[0].shape)
    print(masks[1].shape)
    print(masks[2].shape)


if __name__ == '__main__':
    test()