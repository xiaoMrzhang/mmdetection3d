"""Second FPN with Residual attention"""

import numpy as np
import torch
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, is_norm, kaiming_init)
from mmcv.runner import auto_fp16
from torch import nn as nn
import torch.nn.functional as F

from mmdet.models import NECKS


@NECKS.register_module()
class SECONDFPN_RAN(nn.Module):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(self,
                 in_channels=[128, 128, 256],
                 out_channels=[256, 256, 256],
                 upsample_strides=[1, 2, 4],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 upsample_cfg=dict(type='deconv', bias=False),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 use_conv_for_no_stride=False):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(SECONDFPN_RAN, self).__init__()
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False

        deblocks = []
        spitals = []
        channel_blocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i])
                conv_1 = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i])
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride)
                conv_1 = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride)

            deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, out_channel)[1],
                                    nn.ReLU(inplace=True))
            deblocks.append(deblock)

            conv_2 = build_conv_layer(conv_cfg, in_channels=out_channel, out_channels=out_channel,
                                      kernel_size=1, stride=1)
            spital = nn.Sequential(conv_1, build_norm_layer(norm_cfg, out_channel)[1], nn.ReLU(inplace=True),
                                  conv_2, nn.Sigmoid())
            # [64, 64, 128]
            conv_c = build_conv_layer(conv_cfg, in_channels=in_channels[i], out_channels=out_channel,
                                      kernel_size=1, stride=1)
            channel_layer = nn.Sequential(conv_c, nn.Sigmoid())
            spitals.append(spital)
            channel_blocks.append(channel_layer)

        self.deblocks = nn.ModuleList(deblocks)
        self.spitals = nn.ModuleList(spitals)
        self.channel_blocks = nn.ModuleList(channel_blocks)

    def init_weights(self):
        """Initialize weights of FPN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif is_norm(m):
                constant_init(m, 1)

    @auto_fp16()
    def forward(self, x, seg_mask=None):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]
        if seg_mask is None:
            ras = [spital(x[i]) for i, spital in enumerate(self.spitals)]
        elif len(seg_mask) == 3:
            # ras = [spital(seg_mask[i]) for i, spital in enumerate(self.spitals)]
            ras = [seg_mask[0],
                   F.interpolate(seg_mask[1], scale_factor=2, mode='bilinear'),
                   F.interpolate(seg_mask[2], scale_factor=4, mode='bilinear')]
            ras = [channel_block(ras[i]) for i, channel_block in enumerate(self.channel_blocks)]
        else:
            if isinstance(seg_mask, np.ndarray):
                seg_mask = torch.from_numpy(seg_mask).to(x[0].device).float()
                seg_mask = seg_mask.unsqueeze(1)
            if seg_mask.size(2) != ups[0].size(2):
                scale_factor = ups[0].size(2) / seg_mask.size(2)
                ras = [F.interpolate(seg_mask, scale_factor=scale_factor, mode='bilinear')]
            else:
                ras = [seg_mask]
        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
            att = torch.cat(ras, dim=1)
        else:
            out = ups[0]
            att = ras[0]
        out = torch.mul(out, att) + out
        return [out]
