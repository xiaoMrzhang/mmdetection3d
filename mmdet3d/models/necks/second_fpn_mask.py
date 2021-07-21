import numpy as np
import torch
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, is_norm, kaiming_init)
from mmcv.runner import auto_fp16, force_fp32
from torch import nn as nn
import torch.nn.functional as F

from mmdet.models import NECKS


@NECKS.register_module()
class SECONDFPNMASK(nn.Module):
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
        super(SECONDFPNMASK, self).__init__()
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False

        deblocks = []
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

            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride)

            deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, out_channel)[1],
                                    nn.ReLU(inplace=True))
            deblocks.append(deblock)

            conv_1 = build_conv_layer(conv_cfg, in_channels=out_channel, out_channels=out_channel,
                                      kernel_size=1, stride=1)
           
        self.deblocks = nn.ModuleList(deblocks)
        self.binary_cls = nn.Sequential(
            build_conv_layer(conv_cfg, sum(out_channels), sum(out_channels), 3, padding=1),
            build_norm_layer(norm_cfg, sum(out_channels))[1],
            nn.ReLU(inplace=True),
            build_conv_layer(conv_cfg, sum(out_channels), sum(out_channels), 1, 1),
            build_norm_layer(norm_cfg, sum(out_channels))[1],
            nn.ReLU(inplace=True),
            build_conv_layer(conv_cfg, sum(out_channels), 1, 1),
            nn.Sigmoid()
        )


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

        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]

        # mask = self.binary_cls(out)
        # return tuple([[out], [mask]])
        return [out]


    @force_fp32(apply_to=('prediction'))
    def focal_loss(self, prediction, target):
        loss_dict = dict()
        self.alpha = 2
        self.beta = 4
        positive_index = target.eq(1).float()
        negative_index = target.lt(1).float()
        negative_weights = torch.pow(1 - target, self.beta)
        loss = 0.
        # prediction = torch.clamp(prediction, 1e-3, .999)
        positive_loss = torch.log(prediction + 1e-6) \
                        * torch.pow(1 - prediction, self.alpha) * positive_index
        negative_loss = torch.log(1 - prediction + 1e-6) \
                        * torch.pow(prediction, self.alpha) * negative_weights * negative_index

        num_positive = positive_index.float().sum()
        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()

        if num_positive == 0:
            loss -= negative_loss
        else:
            loss -= (positive_loss + negative_loss) / num_positive
        loss_dict["loss_heatmap"] = loss

        # dice loss
        # intersection = (target * prediction).sum(axis=[1,2,3])
        # dice_score = (2 * intersection + 1) / (target.sum(axis=[1,2,3]) + prediction.sum(axis=[1,2,3]) + 1)
        # dice_loss = 1 - torch.mean(dice_score, axis=0)
        # loss_dict["loss_dice"] = dice_loss * 0.2
        # if torch.isnan(loss) or torch.isnan(dice_loss):
        #     import pdb;pdb.set_trace()

        return loss_dict