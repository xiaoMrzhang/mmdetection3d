from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import load_checkpoint
from torch import nn as nn
import torch

from mmdet.models import BACKBONES
from mmdet3d.utils.soft_mask import SoftMask

@BACKBONES.register_module()
class SECONDRanMask(nn.Module):
    """Backbone network for SECOND with residual attention network

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    def __init__(self,
                 in_channels=128,
                 out_channels=[128, 128, 256],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 conv_cfg=dict(type='Conv2d', bias=False)):
        super(SECONDRanMask, self).__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = [
                build_conv_layer(
                    conv_cfg,
                    in_filters[i],
                    out_channels[i],
                    3,
                    stride=layer_strides[i],
                    padding=1),
                build_norm_layer(norm_cfg, out_channels[i])[1],
                nn.ReLU(inplace=True),
            ]
            for j in range(layer_num):
                block.append(
                    build_conv_layer(
                        conv_cfg,
                        out_channels[i],
                        out_channels[i],
                        3,
                        padding=1))
                block.append(build_norm_layer(norm_cfg, out_channels[i])[1])
                block.append(nn.ReLU(inplace=True))

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        first_layer_conv = build_conv_layer(
                conv_cfg,
                in_filters[0],
                out_channels[0],
                3,
                stride=2,
                padding=1)
        first_bn = build_norm_layer(norm_cfg, out_channels[0])[1]
        first_relu = nn.ReLU(inplace=True)
        soft_mask = SoftMask(in_channels, out_channels)
        self.soft_mask_block = nn.Sequential(first_layer_conv, first_bn, first_relu, soft_mask)

    def init_weights(self, pretrained=None):
        """Initialize weights of the 2D backbone."""
        # Do not initialize the conv layers
        # to follow the original implementation
        if isinstance(pretrained, str):
            from mmdet3d.utils import get_root_logger
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        masks = self.soft_mask_block(x)
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            # x = (1 + masks[i]) * x
            x = torch.mul(x, nn.Sigmoid()(masks[i])) + x
            outs.append(x)
        return tuple([outs, masks])