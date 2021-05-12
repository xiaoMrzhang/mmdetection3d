from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import load_checkpoint
from torch import nn as nn
import torch

from mmdet.models import BACKBONES
from mmdet3d.utils.soft_mask import SoftMaskEncoder1, SoftMaskEncoder2, SoftMaskEncoder3

@BACKBONES.register_module()
class SECOND_RAN_ORI(nn.Module):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

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
        super(SECOND_RAN_ORI, self).__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.

        first_blocks = []
        trunk_blocks = []
        for i, layer_num in enumerate(layer_nums):
            # first block
            first_block = [
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
            first_block = nn.Sequential(*first_block)
            first_blocks.append(first_block)

            # trunk block
            trunk_block = []
            for j in range(layer_num):
                trunk_block.append(
                    build_conv_layer(
                        conv_cfg,
                        out_channels[i],
                        out_channels[i],
                        3,
                        padding=1))
                trunk_block.append(build_norm_layer(norm_cfg, out_channels[i])[1])
                trunk_block.append(nn.ReLU(inplace=True))
            trunk_block = nn.Sequential(*trunk_block)
            trunk_blocks.append(trunk_block)

        self.first_blocks = nn.ModuleList(first_blocks)
        self.trunk_blocks = nn.ModuleList(trunk_blocks)

        soft_mask1 = SoftMaskEncoder1(out_channels[0], out_channels[0])
        soft_mask2 = SoftMaskEncoder2(out_channels[1], out_channels[1])
        soft_mask3 = SoftMaskEncoder3(out_channels[2], out_channels[2])

        soft_masks = []
        soft_masks.append(soft_mask1)
        soft_masks.append(soft_mask2)
        soft_masks.append(soft_mask3)
        self.soft_masks = nn.ModuleList(soft_masks)

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

        outs = []
        for i in range(len(self.first_blocks)):
            x = self.first_blocks[i](x)
            trunk = self.trunk_blocks[i](x)
            mask = self.soft_masks[i](x)
            x = torch.mul(trunk, mask) + trunk
            outs.append(x)
        return tuple(outs)
