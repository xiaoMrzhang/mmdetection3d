from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import load_checkpoint
import torch
from torch import nn as nn
import torch.nn.functional as F
import math

from mmdet.models import BACKBONES
from mmdet3d.models.backbones.sa_block import SA_block

class PositionalEncoding(nn.Module):
    """
    Positional encoding from https://github.com/wzlxjtu/PositionalEncoding2D
    """
    def __init__(self, d_model, height, width):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.height = height
        self.width = width
        # Not a parameter
        self.register_buffer('pos_table', self._positionalencoding2d())

    def _positionalencoding2d(self):
        """
        :return: d_model*height*width position matrix
        """
        if self.d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(self.d_model))
        pe = torch.zeros(self.d_model, self.height, self.width)
        # Each dimension use half of d_model
        d_model = int(self.d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., self.width).unsqueeze(1)
        pos_h = torch.arange(0., self.height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, self.height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, self.height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, self.width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, self.width)
        return pe

    def forward(self, x, coords):
        pos_encode = self.pos_table[:, coords[:, 2].type(torch.cuda.LongTensor), coords[:, 3].type(torch.cuda.LongTensor)]
        return x + pos_encode.permute(1, 0).contiguous().clone().detach()

@BACKBONES.register_module()
class SECOND_FSA(nn.Module):
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
                 grid_size = [432, 496, 1],
                 voxel_size = [0.16, 0.16, 4],
                 point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1],
                 dropout = 0.1,
                 in_dims = 64,
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 conv_cfg=dict(type='Conv2d', bias=False)):
        super(SECOND_FSA, self).__init__()
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

        self.nx, self.ny, self.nz = grid_size
        self.out_channel = in_channels
        self.position_enc = PositionalEncoding(in_dims, height=grid_size[1], width=grid_size[0])
        self.layer_norm = nn.LayerNorm(in_dims, eps=1e-6)
        self.self_attn1 = SA_block(inplanes=in_dims, planes=in_dims)
        self.self_attn2 = SA_block(inplanes=in_dims, planes=in_dims)

    def init_weights(self, pretrained=None):
        """Initialize weights of the 2D backbone."""
        # Do not initialize the conv layers
        # to follow the original implementation
        if isinstance(pretrained, str):
            from mmdet3d.utils import get_root_logger
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def add_context_to_pillars(self, pillar_features, coords, nx, ny, nz, in_channels=128):
        batch_size = coords[:, 0].max().int().item() + 1
        batch_context_features = []
        for batch_idx in range(batch_size):
            batch_mask = coords[:, 0] == batch_idx
            pillars = pillar_features[batch_mask, :].unsqueeze(0)

            # Apply pairwise self-attention on VFE pillar features
            context_pillar = self.self_attn1(pillars.permute(0, 2, 1).contiguous())
            context_pillar = self.self_attn2(context_pillar)
            context_pillar = context_pillar.permute(0, 2, 1).contiguous().squeeze(0)

            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            spatial_feature = torch.zeros(
                in_channels,
                nz * nx * ny,
                dtype=context_pillar.dtype,
                device=context_pillar.device)
            spatial_feature[:, indices] = context_pillar.t()
            batch_context_features.append(spatial_feature)

        context_pillar_features = torch.cat(batch_context_features, 0)
        context_pillar_features = context_pillar_features.view(batch_size, in_channels * nz, ny, nx)
        return context_pillar_features

    def forward(self, x, pillar_features, coors):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        spatial_features = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            spatial_features.append(x)

        # get position encoding for pillars
        pillar_pos_enc = self.position_enc(pillar_features, coors)
        pillar_pos_enc = self.layer_norm(pillar_pos_enc)

        # get context for every pillar
        context_features = self.add_context_to_pillars(pillar_pos_enc, coors, self.nx, self.ny, self.nz, self.out_channel)

        # generate down-sampled SA-features to concatenate with Conv in decoder_2d module
        pillar_context = [F.interpolate(context_features, scale_factor=0.5, mode='bilinear'),
                          F.interpolate(context_features, scale_factor=0.25, mode='bilinear'),
                          F.interpolate(context_features, scale_factor=0.125, mode='bilinear')]

        outs = []
        for i in range(len(self.blocks)):
            out = torch.cat([spatial_features[i], pillar_context[i]], dim=1)
            outs.append(out)
        return tuple(outs)

        # spatial_features = [torch.cat([spatial_features[0], F.interpolate(context_features, scale_factor=0.5, mode='bilinear')], dim=1),
        #                     torch.cat([spatial_features[1], F.interpolate(context_features, scale_factor=0.25, mode='bilinear')], dim=1),
        #                     torch.cat([spatial_features[2], F.interpolate(context_features, scale_factor=0.125, mode='bilinear')], dim=1),]
        # return tuple(spatial_features)