from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import load_checkpoint
import torch
from torch import nn as nn
import torch.nn.functional as F
import math

from mmdet.models import BACKBONES
from mmdet3d.models.backbones.sa_block import SA_block

@BACKBONES.register_module()
class SECOND_DSA(nn.Module):
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
                 dropout = 0.3,
                 in_dims = 64,
                 num_key_points = 2048,
                 local_context_mlps = [[64]],
                 local_context_deform_radius = [3],
                 local_context_pool_radius = [2],
                 local_context_nsample = [16],
                 local_context_pool_method = 'max_pool',
                 decode_mlps = [[64]],
                 decode_pool_radius = [1.6],
                 decode_nsample = [16],
                 decode_pool_method = 'max_pool',
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
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        # layers to deform + aggregate local features
        mlps = local_context_mlps
        for k in range(len(mlps)):
            mlps[k] = [in_channels] + mlps[k]
        self.adapt_context = pointnet2_stack_modules.StackSAModuleMSGAdapt(
            radii=local_context_pool_radius,
            deform_radii=local_context_deform_radius,
            nsamples=local_context_nsample,
            mlps=mlps,
            use_xyz=True,
            pool_method=local_context_pool_method,
            pc_range=self.point_cloud_range,
        )

        # UnPool layers
        mlps_decode = decode_mlps
        for k in range(len(mlps_decode)):
            mlps_decode[k] = [in_dims] + mlps_decode[k]
        self.decode = pointnet2_stack_modules.StackSAModuleMSGDecode(
            radii=decode_pool_radius,
            nsamples=decode_nsample,
            mlps=mlps_decode,
            use_xyz=True,
            pool_method=decode_pool_method,
        )

       # self-attention layers to operate on deformed pillars
        self.self_full_fast_attn = SA_block(inplanes=in_dims, planes=in_dims)
        self.reduce_dim = nn.Sequential(nn.Conv1d(2*in_dims, in_dims, kernel_size=1),
                                        nn.BatchNorm1d(in_dims),
                                        nn.ReLU(inplace=True),
                                        nn.Conv1d(in_dims, in_dims, kernel_size=1),
                                        nn.BatchNorm1d(in_dims),
                                        nn.ReLU(inplace=True)
                                        )
        self.self_attn_ms1 = SA_block(inplanes=2*in_dims, planes=2*in_dims)
        self.self_attn_ms2 = SA_block(inplanes=2*in_dims, planes=2*in_dims)

    def get_keypoints(self, batch_size, coords, src_points):
        """
        Select keypoints, i.e. a subset of pillar coords to deform, aggregate local features and then attend to.
        :param batch_size:
        :param coords:
        :param src_points:
        :return: B x num_keypoints x 3
        """
        keypoints_list = []
        for bs_idx in range(batch_size):
            bs_mask = (coords[:, 0] == bs_idx)
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)  # (1, N, 3)
            cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(sampled_points[:, :, 0:3],
                                                                      self.model_cfg.NUM_KEYPOINTS).long()
            if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                empty_num = self.model_cfg.NUM_KEYPOINTS - sampled_points.shape[1]
                cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]
            keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)
            keypoints_list.append(keypoints)
        keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3)
        return keypoints

    def get_local_keypoint_features(self, keypoints, pillar_center, pillar_features, coords):
        """
        Get local features of deformed pillar-subset/keypoints.
        :param keypoints:
        :param pillar_center:
        :param pillar_features:
        :param coords:
        :return: B x num_keypoints X C
        """
        batch_size, num_keypoints, _ = keypoints.shape
        new_xyz = keypoints.view(-1, 3)
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int().fill_(num_keypoints)

        xyz_batch_cnt = torch.zeros([batch_size]).int().cuda()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (coords[:, 0] == bs_idx).sum()

        def_xyz, local_features = self.adapt_context(
            xyz=pillar_center,
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=pillar_features
        )
        def_xyz = def_xyz.view(batch_size, num_keypoints, -1)
        local_features = local_features.view(batch_size, num_keypoints, -1)
        return def_xyz, local_features

    def get_context_features(self, batch_size, pillars, local_features, coords):
        batch_global_features = []
        for batch_idx in range(batch_size):
            init_idx = batch_idx * self.model_cfg.NUM_KEYPOINTS
            local_feat = local_features[init_idx:init_idx + self.model_cfg.NUM_KEYPOINTS, :].unsqueeze(0)
            local_feat = local_feat.permute(0, 2, 1).contiguous()
            local_sa_feat = self.self_full_fast_attn(local_feat)

            batch_mask = coords[:, 0] == batch_idx
            pillar_feat = pillars[batch_mask, :].unsqueeze(0).permute(0, 2, 1).contiguous()

            attn_feat1 = self.self_attn1(pillar_feat, local_sa_feat)
            attn_feat2 = self.self_attn2(attn_feat1, local_sa_feat)
            context_pillar = attn_feat2.permute(0, 2, 1).contiguous().squeeze(0)

            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            spatial_feature = torch.zeros(
                self.model_cfg.NUM_BEV_FEATURES,
                self.nz * self.nx * self.ny,
                dtype=context_pillar.dtype,
                device=context_pillar.device)
            spatial_feature[:, indices] = context_pillar.t()
            batch_global_features.append(spatial_feature)

        context_pillar_features = torch.cat(batch_global_features, 0)
        context_pillar_features = context_pillar_features.view(batch_size, self.model_cfg.NUM_BEV_FEATURES * self.nz, self.ny, self.nx)
        return context_pillar_features

    def init_weights(self, pretrained=None):
        """Initialize weights of the 2D backbone."""
        # Do not initialize the conv layers
        # to follow the original implementation
        if isinstance(pretrained, str):
            from mmdet3d.utils import get_root_logger
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

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