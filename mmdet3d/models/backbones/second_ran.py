from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import load_checkpoint, force_fp32
from torch import nn as nn
import torch
import numpy as np

from mmdet.models import BACKBONES
from mmdet3d.utils.soft_mask import SoftMask

@BACKBONES.register_module()
class SECOND_RAN(nn.Module):
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
        super(SECOND_RAN, self).__init__()
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
        soft_mask = SoftMask(in_channels, out_channels, out_type=2)
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
        # save_feature = True
        # if save_feature:
        #     import numpy as np
        #     for i in range(len(masks)):
        #         save_path = "/home/zhangxiao/tmp/" + str(i) + ".npy" 
        #         np.save(save_path, masks[i][0].cpu().data.numpy())
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            # x = torch.mul(x, masks[i]) + x
            outs.append(x)
        return tuple([outs, masks])

    @force_fp32(apply_to=('prediction'))
    def focal_loss(self, prediction, target):
        loss_dict = dict()
        self.alpha = 2
        self.beta = 4

        save_mask = np.zeros((prediction.size(0), prediction.size(2)*2, prediction.size(3)*2))
        save_mask[:, 0:prediction.size(2), 0:prediction.size(3)] = prediction[:, 0].cpu().data.numpy()
        save_mask[:, prediction.size(2):, prediction.size(3):] = target[:, 0].cpu().data.numpy()
        np.save("/home/zhangxiao/tmp/" + "1.npy", save_mask)

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
        intersection = (target.eq(1).float() * prediction).sum(axis=[1,2,3])
        dice_score = (2 * intersection + 1) / (target.eq(1).float().sum(axis=[1,2,3]) + prediction.sum(axis=[1,2,3]) + 1)
        dice_loss = 1 - torch.mean(dice_score, axis=0)
        loss_dict["loss_dice"] = dice_loss
        if torch.isnan(loss) or torch.isnan(dice_loss):
            import pdb;pdb.set_trace()

        return loss_dict

    @force_fp32(apply_to=('prediction'))
    def loss(self, prediction, target):
        positive_index = target.eq(1).float()
        loss = 0.
        loss_dict = dict()

        save_mask = np.zeros((prediction.size(0), prediction.size(2)*2, prediction.size(3)*2))
        save_mask[:, 0:prediction.size(2), 0:prediction.size(3)] = prediction[:, 0].cpu().data.numpy()
        save_mask[:, prediction.size(2):, prediction.size(3):] = target[:, 0].cpu().data.numpy()
        np.save("/home/zhangxiao/tmp/" + "1.npy", save_mask)

        positive_loss = torch.log(prediction + 1e-6) * positive_index
        negative_loss = torch.log(1 - prediction + 1e-6) * (1 - positive_index)
        num_positive = positive_index.float().sum()
        num_negative = (1 - positive_index).float().sum()
        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()

        bec_loss = -(positive_loss / (num_positive+1) + negative_loss / (num_negative+1))

        intersection = (target * prediction).sum(axis=[1,2,3])
        dice_score = (2 * intersection + 1) / (target.sum(axis=[1,2,3]) + prediction.sum(axis=[1,2,3]) + 1)
        dice_loss = 1 - dice_score.mean()

        loss_dict["loss_heatmap"] = bec_loss
        loss_dict["loss_dice"] = dice_loss
        if torch.isnan(bec_loss) or torch.isnan(dice_loss):
            import pdb;pdb.set_trace()
        return loss_dict
