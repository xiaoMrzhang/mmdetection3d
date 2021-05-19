import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F
import numpy as np
import cv2

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d, draw_heatmap_gaussian, gaussian_radius
from mmdet3d.ops import Voxelization
from mmdet.models import DETECTORS
from .. import builder
from .single_stage import SingleStage3DDetector
from mmdet3d.core.visualizer.show_result import kitti_vis, center_to_corner_box2d

@DETECTORS.register_module()
class VoxelNetPillar(SingleStage3DDetector):
    r"""`VoxelNetPillar <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(VoxelNetPillar, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.middle_encoder = builder.build_middle_encoder(middle_encoder)

    def extract_feat(self, points, img_metas):
        """Extract features from points."""
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, coors, batch_size)
        # self.heatmap = self.generate_gaussion_heatmap(x.size(), coors, scale=1)
        x, masks = self.backbone(x)
        self.heatmap = self.generate_gaussion_heatmap(masks[0].size(), coors)

        # This for cfa module
        # x = self.backbone(x, voxel_features, coors)
        # voxel_context = self.cfa(voxel_features, coors) 这部分写到backbone里面？

        # x = torch.cat([x, voxel_context], dim=1) 这部分写到neck里面？
        if self.with_neck:
            x = self.neck(x)
        return x, masks

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      bev_seg_image=None,
                      gt_bboxes_ignore=None):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sample
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        segmask_maps = self.generate_mask(points, vis_voxel_size=[0.16, 0.16, 4],
                                vis_point_range=[0, -39.68, -3, 69.12, 39.68, 1],
                                boxes=gt_bboxes_3d)
        x, masks = self.extract_feat(points, img_metas)

        if segmask_maps is not None:
            heatmap_seg = self.heatmap * torch.from_numpy(segmask_maps)
            import pdb;pdb.set_trace()
            heatmap_seg = heatmap_seg.to(x[0].device).unsqueeze(1)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        hm_loss = self.backbone.loss(masks[0], heatmap_seg)
        # loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        # losses = self.bbox_head.loss(*loss_inputs)
        return losses

    def simple_test(self, points, img_metas, imgs=None, bev_seg_image=None,
                    rescale=False, gt_bboxes_3d=None):
        """Test function without augmentaiton."""
        # segmask_maps = self.generate_mask(points, vis_voxel_size=[0.16, 0.16, 4],
        #                         vis_point_range=[0, -39.68, -3, 69.12, 39.68, 1],
        #                         boxes=gt_bboxes_3d)
        x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        # bbox_list = self.bbox_head.get_bboxes(
        #     outs, img_metas, rescale=rescale)
        # bbox_results = [
        #     bbox3d2result(bboxes, scores, labels)
        #     for bboxes, scores, labels in bbox_list
        # ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        feats = self.extract_feats(points, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]

    def generate_mask(self, points, vis_voxel_size, vis_point_range, boxes, scale=2):
        """generate segmask by given pointcloud and bounding boxes

        Args:
            points (torch.Tensor): point cloud batch
            vis_voxel_size (list): voxel size
            vis_point_range (list): point cloud range
            boxes (LiDARInstance3DBoxes): gt boxes
        """
        if boxes is None:
            return None
        w = int((vis_point_range[4] - vis_point_range[1]) / vis_voxel_size[1])
        h = int((vis_point_range[3] - vis_point_range[0]) / vis_voxel_size[0])
        segmask_maps = np.zeros((len(points), int(w/scale), int(h/scale)))
        # import pdb;pdb.set_trace()
        for i in range(segmask_maps.shape[0]):
            vis_point_range = np.array(vis_point_range)
            if isinstance(boxes[i], list):
                assert len(boxes[i]) == 1
                current_bbox = boxes[i][0].tensor.detach().cpu().numpy()
            else:
                current_bbox = boxes[i].tensor.detach().cpu().numpy()
            bev_corners = center_to_corner_box2d(
                current_bbox[:, [0, 1]], current_bbox[:, [3, 4]], current_bbox[:, 6])
            bev_corners -= vis_point_range[:2]
            bev_corners *= np.array(
                (w, h))[::-1] / (vis_point_range[3:5] - vis_point_range[:2])
            segmask = np.zeros((w, h, 3))
            segmask = cv2.drawContours(segmask, bev_corners.astype(np.int), -1, 255, -1)
            segmask = cv2.resize(segmask, (int(segmask.shape[1]/2), int(segmask.shape[0]/2)), interpolation=cv2.INTER_NEAREST)
            segmask_maps[i] = segmask[:, :, 0] / 255.
        # cv2.imwrite("/home/zhangxiao/test_2.png", segmask_maps[1]*255)
        # bev_map = kitti_vis(points[0].data.cpu().numpy(), vis_voxel_size=vis_voxel_size,
        #                     vis_point_range=vis_point_range, boxes=boxes[0].tensor.detach().cpu().numpy())
        # import pdb;pdb.set_trace()
        return segmask_maps

    def generate_gaussion_heatmap(self, heatmap_size, coors_gpu, scale=2):
        coors = coors_gpu.cpu()
        heatmap = torch.zeros((heatmap_size[0], heatmap_size[2], heatmap_size[3]))
        for i in range(coors.size()[0]):
            batch_idx = coors[i][0]
            center = coors[i][-2:] // scale
            radius = gaussian_radius((torch.tensor(heatmap_size[2]), torch.tensor(heatmap_size[3])), min_overlap=0.1)
            # radius = max(2, int(radius))
            radius = 5
            draw_heatmap_gaussian(heatmap[batch_idx], center, radius)
        # cv2.imwrite("/home/zhangxiao/test_2.png", (self.heatmap[1] * 255).numpy().astype(np.uint8))
        # cv2.imwrite("/home/zhangxiao/test_2.png", (heatmap_seg[1] * 255).numpy().astype(np.uint8))
        return heatmap