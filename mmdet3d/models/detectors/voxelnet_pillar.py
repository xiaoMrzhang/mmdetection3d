import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F
import numpy as np
import numba
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
        self.heatmap = None

    def extract_feat(self, points, img_metas, gt_bboxes_3d=None):
        """Extract features from points."""
        masks = None
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)

        # fake_voxels = torch.ones([32000, 20, 5],dtype=torch.float32).cuda()
        # fale_num_points = torch.ones([32000],dtype=torch.float32).cuda()
        # fake_coors = torch.ones([32000, 5],dtype=torch.float32).cuda()
        # pts_voxel_encoder = torch.jit.trace(self.voxel_encoder, (fake_voxels, fale_num_points, fake_coors))
        # pts_voxel_encoder.save("/home/zhangxiao/code/mmdetection3d/work_dirs/save_path/pts_voxel_encoder.zip")

        batch_size = coors[-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, coors, batch_size)
        # fake_batch_size = torch.tensor(1)
        # pts_middle_encoder = torch.jit.trace(self.middle_encoder, (voxel_features, coors, fake_batch_size))
        # pts_middle_encoder.save("/home/zhangxiao/code/mmdetection3d/work_dirs/save_path/pts_middle_encoder.zip")

        # pts_backbone = torch.jit.trace(self.backbone, x)
        # pts_backbone.save("/home/zhangxiao/code/mmdetection3d/work_dirs/save_path/pts_backbone.zip")
        x = self.backbone(x)

        if self.with_neck:
            # pts_neck = torch.jit.trace(self.neck, [x])
            # pts_neck.save("/home/zhangxiao/code/mmdetection3d/work_dirs/save_path/pts_neck.zip")
            x, masks = self.neck(x)

        if gt_bboxes_3d is not None and masks is not None:
            # self.heatmap = self.generate_gaussion_heatmap(masks[0].size(), coors, segmask_maps)
            if 496 % masks[0].size(2) == 0:
                scale = 496 // masks[0].size(2)
                segmask_maps = self.generate_mask(points, vis_voxel_size=[0.16, 0.16, 4],
                                        vis_point_range=[0, -39.68, -3, 69.12, 39.68, 1],
                                        boxes=gt_bboxes_3d, scale=scale)
            else:
                scale = 468 // masks[0].size(2)
                segmask_maps = self.generate_mask(points, vis_voxel_size=[0.32, 0.32, 6],
                                        vis_point_range=[-74.88, -74.88, -2, 74.88, 74.88, 4],
                                        boxes=gt_bboxes_3d, scale=scale)
            gaussian = self.gaussian_2d((2 * 6 + 1, 2 * 6 + 1), sigma=6/6)
            self.heatmap = generate_gaussion_heatmap_array(np.array(masks[0].size()),
                                                            coors.cpu().numpy(),
                                                            segmask_maps, gaussian, scale)
            self.heatmap = torch.from_numpy(self.heatmap)

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
        x, masks = self.extract_feat(points, img_metas, gt_bboxes_3d=gt_bboxes_3d)
        # x, masks = self.extract_feat(points, img_metas)
        if self.heatmap is not None:
            heatmap_seg = self.heatmap.to(x[0].device).unsqueeze(1)
        else:
            heatmap_seg = None

        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        # loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        # losses = self.bbox_head.loss(*loss_inputs)
        if masks is not None and heatmap_seg is not None:
            hm_loss = self.backbone.focal_loss(masks[0], heatmap_seg)
            losses.update(hm_loss)
        if np.random.rand() > 1:
            for i in range(len(masks)):
                save_mask = np.zeros((masks[i].size(0), masks[i].size(1), masks[i].size(2)*2, masks[i].size(3)*2))
                save_mask[:, :, 0:masks[i].size(2), 0:masks[i].size(3)] = masks[i].cpu().data.numpy()
                if heatmap_seg.size() == masks[i].size():
                    save_mask[:, :, masks[i].size(2):, masks[i].size(3):] = heatmap_seg.cpu().data.numpy()
                np.save("/home/zhangxiao/tmp/" + str(i) + ".npy", save_mask)
        return losses

    def simple_test(self, points, img_metas, imgs=None, bev_seg_image=None,
                    rescale=False, gt_bboxes_3d=None):
        """Test function without augmentaiton."""
        # segmask_maps = self.generate_mask(points, vis_voxel_size=[0.16, 0.16, 4],
        #                         vis_point_range=[0, -39.68, -3, 69.12, 39.68, 1],
        #                         boxes=gt_bboxes_3d)
        x, masks = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        # traced_script_module = torch.jit.trace(self.bbox_head, [x])
        # traced_script_module.save("/home/zhangxiao/code/mmdetection3d/work_dirs/save_path/pts_bbox_head.zip")

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
        w = int((vis_point_range[4] - vis_point_range[1]) / vis_voxel_size[1] + 0.5)
        h = int((vis_point_range[3] - vis_point_range[0]) / vis_voxel_size[0] + 0.5)
        segmask_maps = np.zeros((len(points), int(w/scale), int(h/scale)))
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
            segmask = cv2.resize(segmask, (int(segmask.shape[1]/scale), int(segmask.shape[0]/scale)), interpolation=cv2.INTER_NEAREST)
            segmask_maps[i] = segmask[:, :, 0] / 255.
        # cv2.imwrite("/home/zhangxiao/test_2.png", segmask_maps[1]*255)
        # bev_map = kitti_vis(points[0].data.cpu().numpy(), vis_voxel_size=vis_voxel_size,
        #                     vis_point_range=vis_point_range, boxes=boxes[0].tensor.detach().cpu().numpy())
        return segmask_maps

    def generate_gaussion_heatmap(self, heatmap_size, coors_gpu, segmask_maps, scale=2):
        """generate heatmap
        """
        coors = coors_gpu.cpu()
        heatmap = torch.zeros((heatmap_size[0], heatmap_size[2], heatmap_size[3]))
        radius = 5
        gaussian = self.gaussian_2d((2 * radius + 1, 2 * radius + 1), sigma=radius/6)
        for i in range(coors.size()[0]):
            batch_idx = coors[i][0]
            center = coors[i][-2:] // scale
            if segmask_maps[batch_idx, center[0], center[1]] == 0:
                continue
            # draw_heatmap_gaussian(heatmap[batch_idx], center, radius)
            self.draw_heatmap_gaussian(heatmap[batch_idx], center, radius, gaussian)
        # cv2.imwrite("/home/zhangxiao/test_2.png", (self.heatmap[1] * 255).numpy().astype(np.uint8))
        # cv2.imwrite("/home/zhangxiao/test_1.png", (heatmap[1] * 255).numpy().astype(np.uint8))
        return heatmap



    def gaussian_2d(self, shape, sigma=1):
        """Generate gaussian map.

        Args:
            shape (list[int]): Shape of the map.
            sigma (float): Sigma to generate gaussian map.
                Defaults to 1.

        Returns:
            np.ndarray: Generated gaussian map.
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_heatmap_gaussian(self, heatmap, center, radius, gaussian, k=1):
        """Get gaussian masked heatmap.

        Args:
            heatmap (torch.Tensor): Heatmap to be masked.
            center (torch.Tensor): Center coord of the heatmap.
            radius (int): Radius of gausian.
            K (int): Multiple of masked_gaussian. Defaults to 1.

        Returns:
            torch.Tensor: Masked heatmap.
        """

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = torch.from_numpy(
            gaussian[radius - top:radius + bottom,
                    radius - left:radius + right]).to(heatmap.device,
                                                    torch.float32)
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

@numba.jit(nopython=True)
def generate_gaussion_heatmap_array(heatmap_size, coors, segmask_maps, gaussian, scale=2):
    """generate gaussiin heatmap

    Args:
        heatmap_size (numpy array): [b, c , w, h]
        coors (list): pillar coors
        segmask_maps (np.ndarray): [description]
        gaussian (np.ndarray): gaussion heatmap generate by radius
        scale (int, optional): [description]. Defaults to 2.

    Returns:
        [np.ndarray]: gaussion heatmap
    """
    heatmap = np.zeros((heatmap_size[0], heatmap_size[2], heatmap_size[3]))
    radius = 6
    for i in range(coors.shape[0]):
        batch_idx = coors[i][0]
        center = coors[i][-2:][::-1] // scale
        if segmask_maps[batch_idx, center[1], center[0]] == 0:
            continue
        draw_heatmap_gaussian_array(heatmap[batch_idx], center, radius, gaussian)
    # import pdb;pdb.set_trace()
    # cv2.imwrite("/home/zhangxiao/test_2.png", (heatmap[1] * 255).astype(np.uint8))
    # cv2.imwrite("/home/zhangxiao/test_1.png", (segmask_maps[3] * 255).astype(np.uint8))
    # cv2.imwrite("/home/zhangxiao/test_3.png", (segmask_maps[3] * heatmap[3] * 255).astype(np.uint8))
    return heatmap

@numba.jit(nopython=True)
def draw_heatmap_gaussian_array(heatmap, center, radius, gaussian, k=1):
    """Get gaussian masked heatmap.

    Args:
        heatmap (np.ndarray): Heatmap to be masked.
        center (np.ndarray): Center coord of the heatmap.
        radius (int): Radius of gausian.
        K (int): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        np.ndarray: Masked heatmap.
    """

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                                radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        masked_heatmap = np.maximum(masked_heatmap, masked_gaussian * k)
        heatmap[y - top:y + bottom, x - left:x + right] = masked_heatmap
    return heatmap