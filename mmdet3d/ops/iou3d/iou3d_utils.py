import torch

from . import iou3d_cuda


def boxes_iou_bev(boxes_a, boxes_b):
    """Calculate boxes IoU in the bird view.

    Args:
        boxes_a (torch.Tensor): Input boxes a with shape (M, 5).
        boxes_b (torch.Tensor): Input boxes b with shape (N, 5).

    Returns:
        ans_iou (torch.Tensor): IoU result with shape (M, N).
    """
    ans_iou = boxes_a.new_zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])))

    iou3d_cuda.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(),
                                 ans_iou)

    return ans_iou


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """Nms function with gpu implementation.

    Args:
        boxes (torch.Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of boxes with the shape of [N].
        thresh (int): Threshold.
        pre_maxsize (int): Max size of boxes before nms. Default: None.
        post_maxsize (int): Max size of boxes after nms. Default: None.

    Returns:
        torch.Tensor: Indexes after nms.
    """
    order = scores.sort(0, descending=True)[1]

    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes = boxes[order].contiguous()

    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = iou3d_cuda.nms_gpu(boxes, keep, thresh, boxes.device.index)
    keep = order[keep[:num_out].cuda(boxes.device)].contiguous()
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep


def nms_weighted_gpu(boxes, scores, thresh, det_boxes=None, pre_maxsize=None, post_max_size=None):
    order = scores.sort(0, descending=True)[1]

    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes = boxes[order].contiguous()
    if det_boxes is not None:
        det_boxes = det_boxes[order].contiguous()

    keep_boxes = torch.zeros_like(boxes)
    keep_det_boxes = torch.zeros_like(det_boxes)
    keep_scores = torch.zeros_like(scores)
    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    keep_pos = 0

    while(boxes.size(0)):
        ans_iou = boxes_iou_bev(boxes[0:1], boxes).squeeze(0)   #(1, N)
        selected = ans_iou >= thresh
        ###Hard code demo/1013044.bin
        selected[0] = True
        ###
        weights = scores[selected] * ans_iou[selected]

        boxes[0, :4] = (weights.unsqueeze(1) * boxes[selected, :4]).sum(0) / (weights.sum() + 1e-6)
        keep_boxes[keep_pos] = boxes[0]

        if det_boxes is not None:
            det_boxes[0, :4] = (weights.unsqueeze(1) * det_boxes[selected, :4]).sum(0) / (weights.sum())
            keep_det_boxes[keep_pos] = det_boxes[0]

        keep_scores[keep_pos] = scores[0]
        keep[keep_pos] = keep_pos
        keep_pos += 1

        boxes = boxes[~selected]
        scores = scores[~selected]
        det_boxes = det_boxes[~selected]
    if post_max_size is not None:
        keep_pos = min(keep_pos, post_max_size)
    return keep[:keep_pos], keep_det_boxes, keep_scores
    # selected = nms_gpu(boxes, scores, thresh, pre_maxsize=pre_maxsize, post_max_size=post_max_size)
    # weighted_bboxs[:, :4] = (scores * ious * boxes[:, :4])
    # weighted_bboxs = weighted_bboxs[selected]

def nms_normal_gpu(boxes, scores, thresh):
    """Normal non maximum suppression on GPU.

    Args:
        boxes (torch.Tensor): Input boxes with shape (N, 5).
        scores (torch.Tensor): Scores of predicted boxes with shape (N).
        thresh (torch.Tensor): Threshold of non maximum suppression.

    Returns:
        torch.Tensor: Remaining indices with scores in descending order.
    """
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = iou3d_cuda.nms_normal_gpu(boxes, keep, thresh,
                                        boxes.device.index)
    return order[keep[:num_out].cuda(boxes.device)].contiguous()
