import torch
import torchvision.ops as tv_ops


def cxcywh_to_xyxy(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
    """Convert normalized CXCYWH to absolute XYXY on CPU."""
    boxes_abs = boxes.clone()
    boxes_abs[:, [0, 2]] *= w
    boxes_abs[:, [1, 3]] *= h
    boxes_abs[:, :2] -= boxes_abs[:, 2:] / 2  # cx,cy -> xmin,ymin
    boxes_abs[:, 2:] += boxes_abs[:, :2]  # + (w,h) -> xmax,ymax
    return boxes_abs


def nms_xyxy(boxes_xyxy: torch.Tensor, scores: torch.Tensor, iou_thr: float = 0.5) -> torch.Tensor:
    """Thin wrapper around torchvision NMS that keeps indices."""
    keep_idx: torch.Tensor = tv_ops.nms(boxes_xyxy, scores, iou_thr)
    return keep_idx


def box_area_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Compute box area for absolute XYXY tensors (no grad)."""
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
    return wh[:, 0] * wh[:, 1]
