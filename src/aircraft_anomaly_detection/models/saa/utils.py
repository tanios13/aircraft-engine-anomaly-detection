import numpy as np
import torch
import torchvision.ops as tv_ops

Mask = np.ndarray | torch.Tensor  # H×W   bool / {0,255}
Box = list[float]


def cxcywh_to_xyxy(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
    """Convert normalized CXCYWH to absolute XYXY on CPU."""
    boxes_abs = boxes.clone()
    boxes_abs[:, [0, 2]] *= w
    boxes_abs[:, [1, 3]] *= h
    boxes_abs[:, :2] -= boxes_abs[:, 2:] / 2  # cx,cy -> xmin,ymin
    boxes_abs[:, 2:] += boxes_abs[:, :2]  # + (w,h) -> xmax,ymax
    return boxes_abs


def nms_xyxy(boxes_xyxy: torch.Tensor, scores: torch.Tensor, iou_thr: float = 0.5) -> torch.IntTensor:
    """Thin wrapper around torchvision NMS that keeps indices."""
    keep_idx: torch.IntTensor = tv_ops.nms(boxes_xyxy, scores, iou_thr)
    return keep_idx


def box_area_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Compute box area for absolute XYXY tensors (no grad)."""
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
    return wh[:, 0] * wh[:, 1]


def mask_to_box(mask: Mask) -> Box:
    """Compute the tight AABB for a single binary mask.

    Args:
        mask: HxW array-like (bool or uint8/float). Non-zero → foreground.

    Returns:
        (x_min, y_min, x_max, y_max) or ``None`` if mask is empty.
    """
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = mask
    # ensure bool
    mask_bool = mask_np.astype(bool)
    if not mask_bool.any():
        return [0.0, 0.0, 0.0, 0.0]  # empty mask

    ys, xs = np.where(mask_bool)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    return [x_min, y_min, x_max, y_max]
