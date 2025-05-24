import numpy as np
import torch
import torchvision.ops as tv_ops

Mask = np.ndarray | torch.Tensor  # H×W   bool / {0,255}
Box = list[int]


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
        return [0, 0, 0, 0]  # empty mask

    ys, xs = np.where(mask_bool)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    return [x_min, y_min, x_max, y_max]


def scale_box(box: Box, scale: float, w: int, h: int) -> Box:
    """Scale a box by preserving its center and scaling width/height, constrained by image bounds."""
    x_min, y_min, x_max, y_max = box

    # Calculate center of the box
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    # Calculate current width and height
    width = x_max - x_min
    height = y_max - y_min

    # Scale width and height
    new_width = width * scale
    new_height = height * scale

    # Calculate new box coordinates based on center and scaled dimensions
    new_x_min = center_x - new_width / 2
    new_y_min = center_y - new_height / 2
    new_x_max = center_x + new_width / 2
    new_y_max = center_y + new_height / 2

    # Ensure box stays within image bounds
    new_x_min = max(0, int(new_x_min))
    new_y_min = max(0, int(new_y_min))
    new_x_max = min(w, int(new_x_max))
    new_y_max = min(h, int(new_y_max))

    return [new_x_min, new_y_min, new_x_max, new_y_max]
