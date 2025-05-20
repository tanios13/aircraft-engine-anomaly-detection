from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from transformers import SamModel, SamProcessor

from aircraft_anomaly_detection.interface.model import SegmentorInterface


class SamSegmentorHF(SegmentorInterface):
    """Hugging Face wrapper around Segment-Anything Model (SAM).

    Usage pattern::

        seg = SamSegmentorHF()
        seg.set_image(pil_img)
        masks = seg.predict(boxes_xyxy)

    The implementation keeps the API intentionally minimal—no SAM-specific
    objects leak out—so any other segmentor can be swapped in provided it
    respects :class:`SegmentorInterface`.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam-vit-base",
        *,
        device: str | None = None,
    ) -> None:
        """Initialise the Hugging Face SAM backbone.

        Args:
            model_id: Hugging Face model card (e.g. ``"facebook/sam-vit-huge"``).
            device:   Torch device string.  Defaults to ``"cuda"`` if a GPU is
                available, otherwise ``"cpu"``.
        """
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(self.device).eval()

        self._encoded_inputs: dict[str, torch.Tensor] | None = None
        self._orig_shape: tuple[int, int] | None = None  # (H, W)
        self._best_iou_scores: NDArray[np.float32] | None = None

    def set_image(self, image: Image.Image) -> None:  # noqa: D401 – imperative style
        """Pre compute an image embedding for fast prompting.

        Args:
            image: Input RGB image as ``PIL.Image.Image``.
        """
        self._encoded_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        self._orig_shape = (image.height, image.width)
        self._best_iou_scores = None

    def predict(
        self,
        boxes_xyxy: np.ndarray,
        *,
        multimask_output: bool = False,
    ) -> np.ndarray:
        """Return boolean masks for each bounding box.

        Args:
            boxes_xyxy: Absolute XYXY boxes, shape ``(N, 4)``.
            multimask_output: If ``True`` produce three masks per box; else one.

        Raises:
            RuntimeError: If :py:meth:`set_image` has not been called.
        """
        if self._encoded_inputs is None or self._orig_shape is None:
            raise RuntimeError("set_image must be called before predict().")

        # Convert boxes to tensor and add batch dimension expected by SAM.
        boxes_tensor = torch.as_tensor(boxes_xyxy, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(
                **self._encoded_inputs,
                input_boxes=boxes_tensor,
                multimask_output=multimask_output,
            )

        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            self._encoded_inputs["original_sizes"].cpu(),
            self._encoded_inputs["reshaped_input_sizes"].cpu(),
        )[0]

        # Get IOU scores for the single image, shape: (num_boxes, num_masks_per_box)
        iou_scores_for_image = outputs.iou_scores[0]

        # Find the index of the mask with the highest IOU score for each bounding box.
        # best_mask_idxs will have shape (num_boxes,).
        # If multimask_output=False, num_masks_per_box is 1, so argmax will correctly return 0 for each box.
        best_mask_idxs = torch.argmax(iou_scores_for_image, dim=1)

        self._best_iou_scores = (
            iou_scores_for_image[torch.arange(iou_scores_for_image.shape[0]), best_mask_idxs].cpu().numpy()
        )

        # Select the best mask for each box.
        num_boxes = masks.shape[0]

        # Ensure best_mask_idxs is on CPU for indexing, as all_upsampled_masks_for_image is on CPU.
        best_mask_idxs_cpu = best_mask_idxs.cpu()

        # Index to get (num_boxes, H_orig, W_orig)
        selected_best_masks = masks[
            torch.arange(num_boxes, device=masks.device),
            best_mask_idxs_cpu,
        ]
        masks_final_np = np.array(selected_best_masks, dtype=np.uint8)

        return masks_final_np

    def get_scores(self) -> np.ndarray:
        """Return the IOU scores for the last prediction.

        Returns:
            A numpy array of shape (num_boxes,) containing the IOU scores for each box.
        """
        if self._best_iou_scores is None:
            raise RuntimeError("predict must be called before predict().")

        return self._best_iou_scores
