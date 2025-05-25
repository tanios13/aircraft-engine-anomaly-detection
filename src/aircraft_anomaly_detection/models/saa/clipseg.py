from __future__ import annotations

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch.functional import F
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from aircraft_anomaly_detection.interface.model import SegmentorInterface


class ClipSegSegmentorHF(SegmentorInterface):
    """Hugging Face wrapper around CLIPSeg for image segmentation.

    Usage pattern::

        seg = ClipSegSegmentorHF()
        seg.set_image(pil_img)
        masks = seg.predict(text_prompts)
        

    The implementation keeps the API intentionally minimal—no CLIPSeg-specific
    objects leak out—so any other segmentor can be swapped in provided it
    respects :class:`SegmentorInterface`.
    """

    def __init__(
        self,
        model_id: str = "CIDAS/clipseg-rd64-refined",
        *,
        device: str | None = None,
    ) -> None:
        """Initialise the Hugging Face CLIPSeg backbone.

        Args:
            model_id: Hugging Face model card (e.g. ``"CIDAS/clipseg-rd64-refined"``).
            device:   Torch device string. Defaults to ``"cuda"`` if a GPU is
                available, otherwise ``"cpu"``.
        """
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPSegProcessor.from_pretrained(model_id)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_id).to(self.device).eval()

        self._encoded_inputs: dict[str, torch.Tensor] | None = None
        self._orig_shape: tuple[int, int] | None = None

        
    def set_image(self, image: Image.Image) -> None:
        """Pre compute an image embedding for fast prompting.

        Args:
            image: Input RGB image as ``PIL.Image.Image``.
        """
        image = image.convert("RGB")  # Ensure image is in RGB format
        self._encoded_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        self._orig_shape = (image.height, image.width)

    def predict(
        self,
        *,
        text_prompts: list[str],
        **kwargs: object,
    ) -> NDArray[np.uint8]:
        """Predict a binary mask for each text prompt.

        Args:
            text_prompts: List of text prompts.

        Returns:
            A numpy array of shape (N, H, W) with dtype uint8.
        """
        if self._encoded_inputs is None or self._orig_shape is None:
            raise ValueError("Image must be set before calling predict.")

        # Encode text prompts
        inputs = self.processor(text=text_prompts, return_tensors="pt", padding=True).to(self.device)

        # Run model

        logits = self.model(**self._encoded_inputs, **inputs).logits
        # Ensure logits shape is (N, 1, H, W)
        if logits.ndim == 2:
            logits = logits.unsqueeze(0).unsqueeze(0)
        elif logits.ndim == 3:
            logits = logits.unsqueeze(1)
        elif logits.ndim == 4:
            pass  # already fine
        else:
            raise ValueError(f"Unexpected logits shape: {logits.shape}")

        # Sigmoid
        probs = torch.sigmoid(logits)
        #visualize the probs in a matplotlib figure


        # # Create output folder if it doesn't exist
        # os.makedirs("clipseg_visuals", exist_ok=True)

        # # Visualize and save each probability map
        # for i in range(probs.shape[0]):
        #     prob_np = probs[i, 0].detach().cpu().numpy()

        #     plt.figure(figsize=(10, 4))

        #     # Probability map
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(prob_np, cmap="viridis")
        #     plt.title(f"{text_prompts}Probability Map - Prompt {i}")
        #     plt.axis("off")

        #     # Resize to original shape
        #     mask_resized = torch.nn.functional.interpolate(
        #         probs[i : i + 1],  # keep batch dim
        #         size=self._orig_shape,
        #         mode="bilinear",
        #         align_corners=False,
        #     )
        #     with torch.no_grad():
        #         binary_mask = (mask_resized.squeeze(0).squeeze(0).cpu().numpy() > 0.1).astype(np.uint8)

        #     # Binary mask
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(binary_mask, cmap="gray")
        #     plt.title(f"Binary Mask - Prompt {i}")
        #     plt.axis("off")

        #     # Save to file
        #     plt.tight_layout()
        #     plt.savefig(f"clipseg_visuals/prompt_{text_prompts}_{i}_mask.png")
        #     plt.close()


        # Resize to original shape
        mask_resized = torch.nn.functional.interpolate(
            probs,
            size=self._orig_shape,  # (H, W)
            mode="bilinear",
            align_corners=False,
        )

        # Convert to binary masks (N, H, W) and uint8
        with torch.no_grad():
            binary_masks = (mask_resized.squeeze(1).cpu().numpy() > 0.1).astype(np.uint8)
        
        #ensure its actually 
        return binary_masks  # Shape: (N, H, W)
