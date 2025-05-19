

from multiprocessing import process
from typing import Callable, List

import cv2
import numpy as np
from PIL import Image
from torch.functional import F
from transformers import AutoProcessor, CLIPSegForImageSegmentation


class CLIPBackgroundRemover(Callable):
    def __init__(self,
                 background_text: str = "a background",
                 foreground_text: str = "a metal part",
                 treshold: float = 0.5,
                 background_color: List = [0, 255, 0]):
        """
        Initializes the CLIPBackgroundRemover.

        Args:
            background_text (str): Text description of the background.
            foreground_text (str): Text description of the foreground (object of interest).
            treshold (float): Threshold to binarize the segmentation mask.
            background_color (List): Color to use for background replacement (currently unused).
        """
        self.treshold = treshold
        self.background_text = background_text
        self.foreground_text = foreground_text
        self.background_color = background_color

        self.processor = AutoProcessor.from_pretrained(
            "CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined")

    def __call__(self, image):
        """
        Generates a segmentation mask for the input image.

        Args:
            image (PIL.Image): Input image to segment.

        Returns:
            np.ndarray: A binary mask where 1 represents foreground and 0 background.
        """
        # Background segmentation
        texts = [self.background_text, self.foreground_text]
        image = image.convert("RGB")
        inputs = self.processor(
            text=texts, images=[image] * len(texts), padding=True, return_tensors="pt")
        logits = self.model(**inputs).logits
        mask = F.softmax(logits, dim=0).detach().cpu().numpy()[0]
        mask = (mask > 0.5).astype(int)
        resized_mask = cv2.resize(
            mask, image.size, interpolation=cv2.INTER_NEAREST)

        image_np = np.array(image)
        output_np = np.where(
            resized_mask[..., None] == 1, self.background_color, image_np)
        output_image = Image.fromarray(
            output_np.astype(np.uint8), mode="RGB")
        return output_image, resized_mask
