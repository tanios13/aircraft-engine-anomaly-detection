from re import L

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sympy import Q
from torch.functional import F
from transformers import CLIPSegForImageSegmentation, CLIPSegModel, CLIPSegProcessor

from aircraft_anomaly_detection.schemas.data import Annotation


class AnnotationPostProcessor:
    def __call__(self, annotation: Annotation) -> None:
        pass


class BBoxSizeFilter(AnnotationPostProcessor):
    """
    A filter that removes bounding boxes whose areas are either too small or too large
    relative to the total object area.
    """

    def __init__(self, object_area: float, min_bb_size: float = 0.0, max_bb_size: float = 0.1):
        """
        Parameters:
        - object_area: Total area of the object being annotated.
        - min_bb_size: Minimum relative size of a bounding box (compared to object_area).
        - max_bb_size: Maximum relative size of a bounding box (compared to object_area).
        """
        self.object_area = object_area
        self.min_bb_size = min_bb_size
        self.max_bb_size = max_bb_size

    def __call__(self, annotation: Annotation) -> None:
        """
        Filters out bounding boxes from the annotation that fall outside the allowed size range.
        """
        valid_bbox_idx = []

        for i in range(len(annotation.bboxes)):
            x1, y1, x2, y2 = annotation.bboxes[i]
            bbox_area = (x2 - x1) * (y2 - y1)

            if self.object_area * self.min_bb_size < bbox_area <= self.object_area * self.max_bb_size:
                valid_bbox_idx.append(i)

        if annotation.mask is not None:
            valid_bbox_mask = np.zeros_like(annotation.mask)
            for i in valid_bbox_idx:
                x1, y1, x2, y2 = annotation.bboxes[i]
                valid_bbox_mask[y1:y2, x1:x2] = 1.0
            annotation.mask = annotation.mask * valid_bbox_mask

        print(
            f"BBoxSizeFilter: removed {len(annotation.bboxes) - len(valid_bbox_idx)} boxes from {len(annotation.bboxes)}"
        )

        annotation.bboxes = [annotation.bboxes[i] for i in valid_bbox_idx]
        annotation.scores = [annotation.scores[i] for i in valid_bbox_idx]

        annotation.damaged = len(annotation.bboxes) > 0


class TopKFilter(AnnotationPostProcessor):
    """
    A filter that removes bounding boxes whose areas are either too small or too large
    relative to the total object area.
    """

    def __init__(self, top_k: int = 3):
        """
        Parameters:
        - object_area: Total area of the object being annotated.
        - min_bb_size: Minimum relative size of a bounding box (compared to object_area).
        - max_bb_size: Maximum relative size of a bounding box (compared to object_area).
        """
        self.top_k = top_k

    def __call__(self, annotation: Annotation) -> None:
        """
        Filters out bounding boxes from the annotation that fall outside the allowed size range.
        """
        valid_bbox_idx = []

        ids = range(len(annotation.bboxes))
        key = lambda id: annotation.scores[id]
        ids = sorted(ids, key=key, reverse=True)
        if self.top_k < len(annotation.bboxes):
            valid_bbox_idx = ids[: self.top_k]
        else:
            valid_bbox_idx = range(len(annotation.bboxes))

        if annotation.mask is not None:
            valid_bbox_mask = np.zeros_like(annotation.mask)
            for i in valid_bbox_idx:
                x1, y1, x2, y2 = annotation.bboxes[i]
                valid_bbox_mask[y1:y2, x1:x2] = 1.0
            annotation.mask = annotation.mask * valid_bbox_mask

        print(f"TopKFilter: removed {len(annotation.bboxes) - len(valid_bbox_idx)} boxes from {len(annotation.bboxes)}")

        annotation.bboxes = [annotation.bboxes[i] for i in valid_bbox_idx]
        annotation.scores = [annotation.scores[i] for i in valid_bbox_idx]

        annotation.damaged = len(annotation.bboxes) > 0


class BBoxOnObjectFilter(AnnotationPostProcessor):
    """
    A filter that removes bounding boxes that primarily lie on the background
    (based on a provided background mask).
    """

    def __init__(self, background_mask):
        """
        Parameters:
        - background_mask: A binary mask (e.g., 0 for object, 1 for background) of the same shape
                           as the annotated image.
        """
        self.background_mask = background_mask

    def __call__(self, annotation: Annotation) -> None:
        """
        Removes bounding boxes that overlap more than 50% with the background.
        """
        valid_bbox_idx = []

        for i in range(len(annotation.bboxes)):
            x1, y1, x2, y2 = annotation.bboxes[i]

            # Skip invalid bounding boxes with zero area
            if x1 == x2 or y1 == y2:
                continue

            area = (x2 - x1) * (y2 - y1)
            background_area = self.background_mask[y1:y2, x1:x2].sum()

            if background_area / area < 0.5:
                valid_bbox_idx.append(i)

        if annotation.mask is not None:
            valid_bbox_mask = np.zeros_like(annotation.mask)
            for i in valid_bbox_idx:
                x1, y1, x2, y2 = annotation.bboxes[i]
                valid_bbox_mask[y1:y2, x1:x2] = 1.0
            annotation.mask = annotation.mask * valid_bbox_mask
        print(
            f"BBoxOnObjectFilter: removed {len(annotation.bboxes) - len(valid_bbox_idx)} boxes from {len(annotation.bboxes)}"
        )

        annotation.bboxes = [annotation.bboxes[i] for i in valid_bbox_idx]
        annotation.scores = [annotation.scores[i] for i in valid_bbox_idx]

        assert len(annotation.bboxes) == len(valid_bbox_idx) and len(annotation.scores) == len(valid_bbox_idx)

        # TODO: Add similar logic to clean up pixel-level mask annotations
        annotation.damaged = len(annotation.bboxes) > 0


class CLIPAnomalySegmentor(AnnotationPostProcessor):
    def __init__(
        self,
        model_id: str = "CIDAS/clipseg-rd64-refined",
        padding_ratio: float = 0.5,
        background_text: str = "metallic surface",
        threshold: float = 0.5,
        *,
        device: str | None = None,
    ):
        # here I need to initialize the clip model
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPSegProcessor.from_pretrained(model_id)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_id).to(self.device).eval()

        self.padding_ratio = padding_ratio
        self.background_text = background_text
        self.threshold = threshold

    def __call__(self, annotation: Annotation):
        if not isinstance(annotation.image, Image.Image):
            raise ValueError(f"annotation.image must be an instance of PIL Image.Image, but got {annotation.image}")

        # Create zero mask
        image = annotation.image
        width, height = image.size
        annotation.mask = np.zeros((height, width), dtype=np.float16)

        # I go through bboxes cut them with padding, and get segmentation mask.
        for i in range(len(annotation.bboxes)):
            x0, y0, x1, y1 = annotation.bboxes[i]

            box_width = x1 - x0
            box_height = y1 - y0

            x0_padded = int(np.clip(x0 - box_width * self.padding_ratio, 0, width))
            x1_padded = int(np.clip(x1 + box_width * self.padding_ratio, 0, width))
            y0_padded = int(np.clip(y0 - box_height * self.padding_ratio, 0, height))
            y1_padded = int(np.clip(y1 + box_height * self.padding_ratio, 0, height))

            cropped_image = annotation.image.crop(
                (float(x0_padded), float(y0_padded), float(x1_padded), float(y1_padded))
            )
            cropped_image = cropped_image.convert("RGB")

            # plt.imshow(cropped_image)
            # plt.title(annotation.bboxes_labels[i])
            # plt.show()

            # I need to call CLIPSeg here somehow
            # text_prompts = [annotation.bboxes_labels[i], self.background_text]
            # text_prompts = ["scratch, chip", self.background_text]
            text_prompts = ["a scratch", "undamaged metallic surface or background"]
            inputs = self.processor(
                text=text_prompts, images=[cropped_image] * len(text_prompts), return_tensors="pt", padding=True
            ).to(self.device)
            logits = self.model(**inputs).logits
            cropped_mask = F.softmax(logits, dim=0).detach().cpu().numpy()[0]
            # cropped_mask = torch.sigmoid(logits).detach().cpu().numpy()[0]
            # plt.imshow(cropped_mask)
            # plt.colorbar()
            # plt.show()

            # Place the cropped mask back into the full-sized mask at the correct location
            cropped_mask_width, cropped_mask_height = cropped_mask.shape
            # Calculate the paste region dimensions
            padded_box_width, padded_box_height = int(x1_padded - x0_padded), int(y1_padded - y0_padded)
            # Resize crop_mask if dimensions don't match (due to boundary clipping)
            if padded_box_width != cropped_mask_width or padded_box_height != cropped_mask_height:
                print(padded_box_width, padded_box_height, cropped_mask_width, cropped_mask_height)
                cropped_mask = cv2.resize(
                    cropped_mask, (padded_box_width, padded_box_height), interpolation=cv2.INTER_NEAREST
                )
            # Place the crop mask into the full mask
            uncropped_mask = np.zeros_like(annotation.mask)
            uncropped_mask[y0_padded:y1_padded, x0_padded:x1_padded] = cropped_mask
            # plt.imshow(uncropped_mask)
            # plt.colorbar()
            # plt.show()
            annotation.mask += uncropped_mask
            annotation.scores[i] = cropped_mask.max()

        plt.imshow(annotation.mask)
        plt.colorbar()
        plt.show()


class CLIPAnomalyFilter(AnnotationPostProcessor):
    def __init__(
        self,
        model_id: str = "CIDAS/clipseg-rd64-refined",
        padding_ratio: float = 0.5,
        background_texts: list[str] = [
            "undamaged metallic surface",
            "background",
            "screw hole",
            "blade edge",
            "text inscription",
            "letters",
            "undamaged metal component",
            "perfect metal component",
        ],
        threshold: float = 0.2,
        *,
        device: str | None = None,
    ):
        # here I need to initialize the clip model
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPSegProcessor.from_pretrained(model_id)
        self.model = CLIPSegModel.from_pretrained(model_id).to(self.device).eval()

        self.padding_ratio = padding_ratio
        self.background_texts = background_texts
        self.threshold = threshold

    def __call__(self, annotation: Annotation):
        if not isinstance(annotation.image, Image.Image):
            raise ValueError(f"annotation.image must be an instance of PIL Image.Image, but got {annotation.image}")

        # Create zero mask
        image = annotation.image
        width, height = image.size

        # I go through bboxes cut them with padding, and get segmentation mask.
        for i in range(len(annotation.bboxes)):
            x0, y0, x1, y1 = annotation.bboxes[i]

            box_width = x1 - x0
            box_height = y1 - y0

            x0_padded = int(np.clip(x0 - box_width * self.padding_ratio, 0, width))
            x1_padded = int(np.clip(x1 + box_width * self.padding_ratio, 0, width))
            y0_padded = int(np.clip(y0 - box_height * self.padding_ratio, 0, height))
            y1_padded = int(np.clip(y1 + box_height * self.padding_ratio, 0, height))

            cropped_image = annotation.image.crop(
                (float(x0_padded), float(y0_padded), float(x1_padded), float(y1_padded))
            )
            cropped_image = cropped_image.convert("RGB")

            text_prompts = [annotation.bboxes_labels[i]] + self.background_texts
            inputs = self.processor(text=text_prompts, images=cropped_image, return_tensors="pt", padding=True).to(
                self.device
            )
            prob = F.softmax(self.model(**inputs).logits_per_image, dim=1).detach().cpu().numpy()[0][0]
            annotation.scores[i] = prob

        valid_bbox_idx = [i for i in range(len(annotation.bboxes)) if annotation.scores[i] > self.threshold]

        if annotation.mask is not None:
            valid_bbox_mask = np.zeros_like(annotation.mask)
            for i in valid_bbox_idx:
                x1, y1, x2, y2 = annotation.bboxes[i]
                valid_bbox_mask[y1:y2, x1:x2] = 1.0
            annotation.mask = annotation.mask * valid_bbox_mask
        print(
            f"CLIPAnomalyFilter: removed {len(annotation.bboxes) - len(valid_bbox_idx)} boxes from {len(annotation.bboxes)}"
        )

        annotation.bboxes = [annotation.bboxes[i] for i in valid_bbox_idx]
        annotation.scores = [annotation.scores[i] for i in valid_bbox_idx]

        assert len(annotation.bboxes) == len(valid_bbox_idx) and len(annotation.scores) == len(valid_bbox_idx)

        # TODO: Add similar logic to clean up pixel-level mask annotations
        annotation.damaged = len(annotation.bboxes) > 0
