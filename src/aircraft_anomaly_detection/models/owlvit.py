from os import PathLike
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from aircraft_anomaly_detection.interface.model import ModelInterface
from aircraft_anomaly_detection.schemas.data import Annotation


class OwlViT(ModelInterface):
    def __init__(
        self,
        pretrained_model_name_or_pat: str | PathLike = "google/owlvit-base-patch32",
        device: str | None = None,
        text_prompts: list[list[str]] = [["a clean, undamaged metal surface", "a close-up image of a metal scratch"]],
        undamaged_idxes: list[int] = [0],
        threshold: float = 0.01,
        top_k: int = 2,
    ) -> None:
        """
        Initialize the OwlViT model from Hugging Face.

        Args:
            model_id (str): Pretrained model identifier from Hugging Face (default "google/owlvit-base-patch32").
            device (str, optional): Device to run the model. Defaults to "cuda" if available, else "cpu".
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = OwlViTProcessor.from_pretrained(pretrained_model_name_or_pat)
        self.model = OwlViTForObjectDetection.from_pretrained(pretrained_model_name_or_pat).to(self.device)

        self.text_prompts: list[list[str]] = text_prompts
        self.undamaged_idxes: list[int] = undamaged_idxes
        self.threshold: float = threshold
        self.top_k: int = top_k

    def predict(
        self,
        input_image: str | Image.Image | np.ndarray,
        **kwargs: dict[str, Any],
    ) -> Annotation:
        """
        Run prediction on an image and return results as an Annotation.

        Args:
            input_image (Union[str, Image.Image, np.ndarray]): Image file path, PIL Image, or numpy array.
            text_prompts (List[List[str]]): Class descriptions, e.g., [["defect", "no defect"]].
            undamaged_idxes (List[int], optional): Label indices to filter out (e.g., no-defect classes).
            threshold (float, optional): Confidence threshold for detection.
            top_k (int, optional): Number of top predictions to keep.

        Returns:
            Annotation: label=True with bboxes, scores, bboxes_labels if defects found;
                otherwise label=False and empty lists.
        """
        image = self.load_image(input_image)
        inputs = self.processor(text=self.text_prompts, images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=self.threshold,
            target_sizes=target_sizes,
        )
        result = results[0]

        boxes, scores, labels_str = self._filter_boxes(
            result["boxes"], result["scores"], result["labels"], self.undamaged_idxes, self.top_k, self.text_prompts
        )

        if boxes.size and scores:
            ann = Annotation(
                image=image,
                damaged=True,
                bboxes=boxes.astype(np.int32).tolist(),  # type: ignore
                scores=scores,
                bboxes_labels=labels_str,
                mask=self.box_to_mask(image, boxes),
            )
        else:
            ann = Annotation(
                image=image, damaged=False, bboxes=[], scores=[], bboxes_labels=[], mask=self.box_to_mask(image, boxes)
            )
        return ann

    def _filter_boxes(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        undamaged_idxes: list[int],
        top_k: int,
        text_prompts: list[list[str]],
    ) -> tuple[np.ndarray, list[float], list[str]]:
        # filter out “undamaged” classes
        preds = [
            (box, score, label)
            for box, score, label in zip(boxes, scores, labels)
            if label.item() not in undamaged_idxes
        ]
        # sort by score desc
        preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_k]

        if preds:
            boxes_pred, scores_pred, labels_pred = zip(*preds)
            boxes_np = np.stack([b.detach().cpu().numpy().astype(int) for b in boxes_pred], axis=0)
            scores_list = [float(s) for s in scores_pred]
            labels_str = [text_prompts[0][int(l)] for l in labels_pred]
        else:
            boxes_np, scores_list, labels_str = np.array([]), [], []

        return boxes_np, scores_list, labels_str

    def box_to_mask(self, image: Image.Image, boxes: np.ndarray) -> np.ndarray:
        """
        Convert bounding boxes to a binary mask.

        Args:
            image (Image.Image): The input image.
            boxes (np.ndarray): Array of bounding boxes (x0, y0, x1, y1).

        Returns:
            np.ndarray: Binary mask with 1s inside boxes, 0 elsewhere.
        """
        width, height = image.size
        mask = np.zeros((height, width), dtype=np.uint8)

        if boxes.size == 0:
            return mask

        for box in boxes:
            x0, y0, x1, y1 = map(int, box)  # ensure integers
            x0 = np.clip(x0, 0, width)
            x1 = np.clip(x1, 0, width)
            y0 = np.clip(y0, 0, height)
            y1 = np.clip(y1, 0, height)
            mask[y0:y1, x0:x1] = 1

        return mask
