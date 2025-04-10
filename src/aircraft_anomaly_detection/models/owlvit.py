from os import PathLike

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor


class OwlViT:
    def __init__(
        self, pretrained_model_name_or_pat: str | PathLike = "google/owlvit-base-patch32", device: str | None = None
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

    def _load_image(self, image_input: str | Image.Image | np.ndarray) -> Image.Image:
        """
        Load an image from a file path, a numpy array, or a PIL image, and return a PIL Image in RGB.

        Args:
            image_input (Union[str, Image.Image, np.ndarray]): Image file path, numpy array, or PIL Image.

        Returns:
            Image.Image: Loaded image in RGB mode.
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(np.uint8(image_input)).convert("RGB")
        else:
            raise ValueError("image_input must be a file path (str), a PIL.Image.Image, or a numpy.ndarray.")
        return image

    def predict(
        self,
        image_input: str | Image.Image | np.ndarray,
        text_prompts: list[list[str]],
        undamaged_idxes: list[int] = [],
        threshold: float = 0.01,
        top_k: int = 2,
    ) -> tuple[Image.Image, np.ndarray, list[float], list[str]]:
        """
        Run OwlViT on an image and return filtered boxes, scores, and labels.

        Args:
            image_input (Union[str, Image.Image, np.ndarray]): Image file path, a PIL image, or a numpy array.
            text_prompts (List[List[str]]): A list containing a single list of class descriptions, e.g., [["defect", "no defect"]].
            undamaged_idxes (List[int], optional): List of label indices to filter out. Defaults to [].
            threshold (float, optional): Confidence threshold for post-processing. Defaults to 0.01.
            top_k (int, optional): Number of top predictions to keep. Defaults to 2.

        Returns:
            Tuple containing:
                - image (Image.Image): The loaded PIL image (RGB).
                - boxes (np.ndarray): Array of filtered bounding boxes (each as [x0, y0, x1, y1]).
                - scores (List[float]): Confidence scores for each box.
                - labels (List[str]): Detected labels for each box.
        """
        image = self._load_image(image_input)
        inputs = self.processor(text=text_prompts, images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, threshold=threshold, target_sizes=target_sizes
        )

        # Retrieve predictions for the first image.
        result = results[0]
        boxes, scores, labels = self._filter_boxes(
            result["boxes"], result["scores"], result["labels"], undamaged_idxes, top_k
        )
        return image, boxes, scores, labels

    def plot(
        self,
        image_input: str | Image.Image | np.ndarray,
        boxes: np.ndarray,
        text_labels: list[str],
        title: str = "OwlViT Predictions",
    ) -> None:
        """
        Plot the detected bounding boxes and labels on the image.

        Args:
            image_input (Union[str, Image.Image, np.ndarray]): Image file path, PIL image, or numpy array.
            boxes (np.ndarray): Array of bounding boxes (each as [x0, y0, x1, y1]).
            text_labels (List[str]): List of detected labels corresponding to each box.
            title (str): Title for the plot.
        """
        image = self._load_image(image_input)
        image_array = np.array(image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        for box, label in zip(boxes, text_labels):
            x0, y0, x1, y1 = box
            cv2.rectangle(image_array, (x0, y0), (x1, y1), (255, 0, 0), 2)
            cv2.putText(image_array, str(label), (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
        plt.show()

    def _filter_boxes(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        undamaged_idxes: list[int],
        top_k: int,
    ) -> tuple[np.ndarray, list[float], list[str]]:
        predictions = [
            (box, score, label)
            for box, score, label in zip(boxes, scores, labels)
            if label.item() not in undamaged_idxes
        ]
        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
        if top_k:
            predictions = predictions[:top_k]
        if predictions:
            boxes_pred, scores_pred, labels_pred = zip(*predictions)
            boxes_np = np.stack([b.detach().cpu().numpy().astype(int) for b in boxes_pred], axis=0)
            scores_list = [s.item() for s in scores_pred]
            labels_list = [str(l.item()) for l in labels_pred]  # Or use a mapping if available.
        else:
            boxes_np, scores_list, labels_list = np.array([]), [], []
        return boxes_np, scores_list, labels_list
