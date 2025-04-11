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
        text_labels: list[str],
        boxes: np.ndarray,
        labels_idx: list[int],
        scores: list[float],
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
        _, ax = plt.subplots(1, figsize=(6, 6))
        ax.imshow(image)
        plt.axis('off')
        plt.title(title)

        if len(boxes):
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = [max(0, v) for v in box.tolist()]
                width, height = x2 - x1, y2 - y1

                rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)

                label = text_labels[labels_idx[idx]]
                conf = scores[idx]
                label_text = f"{label} ({conf:.2f})"
                ax.text(x1, y1 - 10, label_text, color='red', fontsize=12, backgroundcolor='white')
        else:
            print("No defect found")

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
            labels_list = [l.item() for l in labels_pred]  # Or use a mapping if available.
        else:
            boxes_np, scores_list, labels_list = np.array([]), [], []
        return boxes_np, scores_list, labels_list
