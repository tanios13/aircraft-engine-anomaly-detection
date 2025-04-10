import os
from typing import List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


class DINO:
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny", device: str | None = None) -> None:
        """
        Initializes the GroundingDINO model using the Hugging Face Transformers library.

        Args:
            model_id (str): The model identifier from Hugging Face (default "IDEA-Research/grounding-dino-tiny").
            device (str, optional): Device to use; if None, auto-detects "cuda" if available, else "cpu".
        """
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

    def predict(
        self,
        image_input: str | Image.Image,
        text_labels: list[list[str]],
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
    ) -> tuple[np.ndarray, list[float], list[str]]:
        """
        Predict bounding boxes and labels for an image given zero-shot text prompts.

        Args:
            image_input (Union[str, Image.Image]): Either the file path to an image or a PIL.Image.Image.
            text_labels (List[List[str]]): A list containing a single list of class descriptions, e.g., [["a cat", "a remote control"]].
            box_threshold (float): Bounding box threshold for post-processing (default 0.4).
            text_threshold (float): Text confidence threshold for post-processing (default 0.3).

        Returns:
            Tuple containing:
                - boxes (np.ndarray): Array of bounding boxes (each box as [x0, y0, x1, y1] in integer coordinates).
                - scores (List[float]): List of confidence scores for each box.
                - detected_labels (List[str]): List of the detected text labels.
        """
        # Load image
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("image_input must be a file path (str) or a PIL.Image.Image")

        # Prepare inputs using the processor
        inputs = self.processor(images=image, text=text_labels, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process outputs to get bounding boxes, scores, and text labels.
        results = self.processor.post_process_grounded_object_detection(
            outputs, threshold=box_threshold, text_threshold=text_threshold, target_sizes=[(image.height, image.width)]
        )
        result = results[0]
        boxes = result["boxes"].cpu().numpy().astype(int)
        scores = [score.item() for score in result["scores"]]
        detected_labels = result["text_labels"]
        return boxes, scores, detected_labels

    def plot(
        self,
        image_input: str | Image.Image,
        boxes: np.ndarray,
        text_labels: list[str],
        title: str = "GroundingDINO Predictions",
    ) -> None:
        """
        Plots the detected bounding boxes and labels on the image.

        Args:
            image_input (Union[str, Image.Image]): Either the file path to an image or a PIL.Image.Image.
            boxes (np.ndarray): Array of bounding boxes (each box as [x0, y0, x1, y1]).
            text_labels (List[str]): List of detected labels corresponding to each box.
            title (str): Title of the plot (default "GroundingDINO Predictions").
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("image_input must be a file path (str) or a PIL.Image.Image")

        # Convert image to numpy array in BGR for OpenCV
        image_array = np.array(image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Draw bounding boxes and labels
        for box, label in zip(boxes, text_labels):
            x0, y0, x1, y1 = box
            cv2.rectangle(image_array, (x0, y0), (x1, y1), (255, 0, 0), 2)
            cv2.putText(image_array, str(label), (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Plot using matplotlib
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
        plt.show()
