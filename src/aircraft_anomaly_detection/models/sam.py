import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry


class SAM:
    def __init__(self, checkpoint_path: str, model_type: str = "vit_h", device: str | None = None) -> None:
        """
        Initialize SAM with a checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
            model_type (str): Model type to use. Default is "vit_h".
            device (str, optional): Device to run the model. Defaults to "cuda" if available, else "cpu".
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path).to(self.device)
        self.model = SamPredictor(sam_model)

    def _load_image(self, image_input: str | np.ndarray | Image.Image) -> np.ndarray:
        """
        Load an image from a file path, a numpy array, or a PIL image.

        Args:
            image_input (str | np.ndarray | Image.Image): Either a file path, a numpy array, or a PIL.Image.Image.

        Returns:
            np.ndarray: The loaded image in RGB format.
        """
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Unable to load image from path: {image_input}")
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            # Assume the numpy array is in RGB format.
            image = image_input.copy()
        else:
            raise ValueError("image_input must be a file path (str), a numpy.ndarray, or a PIL.Image.Image.")
        return image

    def predict(
        self,
        image_input: str | np.ndarray | Image.Image,
        boxes: list[list[int]],
        multimask_output: bool = False,
        filter: bool = True,
    ) -> list[np.ndarray]:
        """
        Predict masks for given bounding boxes on an image.

        Args:
            image_input (str | np.ndarray | Image.Image): An image file path, a numpy array, or a PIL.Image.Image.
            boxes (List[List[int]]): List of bounding boxes (each as [x0, y0, x1, y1]).
            multimask_output (bool): Whether to output multiple mask predictions per box. Defaults to False.
            filter (bool): If True, filters boxes based on a minimum area threshold. Defaults to True.

        Returns:
            List[np.ndarray]: A list of predicted masks (binary masks as numpy arrays).
        """
        image = self._load_image(image_input)
        self.model.set_image(image)

        # Optionally filter boxes by area (area threshold 0.1)
        if filter:
            H, W, _ = image.shape
            area_threshold = 0.1
            boxes = [box for box in boxes if ((box[2] - box[0]) * (box[3] - box[1])) / (W * H) >= area_threshold]

        masks = []
        for box in boxes:
            masks_pred, _, _ = self.model.predict(box=box, multimask_output=multimask_output)
            masks.append(masks_pred[0])
        return masks

    def plot(
        self,
        image_input: str | np.ndarray | Image.Image,
        boxes: list[list[int]],
        masks: list[np.ndarray],
        title: str = "SAM Segmentation",
        filter: bool = True,
    ) -> None:
        """
        Plot the predicted masks and bounding boxes over the image.

        Args:
            image_input (str | np.ndarray | Image.Image): An image file path, a numpy array, or a PIL.Image.Image.
            boxes (List[List[int]]): List of bounding boxes (each as [x0, y0, x1, y1]).
            masks (List[np.ndarray]): List of predicted masks corresponding to each box.
            title (str): Title for the plot.
            filter (bool): If True, filters boxes based on an area threshold. Defaults to True.
        """
        image = self._load_image(image_input)
        image_copy = image.copy()

        # Optionally filter boxes
        if filter:
            H, W, _ = image.shape
            area_threshold = 0.1
            boxes = [box for box in boxes if ((box[2] - box[0]) * (box[3] - box[1])) / (W * H) >= area_threshold]

        # Draw boxes
        for box in boxes:
            x0, y0, x1, y1 = box
            cv2.rectangle(image_copy, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # Overlay masks with transparency
        for mask in masks:
            overlay = np.zeros_like(image_copy, dtype=np.uint8)
            overlay[mask] = np.array([255, 0, 0])
            image_copy = cv2.addWeighted(image_copy, 0.5, overlay, 0.5, 0)

        plt.figure(figsize=(10, 10))
        plt.imshow(image_copy)
        plt.axis("off")
        plt.title(title)
        plt.show()
