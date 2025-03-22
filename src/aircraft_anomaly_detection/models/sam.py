import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from segment_anything import SamPredictor, sam_model_registry
from groundingdino.util.inference import load_image


class SAM:
    def __init__(self, checkpoint_path, model_type="vit_h", device=None):
        """
        Takes an image and bouding boxes as input and predicts masks
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)
        self.model = SamPredictor(sam_model)


    def predict(self, image_path, boxes, multimask_output=False):
        """
        Predict masks
        """
        image, _ = load_image(image_path)
        self.model.set_image(image)

        # Filter boxes
        H, W, _ = image.shape
        area_threshold = 0.1
        filtered_boxes = [box for box in boxes if ((box[2]-box[0]) * (box[3]-box[1])) / (W * H) >= area_threshold]

        masks = []
        for box in filtered_boxes:
            masks_pred, _, _ = self.model.predict(box=box, multimask_output=multimask_output)
            masks.append(masks_pred[0])
        return masks


    def plot(self, image_path, boxes, masks, title="SAM Segmentation"):
        """
        Plot the masks
        """
        image, _ = load_image(image_path)
        image_copy = image.copy()

        # Filter boxes
        H, W, _ = image.shape
        area_threshold = 0.1
        filtered_boxes = [box for box in boxes if ((box[2]-box[0]) * (box[3]-box[1])) / (W * H) >= area_threshold]

        # Draw boxes
        for box in filtered_boxes:
            x0, y0, x1, y1 = box
            cv2.rectangle(image_copy, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # Overlay masks
        for mask in masks:
            image_copy[mask] = image_copy[mask] * 0.5 + np.array([255, 0, 0]) * 0.5

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image_copy.astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(title)
        plt.show()
