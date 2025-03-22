import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from groundingdino.util.inference import load_model, load_image, predict as dino_predict


class DINO:
    def __init__(self, config_path, weights_path, device=None):
        """
        Takes an image and text prompt as input and predicts a list of bounding boxes
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(config_path, weights_path, device=device)


    def predict(self, image_path, prompt, box_threshold=0.1, text_threshold=0.5):
        """
        Predict bounding boxes
        """
        image_source, image_tensor = load_image(image_path)

        boxes, _, phrases = dino_predict(
            model=self.model,
            image=image_tensor,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )
        H, W, _ = image_source.shape
        boxes = boxes * torch.Tensor([W, H, W, H])
        boxes = boxes.cpu().numpy().astype(int)
        return boxes, phrases


    def plot(self, image_path, boxes, phrases, title="GroundingDINO Predictions"):
        """
        Plot the predictions
        """
        image, _ = load_image(image_path)
        
        image_copy = image.copy()
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = box
            cv2.rectangle(image_copy, (x0, y0), (x1, y1), (255, 0, 0), 2)
            cv2.putText(image_copy, phrases[i], (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image_copy.astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(title)
        plt.show()