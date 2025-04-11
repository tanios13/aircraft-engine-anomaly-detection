import os

import clip
import numpy as np
import torch
from PIL import Image


class CLIP:
    def __init__(self, class_names=["defect", "no defect"], model_name="ViT-B/32", device=None):
        """
        Clip computes the similarity between a given image's embedding and text input's embeddings.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, self.device)
        self.class_names = class_names
        self.text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(self.device)

    def predict(self, image_path):
        """
        This method takes an image path or directory as input and returns the predictions.
        """
        if os.path.isdir(image_path):
            image_files = [
                os.path.join(image_path, f)
                for f in os.listdir(image_path)
                if f.lower().endswith(("png", "jpg", "jpeg", "bmp"))
            ]
        else:
            image_files = [image_path]

        predictions = {}

        for img_file in image_files:
            image = Image.open(img_file).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(self.text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(len(self.class_names))

            preds = []
            for value, index in zip(values, indices):
                preds.append({"label": self.class_names[index], "confidence": round(100 * value.item(), 2)})

            predictions[os.path.basename(img_file)] = preds

        return predictions

    def predict_single(self, image_input: str | Image.Image | np.ndarray) -> list[dict[str, str | float]]:
        """
        Predict labels for a single image using CLIP.
        The image_input can be:
        1) A string path to an image file
        2) A PIL.Image.Image object
        3) A numpy.ndarray representing the image

        Returns:
            A list of dictionaries, each containing:
            - "label": The predicted class name
            - "confidence": The confidence score (as a percentage)
        """

        # Determine how to handle the input
        if isinstance(image_input, str):
            # Treat as a file path
            if not os.path.isfile(image_input):
                raise ValueError(f"File does not exist: {image_input}")
            pil_img = Image.open(image_input).convert("RGB")

        elif isinstance(image_input, Image.Image):
            # Already a PIL image
            pil_img = image_input.convert("RGB")

        elif isinstance(image_input, np.ndarray):
            # Convert from NumPy array to PIL image
            # Ensure array is in HWC format and has 3 channels or 4 channels (RGBA)
            if image_input.ndim not in [2, 3]:
                raise ValueError("NumPy array must have 2 or 3 dimensions (HW or HWC).")
            if image_input.ndim == 2:
                # Grayscale
                pil_img = Image.fromarray(image_input, mode="L").convert("RGB")
            else:
                # Color (e.g., RGB, RGBA, etc.)
                pil_img = Image.fromarray(image_input).convert("RGB")

        else:
            raise ValueError("Unsupported input type. Use a file path (str), PIL.Image.Image, or np.ndarray.")

        # Preprocess the image
        image_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        # Compute image and text features
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(self.text_inputs)

        # Normalize the features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute similarity via dot product and softmax
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(len(self.class_names))

        # Build prediction output
        preds = []
        for value, index in zip(values, indices):
            preds.append({"label": self.class_names[index], "confidence": round(100 * value.item(), 2)})

        return preds
