import os

import clip
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
        This method takes an image path as input and returns the predictions.
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
