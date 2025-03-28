import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import OwlViTProcessor, OwlViTForObjectDetection


class OwlViT:
    def __init__(self, model_path, device=None):
        """
        Initialize OwlViT model from local path
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = OwlViTProcessor.from_pretrained(model_path)
        self.model = OwlViTForObjectDetection.from_pretrained(model_path).to(self.device)


    def predict(self, image_path, text_prompts, undamaged_idxes=[], threshold=0.01, top_k=2):
        """
        Run OwlViT on an image and return filtered boxes and labels.
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=text_prompts, images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_grounded_object_detection(outputs=outputs, threshold=threshold, target_sizes=target_sizes)

        # Retrieve predictions for the first image
        i = 0  
        boxes, scores, labels = self._filter_boxes(results[i]["boxes"], results[i]["scores"], results[i]["labels"], undamaged_idxes, top_k)

        return image, boxes, scores, labels


    def plot(self, image, text_prompts, boxes, labels_idx, scores):
        """
        Plot bounding boxes on the image with labels and scores.
        """
        # Proceed if any match found
        if boxes:

            _, ax = plt.subplots(1, figsize=(6, 6))
            ax.imshow(image)

            for idx in range(len(boxes)):
                box = boxes[idx].tolist()
                x1, y1, x2, y2 = [max(0, v) for v in box]
                width, height = x2 - x1, y2 - y1

                rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)

                label = text_prompts[labels_idx[idx].item()]
                conf = scores[idx].item()
                label_text = f"{label} ({conf:.2f})"
                ax.text(x1, y1 - 10, label_text, color='red', fontsize=12, backgroundcolor='white')

            plt.axis('off')
            plt.tight_layout()
            plt.show()
        else:
            print("No defect found")




#--------------------------------------------------------HELPERS--------------------------------------------------------#


    def _filter_boxes(self, boxes, scores, labels, undamaged_idxes, top_k):
        """
        Filter boxes by labels and scores.
        """
        # Filter only scratch label
        predictions = [
            (box, score, label)
            for box, score, label in zip(boxes, scores, labels)
            if label.item() not in undamaged_idxes
        ]
        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

        # Keep top_k
        if top_k:
            predictions = predictions[:top_k]

        # Unzip or return empty
        if predictions:
            boxes, scores, labels = zip(*predictions)
        else:
            boxes, scores, labels = [], [], []

        return boxes, scores, labels