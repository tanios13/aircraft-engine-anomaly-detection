import numpy as np
import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from aircraft_anomaly_detection.interface.model import DetectorInterface


class OwlViT(DetectorInterface):

    def __init__(
        self,
        model_id: str = "google/owlvit-base-patch32",
        device: str | None = None,
    ) -> None:
        """
        Initializes the GroundingDINO model using the Hugging Face Transformers library.

        Args:
            model_id: Model identifier from Hugging Face.
            device: Device to run the model on.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = OwlViTProcessor.from_pretrained(model_id)
        self.model = OwlViTForObjectDetection.from_pretrained(model_id).to(self.device)

    def predict(
        self,
        image: Image.Image,
        text_prompts: list[str],
        *,
        box_threshold: float | None = 0.3,
        text_threshold: float | None = 0.4,
    ) -> tuple[np.ndarray, list[float], list[str]]:
        """
        Predict bounding boxes and labels for an image given text prompts.

        Args:
            image: PIL Image.
            text_prompts: List of text prompts.
            box_threshold: Threshold for box confidence.
            text_threshold: Threshold for text confidence.

        Returns:
            Tuple of bounding boxes, scores, and labels.
        """
        text = [prompt.strip().lower() for prompt in text_prompts]
        #append prompts of parts that are often misclassified as defects such as "normal skrew hole", "normal bolt" , "engraving"
        if text_prompts != 'metallic surface':
            text += [
                "normal screw hole",
                "normal bolt",
                "engraving",
                "normal rivet",
                "normal panel",
            ]
        else :
            text = [
                "metal"]
            
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        #
        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=box_threshold,
            target_sizes=target_sizes,
        )
        # check how many results are there

        result = results[0]

        boxes = result["boxes"].cpu().numpy().astype(int)
        scores = [score.item() for score in result["scores"]]
        detected_labels = [text[i] for i in result["labels"].cpu().numpy()]

        #exclude entries that were of the "normal" class
        normal_classes = [
            "normal screw hole",
            "normal bolt",
            "engraving",
            "normal rivet",
            "normal panel",
            "metal object with holes",
            "smooth metal surface",
        ]

        filtered_indices = [
            i for i, label in enumerate(detected_labels) if label not in normal_classes
        ]
        boxes = boxes[filtered_indices]
        scores = [scores[i] for i in filtered_indices]
        detected_labels = [detected_labels[i] for i in filtered_indices]

        sorted_indices = np.argsort(scores)[::-1]
        boxes = boxes[sorted_indices][:5]
        scores = [scores[i] for i in sorted_indices][:5]
        detected_labels = [detected_labels[i] for i in sorted_indices][:5]

        # check if the detected labels are empty

        return boxes, scores, detected_labels
