import numpy as np
import torch
from PIL import Image
from transformers.models.auto.modeling_auto import AutoModelForZeroShotObjectDetection
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.grounding_dino import GroundingDinoProcessor

from aircraft_anomaly_detection.interface.model import DetectorInterface


class DINO(DetectorInterface):
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny", device: str | None = None) -> None:
        """
        Initializes the GroundingDINO model using the Hugging Face Transformers library.

        Args:
            model_id: Model identifier from Hugging Face.
            device: Device to run the model on.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor: GroundingDinoProcessor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)  # type: ignore

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
        inputs = self.processor(images=image, text=[text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs, threshold=box_threshold, text_threshold=text_threshold, target_sizes=[(image.height, image.width)]
        )
        result = results[0]
        boxes = result["boxes"].cpu().numpy().astype(int)
        scores = [score.item() for score in result["scores"]]
        detected_labels = result["text_labels"]
        return boxes, scores, detected_labels
