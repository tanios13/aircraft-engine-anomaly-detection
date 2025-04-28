import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

from os import PathLike
from transformers import OwlViTForObjectDetection, OwlViTProcessor
from ..interfaces import Annotation, ModelInterface
from PIL import Image

class OwlViT(ModelInterface):
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

    def predict(
        self,
        image_input: str | Image.Image | np.ndarray,
        text_prompts: list[list[str]],
        undamaged_idxes: list[int] = [],
        threshold: float = 0.01,
        top_k: int = 2,
    ) -> Annotation:
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
        """
        Run prediction on an image and return results as an Annotation.

        Args:
            image_input (Union[str, Image.Image, np.ndarray]): Image file path, PIL Image, or numpy array.
            text_prompts (List[List[str]]): Class descriptions, e.g., [["defect", "no defect"]].
            undamaged_idxes (List[int], optional): Label indices to filter out (e.g., no-defect classes).
            threshold (float, optional): Confidence threshold for detection.
            top_k (int, optional): Number of top predictions to keep.

        Returns:
            Annotation: label=True with bboxes, scores, bboxes_labels if defects found; otherwise label=False and empty lists.
        """
        image = self._load_image(image_input)
        inputs = self.processor(text=text_prompts, images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=threshold,
            target_sizes=target_sizes
        )
        result = results[0]

        boxes, scores, labels_str = self._filter_boxes(
            result["boxes"], result["scores"], result["labels"], undamaged_idxes, top_k, text_prompts
        )

        if boxes.size and scores:
            ann = Annotation(
                image=image,
                damaged=True,
                bboxes=boxes.tolist(),
                scores=scores,
                bboxes_labels=labels_str,
                mask=None
            )
        else:
            ann = Annotation(
                image=image,
                damaged=False,
                bboxes=[],
                scores=[],
                bboxes_labels=[],
                mask=None
            )
        return ann


    def plot(
        self,
        ann: Annotation,
        title: str = "OwlViT Predictions",
    ) -> None:
        """
        Draws bounding boxes and labels from an Annotation.
        """
        image = ann.image
        boxes = np.array(ann.bboxes)
        labels = ann.bboxes_labels or []
        scores = ann.scores or []

        _, ax = plt.subplots(1, figsize=(6, 6))
        ax.imshow(image)
        plt.axis('off')
        plt.title(title)

        if len(boxes):
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = [max(0, v) for v in box.tolist()]
                width, height = x2 - x1, y2 - y1

                rect = patches.Rectangle((x1, y1), width, height, linewidth=2,
                                         edgecolor='red', facecolor='none')
                ax.add_patch(rect)

                label_text = f"{labels[idx]} ({scores[idx]:.2f})"
                ax.text(x1, y1 - 10, label_text,
                        color='red', fontsize=12,
                        backgroundcolor='white')
        else:
            print("No defect found")


#------------------------------------------HELPER FUNCTIONS------------------------------------------#

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


    def _filter_boxes(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        undamaged_idxes: list[int],
        top_k: int,
        text_prompts: list[list[str]],
    ) -> tuple[np.ndarray, list[float], list[str]]:
        # filter out “undamaged” classes
        preds = [
            (box, score, label)
            for box, score, label in zip(boxes, scores, labels)
            if label.item() not in undamaged_idxes
        ]
        # sort by score desc
        preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_k]

        if preds:
            boxes_pred, scores_pred, labels_pred = zip(*preds)
            boxes_np = np.stack(
                [b.detach().cpu().numpy().astype(int) for b in boxes_pred],
                axis=0
            )
            scores_list = [float(s) for s in scores_pred]
            labels_str = [text_prompts[int(l)] for l in labels_pred]
        else:
            boxes_np, scores_list, labels_str = np.array([]), [], []

        return boxes_np, scores_list, labels_str
    
    def box_to_mask(self, image: Image.Image, boxes: np.ndarray) -> np.ndarray:
        """
        Convert bounding boxes to masks.

        Args:
            image (Image.Image): The input image.
            boxes (np.ndarray): Array of bounding boxes.

        Returns:
            np.ndarray: Array of masks.
        """
        masks = np.zeros((len(boxes), image.size[1], image.size[0]), dtype=np.uint8)
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = box
            masks[i, y0:y1, x0:x1] = 1
        return masks
    def mask_to_image(self, image: Image.Image, masks: np.ndarray) -> Image.Image:
        """
        Convert masks to an image.
        Args:
            image (Image.Image): The input image.
            masks (np.ndarray): Array of masks.
        Returns:
            Image.Image: The image with masks applied.
        """
        image_np = np.array(image)
        for mask in masks:
            image_np[mask == 1] = [255, 0, 0]
        return Image.fromarray(image_np.astype(np.uint8))