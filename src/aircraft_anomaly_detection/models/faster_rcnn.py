import os
import numpy as np
import torch
import torchvision
from PIL import Image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ..interfaces import Annotation, ModelInterface
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNN(ModelInterface):
    def __init__(
        self,
        model_path: str,
        num_classes: int = 2,
        device: str | None = None
    ) -> None:
        """
        Initialize the Faster R-CNN model for scratch detection.

        Args:
            model_path (str): Path to the saved model state_dict (.pth file).
            num_classes (int): Number of classes (background + scratch). Default is 2.
            device (str, optional): Compute device, e.g., 'cuda' or 'cpu'.
        """
        # determine device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # load base model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)
        # load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        # support full checkpoint dict or state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        image_input: str | Image.Image | np.ndarray,
        threshold: float = 0.5
    ) -> Annotation:
        """
        Run the Faster R-CNN model on an image and return an Annotation.

        Args:
            image_input (Union[str, Image.Image, np.ndarray]): Image path, PIL Image, or numpy array.
            threshold (float): Confidence threshold to filter detections.

        Returns:
            Annotation: Contains image, damaged flag, bboxes, scores, labels, and mask.
        """
        image = self._load_image(image_input)
        img_tensor = F.to_tensor(image).to(self.device)
        with torch.no_grad():
            outputs = self.model([img_tensor])[0]

        boxes, scores, labels = outputs['boxes'], outputs['scores'], outputs['labels']
        # filter for scratch class (label == 1) and by threshold
        boxes_np, scores_list = self._filter_boxes(boxes, scores, labels, threshold)
        labels_str = ['scratch'] * len(scores_list)

        if len(scores_list) > 0:
            ann = Annotation(
                image=image,
                damaged=True,
                bboxes=boxes_np.tolist(),
                scores=scores_list,
                bboxes_labels=labels_str,
                mask=self.box_to_mask(image, boxes_np)
            )
        else:
            ann = Annotation(
                image=image,
                damaged=False,
                bboxes=[],
                scores=[],
                bboxes_labels=[],
                mask=self.box_to_mask(image, boxes_np)
            )
        return ann

    def plot(
        self,
        ann: Annotation,
        title: str = "Faster R-CNN Scratch Detections"
    ) -> None:
        """
        Draw bounding boxes and labels from an Annotation on the image.
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
                rect = patches.Rectangle((x1, y1), width, height,
                                         linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                label_text = f"{labels[idx]} ({scores[idx]:.2f})"
                ax.text(x1, y1 - 10, label_text,
                        color='red', fontsize=12,
                        backgroundcolor='white')
        else:
            print("No scratches detected")

    def _load_image(
        self,
        image_input: str | Image.Image | np.ndarray
    ) -> Image.Image:
        """
        Load input into a PIL Image in RGB.
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(np.uint8(image_input)).convert('RGB')
        else:
            raise ValueError("image_input must be a file path, PIL Image, or numpy array.")
        return image

    def _filter_boxes(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        threshold: float
    ) -> tuple[np.ndarray, list[float]]:
        """
        Filter boxes by class (scratch==1) and score threshold.
        """
        keep = (labels == 1) & (scores >= threshold)
        boxes_kept = boxes[keep]
        scores_kept = scores[keep]

        if boxes_kept.numel() > 0:
            boxes_np = boxes_kept.detach().cpu().numpy().astype(int)
            scores_list = [float(s) for s in scores_kept]
        else:
            boxes_np, scores_list = np.array([]), []

        return boxes_np, scores_list

    def box_to_mask(
        self,
        image: Image.Image,
        boxes: np.ndarray
    ) -> np.ndarray:
        """
        Convert bounding boxes to a binary mask.
        """
        width, height = image.size
        mask = np.zeros((height, width), dtype=np.uint8)

        if boxes.size == 0:
            return mask
        
        for box in boxes:
            x0, y0, x1, y1 = map(int, box)
            x0, x1 = np.clip([x0, x1], 0, width)
            y0, y1 = np.clip([y0, y1], 0, height)
            mask[y0:y1, x0:x1] = 1
        return mask
