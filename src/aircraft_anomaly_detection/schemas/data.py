from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, FilePath


@dataclass
class Annotation:
    """Represents the result of a model prediction for a single image.

    Attributes:
        image: The original input image or ``None`` if unavailable.
        damaged: ``True`` if the object in the image is damaged, ``False`` if
            undamaged, or ``None`` if the model could not decide.
        bboxes: A list of bounding boxes in ``[x_min, y_min, x_max, y_max]``
            format (pixel coordinates).
        scores: The confidence score for each bounding box, in the same order
            as *bboxes*.
        bboxes_labels: The class label for each bounding box, in the same order
            as *bboxes*.
        mask: An optional segmentation mask for the full image.

    Raises:
        ValueError: If the length of *bboxes* does not match the length of
            *scores* and *bboxes_labels*.
    """

    image: Image.Image | None = None
    damaged: bool | None = None
    bboxes: list[list[int]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    bboxes_labels: list[str] = field(default_factory=list)
    mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate list lengths right after initialisation."""
        if len(self.bboxes) not in (0, len(self.scores), len(self.bboxes_labels)):
            raise ValueError(
                f"`bboxes` {len(self.bboxes)} must match `scores` {len(self.scores)} and `bboxes_labels` {len(self.bboxes_labels)}"
            )
        if self.image is not None:
            width, height = self.image.size
        else:
            width, height = int(1e6), int(1e6)

        # Fixing negative bbox coordingates
        for bbox in self.bboxes:
            bbox[0] = np.clip(bbox[0], 0, width)
            bbox[2] = np.clip(bbox[2], 0, width)
            bbox[1] = np.clip(bbox[1], 0, height)
            bbox[3] = np.clip(bbox[3], 0, height)
        if self.image is not None and self.mask is None:
            self.box_to_mask()

    def box_to_mask(self) -> None:
        """
        Convert bounding boxes to a binary mask.

        Args:
            image (Image.Image): The input image.
            boxes (np.ndarray): Array of bounding boxes (x0, y0, x1, y1).

        Returns:
            np.ndarray: Binary mask with 1s inside boxes, 0 elsewhere.
        """
        image = self.image
        width, height = image.size
        self.mask = np.zeros((height, width), dtype=np.uint8)

        for box in self.bboxes:
            x0, y0, x1, y1 = map(int, box)  # ensure integers
            x0 = np.clip(x0, 0, width)
            x1 = np.clip(x1, 0, width)
            y0 = np.clip(y0, 0, height)
            y1 = np.clip(y1, 0, height)
            self.mask[y0:y1, x0:x1] = 1


class Metadata(BaseModel):
    component: str  # e.g., oil_pump, pistons, etc.
    condition: str  # e.g., normal, scratched
    ground_truth: FilePath | None = None  # Optional ground truth field
    image_path: FilePath | None = None  # Optional image path field
    description: str = Field(default="")  # Optional description field
    split: str = Field(default="")  # Optional split field (train/test/val)
    annotation: Annotation | None = None  # Optional annotations field

    class Config:
        arbitrary_types_allowed = True
