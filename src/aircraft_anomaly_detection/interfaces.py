from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image


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
    bboxes: list[list[float]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    bboxes_labels: list[str] = field(default_factory=list)
    mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate list lengths right after initialisation."""
        if len(self.bboxes) not in (0, len(self.scores), len(self.bboxes_labels)):
            raise ValueError(
                f"`bboxes` {len(self.bboxes)} must match `scores` {len(self.scores)} and `bboxes_labels` {len(self.bboxes_labels)}"
            )


class ModelInterface(ABC):
    """Abstract interface that all prediction models must implement."""

    @abstractmethod
    def predict(self, input_image: str | Image.Image | np.ndarray, **kwargs) -> Annotation:
        """Return model predictions for the supplied input.

        Args:
            input: The input passed to the model. The concrete type depends
                on the implementation (e.g., a file path, a ``PIL.Image``, or a
                NumPy array).

        Returns:
            An :class:`Annotation` instance for single-image inputs or a list of
            :class:`Annotation` instances for batched inputs.
        """
        raise NotImplementedError
