from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from PIL import Image

from aircraft_anomaly_detection.schemas import Annotation


class DetectorInterface(ABC):
    @abstractmethod
    def predict(
        self,
        image: Image.Image,
        text_prompts: list[str],
        *,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> tuple[np.ndarray, list[float], list[str]]:
        """
        Predict bounding boxes and labels for an image given text prompts.

        Args:
            image: PIL Image.
            text_prompts: List of text prompts.

        Returns:
            Tuple of bounding boxes, scores, and labels.
        """
        raise NotImplementedError


class SegmentorInterface(ABC):
    """Nominal contract for any prompt‑based image segmentor."""

    @abstractmethod
    def set_image(self, image: Image.Image) -> None:
        """Embed *image* once so subsequent prompts are fast.

        Args:
            image: An RGB ``PIL.Image.Image`` instance.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        boxes_xyxy: np.ndarray,
        *,
        multimask_output: bool = False,
    ) -> np.ndarray:
        """Generate segmentation masks for ``boxes_xyxy``.

        Args:
            boxes_xyxy: Absolute XYXY bounding boxes with shape ``(N, 4)``.
            multimask_output: If ``True`` return SAM's three diverse masks per
                box; otherwise return the single best mask.

        Returns:
            A boolean array of shape ``(N, H, W)`` where ``H`` and ``W`` match
            the height and width of the image passed to :py:meth:`set_image`.
        """
        raise NotImplementedError


class ModelInterface(ABC):
    """Abstract interface that all prediction models must implement."""

    @abstractmethod
    def predict(self, input_image: str | Image.Image | np.ndarray, **kwargs: dict[str, Any]) -> Annotation:
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

    def load_image(self, input_image: str | Image.Image | np.ndarray) -> Image.Image:
        """
        Load an image from a file path, a numpy array, or a PIL image, and return a PIL Image in RGB.

        Args:
            input_image (Union[str, Image.Image, np.ndarray]): Image file path, numpy array, or PIL Image.

        Returns:
            Image.Image: Loaded image in RGB mode.
        """
        if isinstance(input_image, str):
            image = Image.open(input_image).convert("RGB")
        elif isinstance(input_image, Image.Image):
            image = input_image.convert("RGB")
        elif isinstance(input_image, np.ndarray):
            image = Image.fromarray(np.uint8(input_image)).convert("RGB")
        else:
            raise ValueError("input_image must be a file path (str), a PIL.Image.Image, or a numpy.ndarray.")
        return image
