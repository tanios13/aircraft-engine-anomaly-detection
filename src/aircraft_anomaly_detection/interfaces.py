import numpy as np

from dataclasses import dataclass, field
from typing import List, Optional, Any, Union
from abc import ABC, abstractmethod
from PIL import Image


@dataclass
class Annotation:
    """
    Represents the output of a model prediction on an image.
    """
    image: Image.Image
    label: Optional[bool]       = None                          # False=undamaged, True=damaged
    bboxes: List[List[float]]   = field(default_factory=list)
    scores: List[float]         = field(default_factory=list)
    bboxes_labels: List[str]    = field(default_factory=list)
    mask: Optional[np.ndarray]  = None           

    def __post_init__(self):
        # simple length check
        if len(self.bboxes) not in (0, len(self.scores), len(self.bboxes_labels)):
            raise ValueError(
                f"`bboxes` ({len(self.bboxes)}) must match `scores` ({len(self.scores)}) "
                f"and `bboxes_labels` ({len(self.bboxes_labels)})"
            )

class ModelInterface(ABC):
    """
    Interface for all models used in the project.
    """
    @abstractmethod
    def predict(self, input: Any) -> Union[Annotation, List[Annotation]]:
        """Run a prediction and return results."""
        pass