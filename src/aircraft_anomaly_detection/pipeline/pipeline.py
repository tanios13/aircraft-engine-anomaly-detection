import numpy as np
from PIL import Image

from ..interfaces import Prediction


class AnomalyPipeline:
    def __init__(self):
        pass

    def run(self, image_input: str | Image.Image | np.ndarray) -> Prediction:
        
        raise NotImplementedError("The run method must be implemented in the subclass.")

        
        return Prediction()