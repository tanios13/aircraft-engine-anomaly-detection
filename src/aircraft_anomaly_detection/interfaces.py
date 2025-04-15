import numpy as np
from PIL import Image
from pydantic import BaseModel, model_validator


class Prediction(BaseModel):
    
    image: Image.Image
    binary_label: bool | None
    bboxes : list[np.ndarray] | None
    labels : list[str] | None
    scores : list[float] | None
    masks : list[np.ndarray] | None
    
    @model_validator(mode='after')
    def check_lists_same_length(self):
        # Create a dictionary of the optional list fields
        list_fields = {
            'bboxes': self.bboxes,
            'labels': self.labels,
            'scores': self.scores,
            'masks': self.masks,
        }
        # Filter out the fields that are None
        present_lists = {field: lst for field, lst in list_fields.items() if lst is not None}
        
        # If there are any lists provided, check that they all have the same length.
        if present_lists:
            # Get the lengths of the provided lists
            lengths = {field: len(lst) for field, lst in present_lists.items()}
            # If there's more than one unique length, raise a ValueError.
            if len(set(lengths.values())) > 1:
                raise ValueError(
                    f"All list fields must have the same length, but got lengths: {lengths}"
                )
        return self
    
    
    
class GroundTruth(BaseModel):
    image: Image.Image
    binary_label: bool | None
    bboxes : list[np.ndarray] | None
    labels : list[str] | None
    scores : list[float] | None
    masks : list[np.ndarray] | None
    
    @model_validator(mode='after')
    def check_lists_same_length(self):
        # Create a dictionary of the optional list fields
        list_fields = {
            'bboxes': self.bboxes,
            'labels': self.labels,
            'scores': self.scores,
            'masks': self.masks,
        }
        # Filter out the fields that are None
        present_lists = {field: lst for field, lst in list_fields.items() if lst is not None}
        
        # If there are any lists provided, check that they all have the same length.
        if present_lists:
            # Get the lengths of the provided lists
            lengths = {field: len(lst) for field, lst in present_lists.items()}
            # If there's more than one unique length, raise a ValueError.
            if len(set(lengths.values())) > 1:
                raise ValueError(
                    f"All list fields must have the same length, but got lengths: {lengths}"
                )
        return self