import numpy as np

from aircraft_anomaly_detection.schemas.data import Annotation


class AnnotationPostProcessor:
    def __call__(self, annotation: Annotation) -> None:
        pass


class BBoxSizeFilter(AnnotationPostProcessor):
    """
    A filter that removes bounding boxes whose areas are either too small or too large
    relative to the total object area.
    """

    def __init__(self, object_area: float, min_bb_size: float = 0.0, max_bb_size: float = 0.1):
        """
        Parameters:
        - object_area: Total area of the object being annotated.
        - min_bb_size: Minimum relative size of a bounding box (compared to object_area).
        - max_bb_size: Maximum relative size of a bounding box (compared to object_area).
        """
        self.object_area = object_area
        self.min_bb_size = min_bb_size
        self.max_bb_size = max_bb_size

    def __call__(self, annotation: Annotation) -> None:
        """
        Filters out bounding boxes from the annotation that fall outside the allowed size range.
        """
        valid_bbox_idx = []

        for i in range(len(annotation.bboxes)):
            x1, y1, x2, y2 = annotation.bboxes[i]
            bbox_area = (x2 - x1) * (y2 - y1)

            if self.object_area * self.min_bb_size < bbox_area <= self.object_area * self.max_bb_size:
                valid_bbox_idx.append(i)

        if annotation.mask is not None:
            valid_bbox_mask = np.zeros_like(annotation.mask)
            for i in valid_bbox_idx:
                x1, y1, x2, y2 = annotation.bboxes[i]
                valid_bbox_mask[y1:y2, x1:x2] = 1.0
            annotation.mask = annotation.mask * valid_bbox_mask

        print(
            f"BBoxSizeFilter: removed {len(annotation.bboxes) - len(valid_bbox_idx)} boxes from {len(annotation.bboxes)}"
        )

        annotation.bboxes = [annotation.bboxes[i] for i in valid_bbox_idx]
        annotation.scores = [annotation.scores[i] for i in valid_bbox_idx]

        annotation.damaged = len(annotation.bboxes) > 0


class BBoxOnObjectFilter(AnnotationPostProcessor):
    """
    A filter that removes bounding boxes that primarily lie on the background
    (based on a provided background mask).
    """

    def __init__(self, background_mask):
        """
        Parameters:
        - background_mask: A binary mask (e.g., 0 for object, 1 for background) of the same shape
                           as the annotated image.
        """
        self.background_mask = background_mask

    def __call__(self, annotation: Annotation) -> None:
        """
        Removes bounding boxes that overlap more than 50% with the background.
        """
        valid_bbox_idx = []

        for i in range(len(annotation.bboxes)):
            x1, y1, x2, y2 = annotation.bboxes[i]

            # Skip invalid bounding boxes with zero area
            if x1 == x2 or y1 == y2:
                continue

            area = (x2 - x1) * (y2 - y1)
            background_area = self.background_mask[y1:y2, x1:x2].sum()

            if background_area / area < 0.5:
                valid_bbox_idx.append(i)

        if annotation.mask is not None:
            valid_bbox_mask = np.zeros_like(annotation.mask)
            for i in valid_bbox_idx:
                x1, y1, x2, y2 = annotation.bboxes[i]
                valid_bbox_mask[y1:y2, x1:x2] = 1.0
            annotation.mask = annotation.mask * valid_bbox_mask
        print(
            f"BBoxOnObjectFilter: removed {len(annotation.bboxes) - len(valid_bbox_idx)} boxes from {len(annotation.bboxes)}"
        )

        annotation.bboxes = [annotation.bboxes[i] for i in valid_bbox_idx]
        annotation.scores = [annotation.scores[i] for i in valid_bbox_idx]

        assert len(annotation.bboxes) == len(valid_bbox_idx) and len(annotation.scores) == len(valid_bbox_idx)

        # TODO: Add similar logic to clean up pixel-level mask annotations
        annotation.damaged = len(annotation.bboxes) > 0
