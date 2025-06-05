import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from aircraft_anomaly_detection.postprocessing import (
    BBoxOnObjectFilter,
    BBoxSizeFilter,
    CLIPAnomalyFilter,
    CLIPAnomalySegmentor,
    TopKFilter,
)


def prepare_postprocessors(args: SimpleNamespace | None, background_mask: np.ndarray | None, image: Image.Image | None):
    postprocessing = []
    if hasattr(args, "postprocessing") and args.postprocessing is not None:
        postprocessing = args.postprocessing
    postprocessors = []
    if "BBoxOnObjectFilter" in postprocessing:
        if background_mask is None:
            raise ValueError("To use BBoxOnObjectFilter background_mask can't be None")
        postprocessors.append(BBoxOnObjectFilter(background_mask))
    if "BBoxSizeFilter" in postprocessing:
        w, h = image.size
        object_size = w * h
        if background_mask is not None:
            object_size = w * h - background_mask.sum()
        postprocessors.append(BBoxSizeFilter(object_size))
    if "CLIPAnomalySegmentor" in postprocessing:
        postprocessors.append(CLIPAnomalySegmentor())
    if "CLIPAnomalyFilter" in postprocessing:
        postprocessors.append(CLIPAnomalyFilter())
    if "TopKFilter" in postprocessing:
        postprocessors.append(TopKFilter())
    return postprocessors


def dump_annotations(images_metadata: list[dict], pred_annotations: list[dict], image_idx: list[int], output_dir: Path):
    """
    Dumps predicted annotations in COCO json format.
    """
    iso_format_date = datetime.now(timezone.utc).isoformat()

    annotations = []
    for image_id, pred_annotation in zip(image_idx, pred_annotations):
        for bbox in pred_annotation.bboxes:
            annotations.append(
                {
                    "id": len(annotations),
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])],
                    "area": float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
                    "iscrowd": 0,
                }
            )

    annotation_dict = {
        "info": {
            "year": "2025",
            "version": "1",
            "description": "Predicted annotations",
            "contributor": "",
            "date_created": iso_format_date,
        },
        "licenses": [],
        "categories": [
            {"id": 0, "name": "blank", "supercategory": "none"},
            {"id": 1, "name": "defect", "supercategory": "none"},
        ],
        "images": images_metadata,
        "annotations": annotations,
    }

    try:
        with open(output_dir.joinpath("predicted_annotations.coco.json"), "w") as file:
            json.dump(annotation_dict, file)
        print(f"Saved predicted annotations in: {output_dir.joinpath('predicted_annotations.coco.json')}")
    except Exception as e:
        print(f"Failed to save the predicted annotations, error: {e}")
