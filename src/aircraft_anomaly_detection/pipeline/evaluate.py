from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

from aircraft_anomaly_detection.dataloader.loader import AnomalyDataset
from aircraft_anomaly_detection.eval.evaluator import Evaluator
from aircraft_anomaly_detection.interfaces import ModelInterface
from aircraft_anomaly_detection.viz_utils import visualize_mask_overlap_with_image


def evaluate(
    dataset: AnomalyDataset, model: ModelInterface, output_dir: str = None, background_remover: Callable | None = None
):
    """
    Evaluate the model on the given dataset.


    Args:
        dataset (AnomalyDataset): The dataset to evaluate the model on.
    """

    # Compute the predictions and ground truth annotations
    grd_annotation_list, pred_annotation_list = [], []

    print("Predicting...")

    for i in tqdm.tqdm(range(len(dataset))):
        image, label, metadata = dataset[i]
        grd_annotation_list.append(metadata.annotation)

        # Remove background
        if background_remover is not None:
            no_background_image, background_mask = background_remover(image)
        else:
            no_background_image, background_mask = image, None

        pred_annotation = model.predict(image)

        # Postprocessing
        if background_remover is not None:
            refine_annotation(pred_annotation, background_mask)

        pred_annotation_list.append(pred_annotation)

        visualize_mask_overlap_with_image(
            no_background_image,
            metadata.annotation.mask,
            pred_annotation.mask,
            save_path=output_dir + f"image_{i}.png",
        )

    print("Evaluating...")
    # Evaluate the model
    evaluator = Evaluator(pred_annotation_list, grd_annotation_list)
    results = evaluator.eval()

    print("Saving results...")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir + "results.csv", index=False)

    evaluator.plot_confusion_matrix(output_dir + "confusion_matrix.png")


def refine_annotation(annotation, background_mask):
    """
    Removes rectangles with large background area.
    """
    valid_bbox_idx = []
    for i in range(len(annotation.bboxes)):
        x1, y1, x2, y2 = annotation.bboxes[i]
        cropped = annotation.image.crop((x1, y1, x2, y2))
        area = (x2 - x1) * (y2 - y1)
        background_area = np.sum(background_mask[y1:y2, x1:x2])
        if background_area / area < 0.5:
            valid_bbox_idx.append(i)
    print(f"Refinement: removed {len(annotation.bboxes) - len(valid_bbox_idx)} boxes from {len(annotation.bboxes)}")
    annotation.bboxes = [annotation.bboxes[i] for i in valid_bbox_idx]
    annotation.scores = [annotation.scores[i] for i in valid_bbox_idx]

    if annotation.mask is not None:
        annotation.mask[(background_mask == 1)] = 0

    # TODO: add similar logic for mask annotations
    annotation.damaged = len(annotation.bboxes) > 0
