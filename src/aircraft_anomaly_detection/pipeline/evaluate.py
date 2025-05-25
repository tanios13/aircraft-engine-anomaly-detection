import os
from collections.abc import Callable
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from PIL import Image

from aircraft_anomaly_detection.dataloader.loader import AnomalyDataset
from aircraft_anomaly_detection.eval.evaluator import Evaluator
from aircraft_anomaly_detection.interface.model import ModelInterface
from aircraft_anomaly_detection.postprocessing import (
    BBoxOnObjectFilter,
    BBoxSizeFilter,
    CLIPAnomalyFilter,
    CLIPAnomalySegmentor,
    TopKFilter,
)
from aircraft_anomaly_detection.schemas.data import Annotation
from aircraft_anomaly_detection.viz_utils import (
    draw_annotation,
    visualize_bb_predictions,
    visualize_mask_overlap_with_image,
)


def evaluate(
    dataset: AnomalyDataset,
    model: ModelInterface,
    output_dir: str | None = None,
    background_remover: Callable | None = None,
    preprocessor: Callable | None = None,
    args: SimpleNamespace | None = None,
) -> None:
    """
    Evaluate the model on the given dataset.


    Args:
        dataset (AnomalyDataset): The dataset to evaluate the model on.
        model (ModelInterface): The model to evaluate.
        output_dir (str): The directory to save the results.
        background_remover (Callable | None): The background remover to use.
        preprocessor (Callable | None): The preprocessor to use.
        args (SimpleNamespace): The arguments to use.
    """

    # Compute the predictions and ground truth annotations
    grd_annotation_list, pred_annotation_list = [], []

    print("Predicting...")

    cmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    dataset_idx = range(len(dataset))
    if hasattr(args, "dataset_idx") and args.dataset_idx is not None:
        dataset_idx = args.dataset_idx
    for i in tqdm.tqdm(dataset_idx):
        image, label, metadata = dataset[i]
        grd_annotation_list.append(metadata.annotation)
        _ = draw_annotation(
            image,
            metadata.annotation,
            show_boxes=True,
            show_mask=True,
            save_path="0_orignal_annotated.png",
        )

        # Remove background
        if background_remover is not None:
            no_background_image, background_mask = background_remover(image)
        else:
            no_background_image, background_mask = image, None

        # Preparing postprocessors
        postprocessors = prepare_postprocessors(args, background_mask, image)

        if preprocessor is not None:
            image = preprocessor(image)

        pred_annotation = model.predict(image)

        # Postprocessing
        for postprocessor in postprocessors:
            postprocessor(pred_annotation)

        pred_annotation_list.append(pred_annotation)

        # create directory if it does not exist yet
        if output_dir is not None and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        visualize_bb_predictions(
            no_background_image, metadata.annotation, pred_annotation, cmap, save_path=output_dir + f"image_{i}.png"
        )

    print("Evaluating...")

    # Evaluate the model
    evaluator = Evaluator(pred_annotation_list, grd_annotation_list)
    results = evaluator.eval()

    print("Saving results...")

    # Prepare results as a dictionary
    if args is not None:
        meta_info = {
            "dataset": args.input.lower(),
            "model": args.model_name.lower(),
            "background_removal": "True" if args.remove_background else "False",
            "preprocessing": args.preprocessing if args.preprocessing else "None",
        }
    else:
        meta_info = {
            "dataset": "unknown",
            "model": "unknown",
            "background_removal": "False",
            "preprocessing": "None",
        }

    # Merge metadata and evaluation results, with metadata first
    combined_results = {**meta_info, **results}

    # Clean values: unwrap lists and convert numpy types
    flat_results = {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in combined_results.items()}
    flat_results = {k: (v.item() if hasattr(v, "item") else v) for k, v in flat_results.items()}

    # Save to DataFrame
    results_df = pd.DataFrame([flat_results])

    # Output path
    output_file = os.path.join(output_dir, str(meta_info["dataset"]) + "_results.csv")

    # Save or append
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        combined_df = pd.concat([existing_df, results_df], ignore_index=True)
        combined_df.to_csv(output_file, index=False)
    else:
        results_df.to_csv(output_file, index=False)

    # Plot confusion matrix
    evaluator.plot_confusion_matrix(os.path.join(output_dir, str(meta_info["dataset"]) + "_confusion_matrix.png"))
    evaluator.save_results_table(output_dir + "results_table.csv")


# Helpers #########


def prepare_postprocessors(args, background_mask, image):
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
