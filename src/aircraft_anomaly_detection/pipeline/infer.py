import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import tqdm
from PIL import Image

from aircraft_anomaly_detection.interface.model import ModelInterface
from aircraft_anomaly_detection.pipeline.pipeline_utils import dump_annotations, prepare_postprocessors
from aircraft_anomaly_detection.viz_utils import visualize_bb_predictions


def infer(
    input: str | Path,
    model: ModelInterface,
    output: str | Path,
    background_remover: Callable | None = None,
    preprocessor: Callable | None = None,
    args: SimpleNamespace | None = None,
):
    """
    Run inference on an image or a directory of images using the specified model.

    Parameters:
        input (str | Path): Path to a single image file or a directory containing images.
        model (ModelInterface): The model object used to make predictions. Must implement a `predict(image)` method.
        output (str | Path): Path to the directory where prediction visualizations and annotations will be saved.
        background_remover (Callable | None, optional): A function that removes the background from an image.
            Should return a tuple (processed_image, background_mask). If None, no background removal is performed.
        preprocessor (Callable | None, optional): A function to preprocess the input image before prediction.
            If None, no preprocessing is applied.
        args (SimpleNamespace | None, optional): Additional arguments for customizing the pipeline, such as:
            - dataset_idx: Iterable of indices to process specific images from the dataset.
            - Any additional config options used by postprocessors.

    Behavior:
        - Loads each image (or all images in a directory).
        - Optionally removes backgrounds and preprocesses the images.
        - Performs prediction with the given model.
        - Applies any specified postprocessing steps.
        - Saves prediction visualizations and metadata to the output directory.

    Output:
        - Visualizations of predictions saved as PNG files in the output directory.
        - Annotations and metadata saved in a structured format (e.g., JSON).

    Raises:
        OSError: If an image cannot be loaded.
    """
    input = Path(input)
    output_dir = Path(output)

    image_paths = [path for path in input.iterdir() if path.suffix in (".png", ".jpg")]

    pred_annotation_list = []
    images_metadata = []
    image_idx = []

    cmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    dataset_idx = range(len(image_paths))
    if hasattr(args, "dataset_idx") and args.dataset_idx is not None:
        dataset_idx = args.dataset_idx
    for i in tqdm.tqdm(dataset_idx):
        image_path = image_paths[i]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise OSError(f"Error loading file {image_path}: {e}")

        # save metadata
        images_metadata.append(
            {
                "id": i,
                "file_name": image_path.name,
                "height": image.size[1],
                "width": image.size[0],
            }
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

        # postprocessing
        for postprocessor in postprocessors:
            postprocessor(pred_annotation)

        pred_annotation_list.append(pred_annotation)
        image_idx.append(i)

        # create directory if it does not exist yet
        if output_dir is not None and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        visualize_bb_predictions(
            no_background_image, None, pred_annotation, cmap, save_path=output_dir.joinpath(f"image_{i}.png")
        )

    # save pred annotations
    dump_annotations(images_metadata, pred_annotation_list, image_idx, output_dir)
