"""Main entrypoint for aircraft engine anomaly detection.

This module allows evaluation on a dataset or inference on a single image
using different models (OwlViT or FasterRCNN or SAA).
"""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import yaml

from aircraft_anomaly_detection.dataloader.loader import AnomalyDataset, DatasetString
from aircraft_anomaly_detection.interface.model import ModelInterface
from aircraft_anomaly_detection.models.faster_rcnn import FasterRCNN
from aircraft_anomaly_detection.models.owlvit import OwlViT
from aircraft_anomaly_detection.models.saa import saa
from aircraft_anomaly_detection.pipeline import evaluate, infer
from aircraft_anomaly_detection.preprocessing import CLIPBackgroundRemover, PreProcessor


def main(args: SimpleNamespace) -> None:
    """Run evaluation or inference based on provided arguments.

    Args:
        args: SimpleNamespace with attributes:
            model_name (str): 'owlvit' or 'fasterrcnn'.
            remove_background (bool): Whether to remove background in preprocessing.
            model_config (dict): Model configuration.
            preprocessing (Optional[str]): Name of the preprocessing pipeline.
            input (str): Dataset identifier or image file path.
            output_path (str): Directory to save results.
            category (Optional[str]): Specific category for dataset evaluation.

    Raises:
        ValueError: If model_name is invalid or input is neither a dataset nor a file.
    """
    model_name = args.model_name.lower()
    if model_name == "owlvit":
        model: ModelInterface = OwlViT(**args.model_config)
    elif model_name == "fasterrcnn":
        model: ModelInterface = FasterRCNN(**args.model_config)
    elif model_name == "saa":
        model: ModelInterface = saa.SAA(**args.model_config)
    else:
        raise ValueError(f"Invalid model name: {args.model_name!r}. Must be 'owlvit' or 'fasterrcnn'.")

    background_remover = CLIPBackgroundRemover() if args.remove_background else None
    preprocessor = PreProcessor(args.preprocessing) if args.preprocessing else None

    input_val = args.input.lower()
    if input_val in get_args(DatasetString) or input_val == "all":
        category = getattr(args, "category", None)
        evaluate(
            AnomalyDataset(input_val, category),
            model,
            args.output_path,
            background_remover,
            preprocessor,
            args,
        )
    elif os.path.isfile(args.input):
        infer(Path(args.input), model, args.output_path)
    else:
        raise ValueError(f"Invalid input: {args.input!r}. Must be a known dataset or a valid image path.")


def run() -> None:
    """Load configuration from a YAML file and invoke main().

    Args:
        config_path (str): Path to the YAML configuration file.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Aircraft engine anomaly detection")
    parser.add_argument(
        "--config", type=str, default="configs/main_config.yaml", help="Path to configuration YAML file"
    )
    parser.add_argument("--model_config", type=str, help="Path to model configuration YAML file")
    args_cli = parser.parse_args()

    # Load configuration from the provided path
    with open(args_cli.config, "r") as fp:
        cfg = yaml.safe_load(fp)

    args = SimpleNamespace(**cfg)
    model_cfg = {}
    if args_cli.model_config:
        with open(args_cli.model_config, "r") as fp:
            model_cfg = yaml.safe_load(fp)
        # TODO: improve config management
    args.model_config = model_cfg

    print(f"Running config: {args}")
    main(args)


if __name__ == "__main__":
    run()
