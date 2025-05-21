import os
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import yaml

from aircraft_anomaly_detection.dataloader.loader import AnomalyDataset, DatasetString
from aircraft_anomaly_detection.models.faster_rcnn import FasterRCNN
from aircraft_anomaly_detection.models.owlvit import OwlViT
from aircraft_anomaly_detection.pipeline import evaluate, infer
from aircraft_anomaly_detection.preprocessing import CLIPBackgroundRemover,PreProcessor


def main(args):
    # Parse model
    if args.model_name.lower() == "owlvit":
        model = OwlViT(args.model_path)
    elif args.model_name.lower() == "fasterrcnn":
        model = FasterRCNN(args.model_path)
    else:
        raise ValueError(f"Invalid model name: {args.model_name}. Must be 'owlvit' or 'fasterrcnn'.")

    background_remover = CLIPBackgroundRemover() if args.remove_background else None
    preprocessor = PreProcessor(args.preprocessing) if args.preprocessing else None

    # Parse dataset
    if args.input.lower() in get_args(DatasetString) or args.input.lower() == "all":
        category = None if not hasattr(args, "category") else args.category
        evaluate(AnomalyDataset(args.input.lower(), category), model, args.output_path, background_remover,preprocessor,args)
    elif os.path.isfile(args.input):
        infer(Path(args.input), model, args.output_path)
    else:
        raise ValueError(f"Invalid input: {args.input}. Must be a known dataset or valid image path.")


def run(config_path="main_config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    args = SimpleNamespace(**cfg)
    print(f"Running config: {config_path}")
    main(args)


if __name__ == "__main__":
    run()
