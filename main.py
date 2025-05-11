import argparse
import os

from typing import get_args
from pathlib import Path

from aircraft_anomaly_detection.dataloader.loader import DatasetString, AnomalyDataset
from aircraft_anomaly_detection.pipeline import evaluate, infer
from aircraft_anomaly_detection.models.owlvit import OwlViT

def main():
    parser = argparse.ArgumentParser(description="Aircraft Engine Anomaly Detection")
    parser.add_argument("--input", type=str, help="Dataset name (synthetic, lufthansa, mvtec, all) or path to an image file")
    parser.add_argument("--model_name", type=str, help="Name of the model to use")
    parser.add_argument("--model_path", type=str, help="Path of the model to use")
    parser.add_argument("--output_path", type=str, help="Path of folder output")

    args = parser.parse_args()

    # Parse model
    if args.model_name.lower() == "owlvit":
        model = OwlViT(args.model_path)
    else:
        raise ValueError(f"Invalid model name: {args.model_name}. Must be 'owlvit' or 'gemini'.")

    # Parse dataset
    if args.input.lower() in get_args(DatasetString) or args.input.lower() == "all":
        evaluate(AnomalyDataset(args.input.lower()), model, args.output_path)
    elif os.path.isfile(args.input):
        infer(Path(args.input), model, args.output_path)
    else:
        raise ValueError(f"Invalid input: {args.input}. Must be a known dataset or valid image path.")
    


if __name__ == "__main__":
    main()
