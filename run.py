import itertools
import os
import subprocess

import yaml

# Base config content
base_config = {
    "input": "lufthansa",
    "model_name": "owlvit",
    "output_path": "data/results/owlvit/",
    "remove_background": True,
    "preprocessing": [],
    "postprocessing": ["BBoxSizeFilter", "BBoxOnObjectFilter", "TopKFilter"],
    "evaluator": {"threshold": 0.0, "iou_threshold": 0.1},
}

# Postprocessing options
options = {
    "BBoxSizeFilter": [True, False],
    "BBoxOnObjectFilter": [True, False],
    "TopKFilter": [True],  # Always included
}

# Generate all valid combinations
combinations = []
for size, on_object in itertools.product(options["BBoxSizeFilter"], options["BBoxOnObjectFilter"]):
    filters = []
    if size:
        filters.append("BBoxSizeFilter")
    if on_object:
        filters.append("BBoxOnObjectFilter")
    filters.append("TopKFilter")  # Always included
    combinations.append(filters)

# Output directory for generated configs
output_dir = "generated_configs"
os.makedirs(output_dir, exist_ok=True)

# Process each config
for idx, postprocessing in enumerate(combinations):
    config = base_config.copy()
    config["postprocessing"] = postprocessing

    # Generate filename
    postfix = "_".join(postprocessing)
    config_filename = f"main_config_{idx + 1}_{postfix}.yaml"
    config_path = os.path.join("configs", config_filename)

    # Write YAML config
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    print(f"[✓] Saved config: {config_path}")

    # Run command
    command = ["python", "main.py", "--config", config_path, "--model_config", "configs/owlvit_config.yaml"]
    print(f"[→] Running: {' '.join(command)}")
    subprocess.run(command)
