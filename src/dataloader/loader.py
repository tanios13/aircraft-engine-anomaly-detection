import pathlib
from typing import Literal

from PIL import Image  # Optional: for loading images
from pydantic import BaseModel, Field


class Metadata(BaseModel):
    component: str  # e.g., oil_pump, pistons, etc.
    condition: str  # e.g., normal, scratched
    description: str = Field(default="")  # Optional description field


class AnomalyDataset:
    def __init__(self, dataset: Literal["synthetic", "mvtech", "all"]):
        """
        Initializes the dataset.

        The `dataset` parameter indicates which dataset to load:
          - "synthetic": Loads files from the Synthetic_anomaly_dataset folder.
          - "mvtech":   Loads files from the MVTech dataset (assumed structure).
          - "all":      Loads both the synthetic and mvtech datasets.

        The class assumes that the data folder is located two directories up from this file.
        """
        self.dataset = dataset
        self.data_root = pathlib.Path(__file__).parent.parent.parent.resolve() / "data"
        # Initialize empty lists to store file paths, metadata and labels.
        self.data: list[pathlib.Path] = []
        self.metadata: list[Metadata] = []
        self.labels: list[int] = []

        # Load the relevant dataset(s) based on the user input.
        if self.dataset == "all":
            self._load("synthetic")
            self._load("mvtech")
        else:
            self._load(self.dataset)

    def _load(self, dataset: Literal["synthetic", "mvtech"]) -> None:
        """
        Loads the files for the specified dataset.

        For the synthetic dataset, it expects a folder structure as:
            Synthetic_anomaly_dataset/
                <component>/
                    normal/
                    scratched/

        Each component represents a machine part (e.g., oil_pump, pistons, etc.). For each image,
        we store the file path, assign a label (0 for normal, 1 for scratched), and record its metadata.

        For "mvtech", you can implement the appropriate logic based on the dataset's structure.
        """
        if dataset not in ["synthetic", "mvtech"]:
            raise ValueError(f"Dataset {dataset} is not supported. Choose 'synthetic' or 'mvtech'.")

        if dataset == "synthetic":
            synthetic_root = self.data_root / "Synthetic_anomaly_dataset"
            if not synthetic_root.exists():
                raise ValueError(f"Synthetic dataset directory {synthetic_root} does not exist.")

            # Iterate over each component folder (oil_pump, pistons, etc.)
            for component_dir in synthetic_root.iterdir():
                if component_dir.is_dir():
                    # Iterate over each condition folder inside the component folder.
                    for condition_dir in component_dir.iterdir():
                        if condition_dir.is_dir():
                            condition = condition_dir.name.lower()  # Expected to be "normal" or "scratched"
                            if condition not in ["normal", "scratched"]:
                                print(f"Skipping unrecognized condition folder: {condition_dir}")
                                continue
                            # Iterate through each file in the condition directory.
                            for file_path in condition_dir.iterdir():
                                self.data.append(file_path)
                                # Assign a label: 0 for "normal", 1 for "scratched"
                                label = 0 if condition == "normal" else 1
                                self.labels.append(label)
                                # Save metadata for this sample.
                                self.metadata.append(Metadata(component=component_dir.name, condition=condition))
            print(f"Loaded {len(self.data)} files from synthetic dataset at {synthetic_root}")

    def __len__(self) -> int:
        """Return the total number of loaded samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int, Metadata]:
        """
        Returns a single sample from the dataset as a tuple (data, label).
        Here, 'data' could be further processed. For example, if you want to load images, you could use PIL.Image.open.
        """
        file_path = self.data[idx]
        label = self.labels[idx]
        metadata = self.metadata[idx]

        # Optionally load the image (or file) here
        try:
            # This will open an image file using PIL. Adjust accordingly if working with different data.
            image = Image.open(file_path).convert("RGB")
        except Exception as e:
            raise OSError(f"Error loading file {file_path}: {e}")

        return image, label, metadata
