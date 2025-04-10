import os
import pathlib
import tarfile
from typing import Literal

from PIL import Image  # Optional: for loading images
from pydantic import BaseModel, Field, FilePath


class Metadata(BaseModel):
    component: str  # e.g., oil_pump, pistons, etc.
    condition: str  # e.g., normal, scratched
    ground_truth: FilePath | None = None  # Optional ground truth field
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
        elif dataset == "mvtech":
            # Look for tar.xz files in the data folder.
            mvtech_root = self.data_root / "mvtech"
            mvtech_tar_files = list(mvtech_root.glob("*.tar.xz"))
            if not mvtech_tar_files:
                raise ValueError(f"No tar.xz files found for mvtech dataset in {self.data_root}")

            # Process each tar file.
            for tar_file in sorted(mvtech_tar_files):
                if not tar_file.is_file():
                    continue

                # Determine the expected extracted folder name by removing the ".tar.xz" suffix.
                # For example, "cable.tar.xz" becomes "cable".
                folder_name = tar_file.name[:-7] if tar_file.name.endswith(".tar.xz") else tar_file.stem
                extracted_folder = mvtech_root / folder_name
                if not extracted_folder.exists():
                    print(f"Extracting {tar_file} into {extracted_folder} ...")
                    try:
                        with tarfile.open(tar_file, "r:xz") as tf:
                            tf.extractall(path=mvtech_root)
                        # Fix permissions on the extracted files.
                        self._fix_permissions(extracted_folder)
                    except Exception as e:
                        print(f"Error extracting {tar_file}: {e}")
                        continue
                else:
                    print(f"Using existing extracted folder: {extracted_folder}")
                    self._fix_permissions(extracted_folder)

                # Now assume that the extracted folder has a structure similar to:
                #   <extracted_folder>/
                #       train/
                #           good/
                #       test/
                #           <condition>/   (e.g., bent_wire, cable_swap, etc.)
                #       ground_truth/ (optional, for test images with anomalies)
                #
                # Load training images: only "good" images.
                train_good_dir = extracted_folder / "train" / "good"
                if train_good_dir.exists():
                    for file in sorted(train_good_dir.iterdir()):
                        if file.is_file() and file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                            self.data.append(file)
                            self.labels.append(0)  # Good images
                            self.metadata.append(
                                Metadata(
                                    component=folder_name,
                                    condition="normal",
                                    description=f"MVTech {tar_file.name} (train)",
                                )
                            )

                # Load test images from the "test" folder.
                test_dir = extracted_folder / "test"
                # Optionally, load corresponding ground truth masks from the ground_truth folder.
                ground_truth_dir = extracted_folder / "ground_truth"

                if test_dir.exists():
                    for condition_dir in sorted(test_dir.iterdir()):
                        if condition_dir.is_dir():
                            condition = condition_dir.name.lower()  # e.g., bent_wire, cable_swap, good, etc.
                            for file in sorted(condition_dir.iterdir()):
                                if file.is_file() and file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                                    # For test images, label 0 if condition is "good", else 1 indicating anomaly.
                                    label = 0 if condition == "good" else 1
                                    self.data.append(file)
                                    self.labels.append(label)
                                    gt = (
                                        ground_truth_dir
                                        / condition_dir.name
                                        / f"{os.path.splitext(file.name)[0]}_mask.png"
                                    )
                                    self.metadata.append(
                                        Metadata(
                                            component=folder_name,
                                            condition=condition,
                                            description=f"MVTech {tar_file.name} (test)",
                                            ground_truth=gt if gt.exists() else None,
                                        )
                                    )

            print(f"Loaded {len(self.data)} files from mvtech dataset.")

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

    def _fix_permissions(self, folder: pathlib.Path) -> None:
        """
        Recursively set full permissions (0o777) for all files and directories within the given folder.
        """
        for path in folder.rglob("*"):
            try:
                os.chmod(path, 0o777)
            except Exception as e:
                print(f"Failed to change permissions for {path}: {e}")
