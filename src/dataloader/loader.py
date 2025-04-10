import os
import pathlib
import tarfile
from collections.abc import Callable
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

    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        self.components = set(meta.component for meta in self.metadata)
        return f"AnomalyDataset(dataset={self.dataset}, num_samples={len(self.data)}) \n \
                components={self.components})"

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
            mvtech_root = self.data_root / "mvtech"
            if not mvtech_root.exists():
                raise ValueError(f"MVTech folder {mvtech_root} does not exist.")

            processed_folders = set()

            # First, check for tar.xz files.
            mvtech_tar_files = list(mvtech_root.glob("*.tar.xz"))
            for tar_file in sorted(mvtech_tar_files):
                if not tar_file.is_file():
                    continue
                # Determine the expected extracted folder name.
                folder_name = tar_file.name[:-7] if tar_file.name.endswith(".tar.xz") else tar_file.stem
                processed_folders.add(folder_name)
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
                self._process_mvtech_extracted_folder(extracted_folder, source_desc=tar_file.name)

            # Next, check for any extracted folders (without corresponding tar.xz files).
            for folder in mvtech_root.iterdir():
                if folder.is_dir() and folder.name not in processed_folders:
                    # Check if the folder contains expected MVTech subdirectories.
                    if (folder / "train").exists() or (folder / "test").exists():
                        print(f"Processing already extracted folder: {folder}")
                        self._process_mvtech_extracted_folder(folder, source_desc="extracted_folder")

            print(f"Loaded {len(self.data)} files from mvtech dataset.")

    def __len__(self) -> int:
        """Return the total number of loaded samples."""
        return len(self.data)

    def _process_mvtech_extracted_folder(self, extracted_folder: pathlib.Path, source_desc: str) -> None:
        """
        Process an extracted MVTech folder by loading training and test images along with metadata.

        Parameters:
            extracted_folder: The folder assumed to contain MVTech data.
            source_desc: A description of the source (e.g., the tar file name or a default string) for the metadata.
        """
        folder_name = extracted_folder.name

        # Fix permissions (in case they haven't been set properly)
        self._fix_permissions(extracted_folder)

        # Process training images: only the 'good' images are used for training.
        train_good_dir = extracted_folder / "train" / "good"
        if train_good_dir.exists():
            for file in sorted(train_good_dir.iterdir()):
                if file.is_file() and file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    self.data.append(file)
                    self.labels.append(0)  # Good images are labeled as normal (0)
                    self.metadata.append(
                        Metadata(
                            component=folder_name,
                            condition="normal",
                            description=f"MVTech {source_desc} (train)",
                        )
                    )

        # Process test images.
        test_dir = extracted_folder / "test"
        ground_truth_dir = extracted_folder / "ground_truth"
        if test_dir.exists():
            for condition_dir in sorted(test_dir.iterdir()):
                if condition_dir.is_dir():
                    condition = condition_dir.name.lower()  # e.g., bent_wire, cable_swap, good, etc.
                    for file in sorted(condition_dir.iterdir()):
                        if file.is_file() and file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                            # Label: 0 if condition is "good", else 1 (anomaly)
                            label = 0 if condition == "good" else 1
                            self.data.append(file)
                            self.labels.append(label)
                            # Build corresponding ground truth mask path.
                            gt = ground_truth_dir / condition_dir.name / f"{os.path.splitext(file.name)[0]}_mask.png"
                            self.metadata.append(
                                Metadata(
                                    component=folder_name,
                                    condition=condition,
                                    description=f"MVTech {source_desc} (test)",
                                    ground_truth=gt if gt.exists() else None,
                                )
                            )

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

    @classmethod
    def from_existing(
        cls,
        *,
        data: list[pathlib.Path],
        labels: list[int],
        metadata: list[Metadata],
        dataset: Literal["synthetic", "mvtech", "all"],
        data_root: pathlib.Path,
    ) -> "AnomalyDataset":
        """
        Alternative constructor to create an instance using pre-loaded data.
        """
        instance = cls.__new__(cls)  # Bypass __init__
        instance.data = data
        instance.labels = labels
        instance.metadata = metadata
        instance.dataset = dataset
        instance.data_root = data_root
        return instance

    def filter_by_component(self, component: str) -> "AnomalyDataset":
        """
        Returns a new instance of AnomalyDataset containing only samples whose metadata match the specified component.
        The filtering is performed in a case-insensitive manner.
        """
        filtered_data = []
        filtered_labels = []
        filtered_metadata = []
        for file_path, label, meta in zip(self.data, self.labels, self.metadata):
            if meta.component.lower() == component.lower():
                filtered_data.append(file_path)
                filtered_labels.append(label)
                filtered_metadata.append(meta)
        return AnomalyDataset.from_existing(
            data=filtered_data,
            labels=filtered_labels,
            metadata=filtered_metadata,
            dataset=self.dataset,
            data_root=self.data_root,
        )

    def filter_by(self, filter_func: Callable[[pathlib.Path, Metadata, int], bool]) -> "AnomalyDataset":
        """
        Returns a new instance of AnomalyDataset that includes only the samples for which the
        provided filter function returns True, using the sample's metadata.

        Args:
            filter_func (Callable[[pathlib.Path, Metadata, int], bool]): A callable that takes a file path,
            metadata, and label and returns a boolean indicating whether to include the sample.

        Returns:
            An AnomalyDataset instance filtered according to the callable.
        """
        filtered_data = []
        filtered_labels = []
        filtered_metadata = []
        for file_path, label, meta in zip(self.data, self.labels, self.metadata):
            if filter_func(file_path, meta, label):
                filtered_data.append(file_path)
                filtered_labels.append(label)
                filtered_metadata.append(meta)
        return AnomalyDataset.from_existing(
            data=filtered_data,
            labels=filtered_labels,
            metadata=filtered_metadata,
            dataset=self.dataset,
            data_root=self.data_root,
        )
