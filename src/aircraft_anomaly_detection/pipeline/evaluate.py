import tqdm
import pandas as pd

from aircraft_anomaly_detection.interfaces import ModelInterface
from aircraft_anomaly_detection.dataloader.loader import AnomalyDataset
from aircraft_anomaly_detection.eval.evaluator import Evaluator
from aircraft_anomaly_detection.viz_utils import visualize_mask_overlap_with_image


def evaluate(dataset: AnomalyDataset, model: ModelInterface, output_dir: str = None):
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
        
        pred_annotation = model.predict(image)
        pred_annotation_list.append(pred_annotation)

        visualize_mask_overlap_with_image(
            image,
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
