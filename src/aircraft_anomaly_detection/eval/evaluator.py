import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve

from aircraft_anomaly_detection.interface.model import ModelInterface
from aircraft_anomaly_detection.schemas.data import Annotation


class Evaluator:
    def __init__(self, predictions: list[Annotation], ground_truth: list[Annotation]) -> None:
        if len(predictions) != len(ground_truth):
            raise ValueError("The length of predictions and ground truth must be the same")
        self.predictions = predictions
        self.ground_truth = ground_truth
        self.threshold = 0.0

    def __call__(self):
        pass

    def accuracy(self):
        """
        Computes the accuracy of the binary predictions.
        """
        correct = sum(1 for pred, gt in zip(self.predictions, self.ground_truth) if pred.damaged == gt.damaged)
        return correct / len(self.predictions)

    def f1_score(self):
        """
        Computes the F1 score of the binary predictions.
        """
        y_true = [1 if gt.damaged else 0 for gt in self.ground_truth]
        y_pred = [1 if pred.damaged else 0 for pred in self.predictions]
        return f1_score(y_true, y_pred)

    def max_accuracy(self):
        """
        Computes the maximum accuracy based on predicted scores.
        """
        y_true = np.array([1 if gt.damaged else 0 for gt in self.ground_truth])
        y_probs = np.array([0.0 if len(pred.scores) == 0 else min(pred.scores) for pred in self.predictions])
        assert y_true.shape == y_probs.shape

        _, _, thresholds = roc_curve(y_true, y_probs)
        accuracies = []
        for threshold in thresholds:
            accuracies.append(accuracy_score(y_true, (y_probs > threshold)))
        return max(accuracies)

    def max_f1_score(self):
        y_true = np.array([1 if gt.damaged else 0 for gt in self.ground_truth])
        y_probs = np.array([0.0 if len(pred.scores) == 0 else min(pred.scores) for pred in self.predictions])
        assert y_true.shape == y_probs.shape

        _, _, thresholds = roc_curve(y_true, y_probs)
        f1_scores = []
        for threshold in thresholds:
            f1_scores.append(f1_score(y_true, (y_probs > threshold)))
        return max(f1_scores)

    def auroc(self):
        y_true = np.array([1 if gt.damaged else 0 for gt in self.ground_truth])
        y_probs = np.array([0.0 if len(pred.scores) == 0 else min(pred.scores) for pred in self.predictions])
        assert y_true.shape == y_probs.shape

        return roc_auc_score(y_true, y_probs)

    def plot_confusion_matrix(self, save_path: str = None):
        """
        Plots the confusion matrix of the binary predictions and optionally saves it to a file.
        """
        y_true = [1 if gt.damaged else 0 for gt in self.ground_truth]
        y_pred = [1 if pred.damaged else 0 for pred in self.predictions]

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Undamaged", "Damaged"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.grid(False)

        if save_path:
            plt.savefig(save_path)  # Save the plot to the file
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()  # Show the plot if no save_path is provided

    def save_results_table(self, save_path: str):
        y_true = [1 if gt.damaged else 0 for gt in self.ground_truth]
        y_probs = [0.0 if len(pred.scores) == 0 else min(pred.scores) for pred in self.predictions]
        y_pred = [1 if pred.damaged else 0 for pred in self.predictions]

        results_table = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "y_scores": y_probs})
        results_table.to_csv(save_path, index=True)
        print(f"Results table saved to : {save_path}")

    def make_binary(self, mask):
        return mask > self.threshold

    def IoU(self):
        """
        Computes the Intersection over Union (IoU) score for the binary masks.
        """
        self._check_masks()
        total_iou = 0.0
        count = 0

        for pred, gt in zip(self.predictions, self.ground_truth):
            pred_mask = self.make_binary(pred.mask)
            gt_mask = self.make_binary(gt.mask)
            intersection = (pred_mask & gt_mask).sum()
            union = (pred_mask | gt_mask).sum()

            total_iou += intersection / union if union > 0 else 1.0
            count += 1

        return total_iou / count if count > 0 else 0.0

    def pixel_auroc(self):
        """
        Computes the pixel AUROC score.
        """
        self._check_masks()
        y_true = [gt.mask for gt in self.ground_truth]
        y_pred = [pred.mask for pred in self.predictions]

        # Flatten the masks and compute AUROC
        y_true_flat = [item for sublist in y_true for item in sublist]
        y_pred_flat = [item for sublist in y_pred for item in sublist]

        return roc_auc_score(y_true_flat, y_pred_flat)

    def eval(self):
        """
        Compute all metrics and return as a dictionary.
        """
        results = {}
        self._check_labels()
        try:
            results["accuracy"] = [self.accuracy()]
        except Exception as e:
            print(f"Error in accuracy computation: {e}")

        try:
            results["f1_score"] = [self.f1_score()]
        except Exception as e:
            print(f"Error in F1 Score computation: {e}")

        try:
            results["max_accuracy"] = [self.max_accuracy()]
        except Exception as e:
            print(f"Error in max. accuracy computation: {e}")

        try:
            results["max_f1_score"] = self.max_f1_score()
        except Exception as e:
            print(f"Error in max. F1 Score computation: {e}")

        try:
            results["auroc"] = self.auroc()
        except Exception as e:
            print(f"Error in AUROC computation: {e}")

        try:
            results["pixel_auroc"] = [self.pixel_auroc()]
        except Exception as e:
            print(f"Error in pixel AUROC calculation: {e}")

        try:
            results["IoU"] = [self.IoU()]
        except Exception as e:
            print(f"Error in IoU calculation: {e}")

        return results

    # -----------------------------------------------HELPERS--------------------------------------------------#
    def _check_labels(self) -> None:
        """
        Ensures that all predictions and ground truths have a binary label.
        """
        if any(pred.damaged is None for pred in self.predictions):
            raise ValueError("Prediction label cannot be None")
        if any(gt.damaged is None for gt in self.ground_truth):
            raise ValueError("Ground truth label cannot be None")

    def _check_masks(self) -> None:
        """
        Ensures that all predictions and ground truths have a binary mask.
        """
        if any(pred.mask is None for pred in self.predictions):
            raise ValueError("Prediction mask cannot be None")
        if any(gt.mask is None for gt in self.ground_truth):
            raise ValueError("Ground truth mask cannot be None")
        if any(pred.mask.shape != gt.mask.shape for pred, gt in zip(self.predictions, self.ground_truth)):
            raise ValueError("Prediction and ground truth masks must have the same shape")
        if any(pred.mask.dtype != gt.mask.dtype for pred, gt in zip(self.predictions, self.ground_truth)):
            raise ValueError("Prediction and ground truth masks must have the same dtype")
