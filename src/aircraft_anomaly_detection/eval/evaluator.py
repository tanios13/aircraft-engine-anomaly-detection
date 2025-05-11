import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, f1_score
from typing import List
from ..interfaces import Annotation


class Evaluator:

    def __init__(self, predictions: List[Annotation], ground_truth: List[Annotation]) -> None:
        if len(predictions) != len(ground_truth):
            raise ValueError(
                "The length of predictions and ground truth must be the same"
            )
        self.predictions = predictions
        self.ground_truth = ground_truth

    def __call__(self):
        pass


    def accuracy(self):
        """
        Computes the accuracy of the binary predictions.
        """        
        self._check_labels()
        correct = sum(
            1 for pred, gt in zip(self.predictions, self.ground_truth)
            if pred.damaged == gt.damaged
        )
        return correct / len(self.predictions)


    def f1_score(self):
        """
        Computes the F1 score of the binary predictions.
        """
        self._check_labels()
        y_true = [1 if gt.damaged else 0 for gt in self.ground_truth]
        y_pred = [1 if pred.damaged else 0 for pred in self.predictions]
        return f1_score(y_true, y_pred)
    
    def plot_confusion_matrix(self, save_path: str = None):
        """
        Plots the confusion matrix of the binary predictions and optionally saves it to a file.
        """
        self._check_labels()
        y_true = [1 if gt.damaged else 0 for gt in self.ground_truth]
        y_pred = [1 if pred.damaged else 0 for pred in self.predictions]

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Undamaged", "Damaged"]
        )
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.grid(False)
        
        if save_path:
            plt.savefig(save_path)  # Save the plot to the file
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()  # Show the plot if no save_path is provided

    def IoU(self):
        """
        Computes the Intersection over Union (IoU) score for the binary masks.
        """
        self._check_masks()
        intersection = 0
        union = 0

        for pred, gt in zip(self.predictions, self.ground_truth):
            intersection += (pred.mask & gt.mask).sum()
            union += (pred.mask | gt.mask).sum()

        return intersection / union if union > 0 else 0.0

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
        try:
            results["accuracy"] = [self.accuracy()]
            results["f1_score"] = [self.f1_score()]
        except Exception as e:
            print(f"Error in accuracy and f1 evaluation: {e}")
        
        try:
            results["pixel_auroc"] = [self.pixel_auroc()]
        except Exception as e:
            print(f"Error in pixel AUROC calculation: {e}")

        try:
            results["IoU"] = [self.IoU()]
        except Exception as e:
            print(f"Error in IoU calculation: {e}")

        return results


#-----------------------------------------------HELPERS--------------------------------------------------#
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
