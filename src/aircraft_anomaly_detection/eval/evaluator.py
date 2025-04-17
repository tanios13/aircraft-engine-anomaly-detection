import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import List
from ..interfaces import Prediction


class Evaluator:

    def __init__(self, predictions: List[Prediction], ground_truth: list) -> None:
        if len(self.predictions) != len(self.ground_truth):
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
        self._check_binary_labels()
        
        correct = 0
        for i,pred in enumerate(self.predictions):
            if pred.binary_label == self.ground_truth[i].binary_label:
                correct += 1
        return correct / len(self.predictions)


    def f1_score(self):
        """
        Computes the F1 score of the binary predictions.
        """
        self._check_binary_labels()

        tp = 0
        fp = 0
        fn = 0
        for i, pred in enumerate(self.predictions):
            if pred.binary_label == self.ground_truth[i].binary_label:
                tp += 1
            else:
                fp += 1
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1
    

    def plot_confusion_matrix(self):
        """
        Plots the confusion matrix of the binary predictions.
        """
        self._check_binary_labels()

        y_true = [gt.binary_label for gt in self.ground_truth]
        y_pred = [pred.binary_label for pred in self.predictions]

        cm = confusion_matrix(y_true, y_pred, labels=[True, False])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Positive", "Negative"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.grid(False)
        plt.show()


    def auroc(self):
        pass

    def eval(self):
        pass


#-----------------------------------------------HELPERS--------------------------------------------------#
    def _check_binary_labels(self):
        """
        Check that the binary labels are never None
        """
        if any(pred.binary_label is None for pred in self.predictions):
            raise ValueError("Binary labels cannot be None")
        if any(gt.binary_label is None for gt in self.ground_truth):
            raise ValueError("Binary labels cannot be None")
