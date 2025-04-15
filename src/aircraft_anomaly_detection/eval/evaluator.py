from PIL import Image
from ..interfaces import Prediction
from typing import List


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
        # check that the binary labels are never None
        if any(pred.binary_label is None for pred in self.predictions):
            raise ValueError("Binary labels cannot be None")
        if any(gt.binary_label is None for gt in self.ground_truth):
            raise ValueError("Binary labels cannot be None")
        
        
        correct = 0
        for i,pred in enumerate(self.predictions):
            if pred.binary_label == self.ground_truth[i].binary_label:
                correct += 1
        return correct / len(self.predictions)

    def f1_score(self):
        # check that the binary labels are never None
        if any(pred.binary_label is None for pred in self.predictions):
            raise ValueError("Binary labels cannot be None")
        if any(gt.binary_label is None for gt in self.ground_truth):
            raise ValueError("Binary labels cannot be None")
        

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

    def auroc(self):
        pass

    def eval(self):
        pass
