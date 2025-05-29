import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve

from aircraft_anomaly_detection.interface.model import ModelInterface
from aircraft_anomaly_detection.schemas.data import Annotation


class Evaluator:
    def __init__(self, predictions: list[Annotation], ground_truth: list[Annotation], **kwargs) -> None:
        if len(predictions) != len(ground_truth):
            raise ValueError("The length of predictions and ground truth must be the same")
        self.predictions = predictions
        self.ground_truth = ground_truth
        self.threshold = kwargs.get("threshold", 0.0)
        self.iou_threshold = kwargs.get("iou_threshold", 0.1)

    def __call__(self):
        pass

    def accuracy_binary(self):
        """
        Computes the accuracy of the binary predictions.
        """
        correct = sum(1 for pred, gt in zip(self.predictions, self.ground_truth) if pred.damaged == gt.damaged)
        return correct / len(self.predictions)

    def f1_score_binary(self):
        """
        Computes the F1 score of the binary predictions.
        """
        y_true = [1 if gt.damaged else 0 for gt in self.ground_truth]
        y_pred = [1 if pred.damaged else 0 for pred in self.predictions]
        return f1_score(y_true, y_pred)

    def bbox_iou(self, bbox1, bbox2):
        x0, y0, x1, y1 = bbox1
        x2, y2, x3, y3 = bbox2

        if min(x1, x3) >= max(x0, x2) and min(y1, y3) >= max(y0, y2):
            intersection = float((min(x1, x3) - max(x0, x2)) * (min(y1, y3) - max(y0, y2)))
        else:
            intersection = 0.0
        area1 = (x1 - x0) * (y1 - y0)
        area2 = (x3 - x2) * (y3 - y2)
        assert area1 > 0.0 and area2 > 0.0 and intersection >= 0.0
        union = area1 + area2 - intersection

        return intersection / union

    def f1_score_localization(self):
        """
        Computes the F1 score of the localization (bounding boxes) predictions.
        """
        unmarked_pred_bbox_idx = []
        for pred_id in range(len(self.predictions)):
            unmarked_pred_bbox_idx += [(pred_id, bbox_id) for bbox_id in range(len(self.predictions[pred_id].bboxes))]
        unmakred_gt_bbox_idx = []
        for gt_id in range(len(self.ground_truth)):
            unmakred_gt_bbox_idx += [(gt_id, bbox_id) for bbox_id in range(len(self.ground_truth[gt_id].bboxes))]

        # sort pred bboxes by score (decreasing)
        bbox_score = lambda id: self.predictions[id[0]].scores[id[1]]
        unmarked_pred_bbox_idx = sorted(unmarked_pred_bbox_idx, key=bbox_score, reverse=True)

        f1_score = None
        recall = None
        precision = None
        max_f1_score = 0.0
        true_pos = 0
        false_pos = 0

        cur_precision = 0.0
        cur_recall = 0.0
        cur_f1_score = 0.0
        for pred_id, pred_bbox_id in unmarked_pred_bbox_idx:
            if self.predictions[pred_id].scores[pred_bbox_id] < self.threshold and f1_score is None:
                f1_score = cur_f1_score
                recall = cur_recall
                precision = cur_precision
            matched = False
            for gt_id, gt_bbox_id in unmakred_gt_bbox_idx:
                bbox_iou = self.bbox_iou(
                    self.predictions[pred_id].bboxes[pred_bbox_id], self.ground_truth[gt_id].bboxes[gt_bbox_id]
                )
                if bbox_iou >= self.iou_threshold:
                    true_pos += 1
                    unmakred_gt_bbox_idx.remove((gt_id, gt_bbox_id))
                    matched = True
                    break
            if not matched:
                false_pos += 1

            # Current F1 score computation
            false_neg = len(unmakred_gt_bbox_idx)
            cur_precision = true_pos / (true_pos + false_pos)
            cur_recall = true_pos / (true_pos + false_neg) if true_pos + false_neg > 0.0 else 0.0
            cur_f1_score = (
                2.0 * (cur_precision * cur_recall) / (cur_precision + cur_recall)
                if cur_precision + cur_recall > 0.0
                else 0.0
            )

            max_f1_score = max(max_f1_score, cur_f1_score)
        if f1_score is None:
            f1_score = cur_f1_score
            precision = cur_precision
            recall = cur_recall
        return f1_score, max_f1_score, precision, recall

    def max_accuracy_binary(self):
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

    def max_f1_score_binary(self):
        y_true = np.array([1 if gt.damaged else 0 for gt in self.ground_truth])
        y_probs = np.array([0.0 if len(pred.scores) == 0 else min(pred.scores) for pred in self.predictions])
        assert y_true.shape == y_probs.shape

        _, _, thresholds = roc_curve(y_true, y_probs)
        f1_scores = []
        for threshold in thresholds:
            f1_scores.append(f1_score(y_true, (y_probs > threshold)))
        return max(f1_scores)

    def auroc_binary(self):
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

    def make_binary(self, mask, threshold=None):
        if threshold is None:
            threshold = self.threshold
        return mask > threshold

    def IoU(self, threshold=None):
        """
        Computes the Intersection over Union (IoU) score for the binary masks.
        """
        self._check_masks()

        if threshold is None:
            threshold = self.threshold

        total_iou = 0.0
        count = 0

        for pred, gt in zip(self.predictions, self.ground_truth):
            pred_mask = self.make_binary(pred.mask, threshold)
            gt_mask = self.make_binary(gt.mask, threshold)
            intersection = (pred_mask & gt_mask).sum()
            union = (pred_mask | gt_mask).sum()

            total_iou += intersection / union if union > 0 else 0.0
            count += 1 if union > 0 else 0.0

        return total_iou / count if count > 0 else 0.0

    def max_IoU(self):
        """
        Computes the maximum Intersection over Union (IoU) score for the binary masks.
        """
        self._check_masks()

        y_true_flat_masks = [gt.mask.flatten() for gt in self.ground_truth]
        y_pred_flat_masks = [pred.mask.flatten() for pred in self.predictions]

        # Flatten the masks and compute AUROC
        y_true_flat = np.concatenate(y_true_flat_masks)
        y_pred_flat = np.concatenate(y_pred_flat_masks)

        all_scores = np.unique(np.concatenate(y_true_flat, y_pred_flat))
        max_iou = 0.0
        for score in all_scores:
            max_iou = max(max_iou, self.IoU(threshold=score))
        return max_iou

    def pixel_auroc(self):
        """
        Computes the pixel AUROC score.
        """
        self._check_masks()
        y_true_flat_masks = [gt.mask.flatten() for gt in self.ground_truth]
        y_pred_flat_masks = [pred.mask.flatten() for pred in self.predictions]

        # Flatten the masks and compute AUROC
        y_true_flat = np.concatenate(y_true_flat_masks)
        y_pred_flat = np.concatenate(y_pred_flat_masks)

        return roc_auc_score(y_true_flat, y_pred_flat)

    def eval(self):
        """
        Compute all metrics and return as a dictionary.
        """
        results = {}
        self._check_labels()
        try:
            results["binary:accuracy"] = [self.accuracy_binary()]
        except Exception as e:
            print(f"Error in accuracy (binary) computation: {e}")

        try:
            results["binary:f1_score"] = [self.f1_score_binary()]
        except Exception as e:
            print(f"Error in F1 Score (binary) computation: {e}")

        try:
            results["binary:max_accuracy"] = [self.max_accuracy_binary()]
        except Exception as e:
            print(f"Error in max. accuracy (binary) computation: {e}")

        try:
            results["binary:max_f1_score"] = self.max_f1_score_binary()
        except Exception as e:
            print(f"Error in max. F1 Score (binary) computation: {e}")

        try:
            results["binary:auroc"] = [self.auroc_binary()]
        except Exception as e:
            print(f"Error in AUROC (binary) computation: {e}")

        try:
            results["pixel_auroc"] = [self.pixel_auroc()]
        except Exception as e:
            print(f"Error in pixel AUROC calculation: {e}")

        try:
            results["IoU"] = [self.IoU()]
        except Exception as e:
            print(f"Error in IoU calculation: {e}")

        try:
            results["max_IoU"] = [self.IoU()]
        except Exception:
            print("Error in Max-IoU calculation: {e}")

        try:
            results["localizaiton:f1_score"]
        except Exception:
            localization_results = self.f1_score_localization()
            results["localization:f1_score"] = [localization_results[0]]
            results["localization:max_f1_score"] = [localization_results[1]]
            results["localization:precision"] = [localization_results[2]]
            results["localization:recall"] = [localization_results[3]]

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
