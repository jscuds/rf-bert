from typing import Dict

import abc
import collections
import torch


@abc.abstractclass
class Metric:
    """A generic metric calculated over multiple batches."""
    @abc.abstractmethod
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """Updates values from a single batch of y_pred and y_true."""
        raise NotImplementedError()
    
    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        """Computes metric based on all the stored values from all the batches.
        
        Returns: Dict that looks like
        {
            'metric_key': metric (float torch.Tensor): the average metric of shape ()
        }
        """
        raise NotImplementedError()


class Accuracy(Metric):
    """The number of predictions for y_true that are 100% correct."""
    num_correct: collections.defaultdict
    total: collections.defaultdict
    total: int
    def __init__(self):
        self.num_correct = collections.defaultdict(lambda: 0)
        self.total = collections.defaultdict(lambda: 0)

    def update(self, key: str, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        assert y_pred.shape == y_true.shape
        y_pred = torch.round(y_pred)
        self.num_correct[key] += (y_true == y_pred).sum()
        self.total[key] += len(y_pred)

    def compute(self) -> Dict[str, torch.Tensor]:
        metrics_dict = {}
        for key in self.num_correct.keys():
            y_pred = torch.round(y_pred)
            metrics_dict[f'{key}/accuracy'] = (y_true == y_pred).sum() / len(y_pred)
        self.num_correct.clear()
        self.total.clear()
        return metrics_dict


class PrecisionRecallF1(Metric):
    """
    Precision: (True Positives) / (True Positives + False Positives)
    Recall: (True Positives) / (True Positives + False Negatives)
    F1: 2 * (Precision * Recall) / (Precision + Recall)
    """
    true_positives: collections.defaultdict
    false_positives: collections.defaultdict
    false_negatives: collections.defaultdict
    def __init__(self):
        self.true_positives = collections.defaultdict(lambda: 0)
        self.false_positives = collections.defaultdict(lambda: 0)
        self.false_negatives = collections.defaultdict(lambda: 0)

    def update(self, key: str, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        assert y_pred.shape == y_true.shape
        y_pred = torch.round(y_pred)
        self.true_positives[key] += torch.sum(torch.logical_and((y_true == 1), (y_pred == 1)))
        self.false_positives[key] += torch.sum(torch.logical_and((y_true == 0), (y_pred == 1)))
        self.false_negatives[key] += torch.sum(torch.logical_and((y_true == 1), (y_pred == 0)))

    def compute(self) -> torch.Tensor:
        metrics_dict = {}
        for key in self.true_positives.keys():
            p = self.true_positives / float(self.true_positives + self.false_positives)
            r = self.true_positives / float(self.true_positives + self.false_negatives)
            f1 = 2 * (p * r) / (p + r)
            metrics_dict[f'{key}/precision'] = p
            metrics_dict[f'{key}/recall'] = r
            metrics_dict[f'{key}/f1'] = f1

        self.true_positives.clear()
        self.false_positives.clear()
        self.false_negatives.clear()

        return metrics_dict