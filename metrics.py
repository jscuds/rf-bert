import torch

def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """The number of predictions for y_true that are 100% correct.
    """
    y_pred = torch.round(y_pred)
    return (y_true == y_pred).sum() / len(y_pred)

def precision(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Precision: (True Positives) / (True Positives + False Positives)"""
    y_pred = torch.round(y_pred)
    true_positives = torch.sum(torch.logical_and((y_true == 1), (y_pred == 1)))
    false_positives = torch.sum(torch.logical_and((y_true == 0), (y_pred == 1)))
    return true_positives / (true_positives + false_positives)

def recall(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Recall: (True Positives) / (True Positives + False Negatives)"""
    y_pred = torch.round(y_pred)
    true_positives = torch.sum(torch.logical_and((y_true == 1), (y_pred == 1)))
    false_negatives = torch.sum(torch.logical_and((y_true == 1), (y_pred == 0)))
    return true_positives / (true_positives + false_negatives)

def f1(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """F1: 2 * (Precision * Recall) / (Precision + Recall)"""
    p = precision(y_pred, y_true)
    r = recall(y_pred, y_true)
    return 2 * (p * r) / (p + r)