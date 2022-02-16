import pytest
import torch

from metrics import Accuracy, PrecisionRecallF1

class TestMetrics:
    def test_acc_half(self):
        accuracy = Accuracy()
        # these preds are both correct
        pred1 = torch.tensor([0.8, 0.9], dtype=float)
        true1 = torch.tensor([1, 1], dtype=float)
        accuracy.update('Train', pred1, true1)
        # these preds are both wrong
        pred2 = torch.tensor([0.2, 0.4], dtype=float)
        true2 = torch.tensor([1, 1], dtype=float)
        accuracy.update('Train', pred2, true2)
        # acc should be 0.5
        metrics = accuracy.compute()
        assert metrics == {
            'Train/Accuracy': 0.5
        }


    def test_acc_multikey(self):
        accuracy = Accuracy()
        # these preds are 25% correct
        pred1 = torch.tensor([0.8, 0.9, 0.7, 0.6], dtype=float)
        true1 = torch.tensor([1, 0, 0, 0,], dtype=float)
        accuracy.update('Train', pred1, true1)
        # these preds are 75% correct
        pred2 = torch.tensor([0.2, 0.4, 0.1, 0.3], dtype=float)
        true2 = torch.tensor([0, 1, 0, 0], dtype=float)
        accuracy.update('Test', pred2, true2)
        # acc should be 0.5 for both keys
        metrics = accuracy.compute()
        assert metrics == {
            'Train/Accuracy': 0.25,
            'Test/Accuracy': 0.75
        }
    
    def test_precision_recall_f1_one_and_zero(self):
        pr_metric = PrecisionRecallF1()
        # everything should be 1s here
        pred1 = torch.tensor([1, 1, 1, 1], dtype=float)
        true1 = torch.tensor([1, 1, 1, 1], dtype=float)
        pr_metric.update('Train', pred1, true1)
        # and these are all wrong
        pred2 = torch.tensor([0, 0, 0, 0], dtype=float)
        true2 = torch.tensor([1, 1, 1, 1], dtype=float)
        pr_metric.update('Eval', pred2, true2)
        
        metrics = pr_metric.compute()
        assert metrics == {
            'Train/Precision': 1.0,
            'Train/Recall': 1.0,
            'Train/F1': 1.0,
            #######################
            'Eval/Precision': 0.0,
            'Eval/Recall': 0.0,
            'Eval/F1': 0.0,
        }

    
    def test_precision_recall_f1(self):
        pr_metric = PrecisionRecallF1()
        # log data in two separate batches (important!)
        # tp, fn, fn
        pred1 = torch.tensor([0.9, 0.1, 0.2,], dtype=float)
        true1 = torch.tensor([1.0, 1.0, 1.0], dtype=float)
        pr_metric.update('Train', pred1, true1)
        # fp, tn, tp
        pred2 = torch.tensor([ 0.8, 0.2, 0.7], dtype=float)
        true2 = torch.tensor([0.0, 0.0, 1.0], dtype=float)
        pr_metric.update('Train', pred2, true2)

        # precision = tp / (tp + fp) = 2 / 3 = 0.666
        # recall = tp / (tp + fn) = 2 / 4 = 0.5
        # f1 = 2 * (p * r) / (p + r) = 2 * (0.333) / (1.1666) = 4 / 7 = 0.2857
        metrics = pr_metric.compute()
        assert metrics == pytest.approx({
            'Train/Precision': 2/3,
            'Train/Recall': 2/4,
            'Train/F1': 4/7,
        })