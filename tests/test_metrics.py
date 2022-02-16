import pytest
import torch

from metrics import Accuracy, PrecisionRecallF1

class TestMetrics:
    def test_acc_half(self):
        accuracy = Accuracy()
        # these preds are both correct
        pred1 = torch.tensor([0.8, 0.9], dtype=float)
        true1 = torch.tensor([1, 1], dtype=float)
        accuracy.update('train', pred1, true1)
        # these preds are both wrong
        pred2 = torch.tensor([0.2, 0.4], dtype=float)
        true2 = torch.tensor([1, 1], dtype=float)
        accuracy.update('train', pred2, true2)
        # acc should be 0.5
        metrics = accuracy.compute()
        assert metrics == {
            'train/accuracy': 0.5
        }


    def test_acc_multikey(self):
        accuracy = Accuracy()
        # these preds are 25% correct
        pred1 = torch.tensor([0.8, 0.9, 0.7, 0.6], dtype=float)
        true1 = torch.tensor([1, 0, 0, 0,], dtype=float)
        accuracy.update('train', pred1, true1)
        # these preds are 75% correct
        pred2 = torch.tensor([0.2, 0.4, 0.1, 0.3], dtype=float)
        true2 = torch.tensor([0, 1, 0, 0], dtype=float)
        accuracy.update('test', pred2, true2)
        # acc should be 0.5 for both keys
        metrics = accuracy.compute()
        assert metrics == {
            'train/accuracy': 0.25,
            'test/accuracy': 0.75
        }
    
    def test_precision_recall_f1_one_and_zero(self):
        pr_metric = PrecisionRecallF1()
        # everything should be 1s here
        pred1 = torch.tensor([1, 1, 1, 1], dtype=float)
        true1 = torch.tensor([1, 1, 1, 1], dtype=float)
        pr_metric.update('train', pred1, true1)
        # and these are all wrong
        pred2 = torch.tensor([0, 0, 0, 0], dtype=float)
        true2 = torch.tensor([1, 1, 1, 1], dtype=float)
        pr_metric.update('eval', pred2, true2)
        
        metrics = pr_metric.compute()
        assert metrics == {
            'train/precision': 1.0,
            'train/recall': 1.0,
            'train/f1': 1.0,
            #######################
            'eval/precision': 0.0,
            'eval/recall': 0.0,
            'eval/f1': 0.0,
        }

    
    def test_precision_recall_f1(self):
        pr_metric = PrecisionRecallF1()
        # log data in two separate batches (important!)
        # tp, fn, fn
        pred1 = torch.tensor([0.9, 0.1, 0.2,], dtype=float)
        true1 = torch.tensor([1.0, 1.0, 1.0], dtype=float)
        pr_metric.update('train', pred1, true1)
        # fp, tn, tp
        pred2 = torch.tensor([ 0.8, 0.2, 0.7], dtype=float)
        true2 = torch.tensor([0.0, 0.0, 1.0], dtype=float)
        pr_metric.update('train', pred2, true2)

        # precision = tp / (tp + fp) = 2 / 3 = 0.666
        # recall = tp / (tp + fn) = 2 / 4 = 0.5
        # f1 = 2 * (p * r) / (p + r) = 2 * (0.333) / (1.1666) = 4 / 7 = 0.2857
        metrics = pr_metric.compute()
        assert metrics == pytest.approx({
            'train/precision': 2/3,
            'train/recall': 2/4,
            'train/f1': 4/7,
        })