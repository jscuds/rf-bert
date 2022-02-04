from typing import Callable, Dict, Tuple
import abc
import argparse
import functools
import logging

import torch

from torch.utils.data import Dataset

from dataloaders import ParaphraseDatasetElmo, QuoraDataset
from metrics import f1, accuracy, precision, recall
from models import ElmoClassifier
from utils import TensorRunningAverages


logger = logging.getLogger(__name__)

Metric = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

class Experiment(abc.ABC):
    model: torch.nn.Module
    dataset: Dataset
    metrics: Dict[str, Metric]
    metric_averages: TensorRunningAverages

    @abc.abstractmethod
    def __init__(self, args: argparse.Namespace):
        raise NotImplementedError()

    @abc.abstractmethod
    def compute_loss_and_update_metrics(self) -> torch.Tensor:
        """Computes loss for the current experiment. Also updates
        metrics stored in `self.metric_averages`.
        """
        raise NotImplementedError()
    
    def compute_and_reset_metrics(self) -> Dict[str, float]:
        """Computes all metrics that have running averages stored in 
        `self.metric_averages`. Clears the running averages and returns.

        Returns:
            Dict[str, float]: dictionary of running metrics, like
            {
                'train_f1': 0.77133,
                'train_loss': 0.001,
                ...
                'test_loss': 0.002,
                ...
            }
        """
        metrics: Dict[str, float] = {}
        # Compute running averages of metrics and store in a dict
        for name in sorted(self.metric_averages.keys()):
            val = self.metric_averages.get(name)
            metrics[name] = val
            # todo(jxm): use `logging` package here
            logger.info('\t%s = %f', name, val)
        # Clear all metrics and return the averages
        self.metric_averages.clear_all()
        return metrics



class RetrofitExperiment(Experiment):
    """Configures experiments with retrofitting loss."""
    model: ElmoClassifier
    dataset: ParaphraseDatasetElmo # TODO: make this abstract ParaphraseDataset class w/ inheritance for ELMO/BERT
    metrics: Dict[str, Metric]
    metric_averages: TensorRunningAverages

    def __init__(self, args: argparse.Namespace):
        assert args.model_name_or_path == "elmo" # TODO: Support choice of model via argparse.
        self.model = (
            ElmoClassifier(
                num_output_representations = 1, 
                requires_grad=False,
                dropout=0,
                m_transform=True
            )
        )
        # TODO: pass proper args to ParaphaseDataset
        self.dataset = ParaphraseDatasetElmo(
            'quora',
            model_name='bert-base-uncased', num_examples=args.num_examples, 
            max_length=40, stop_words_file=f'stop_words_en.txt',
            r1=0.5, seed=42
        )
        self.metric_averages = TensorRunningAverages()
        # TODO: implement retrofit metrics
        self.metrics = {}


    def compute_loss_and_update_metrics(self, *args, **kwargs) -> torch.Tensor:
        # TODO: add some test that gradients are only applying to the M matrix
        # TODO: Implement retrofitting loss (both parts!)
        raise NotImplementedError()



class FinetuneExperiment(Experiment):
    """Configures experiments for fine-tuning models, typically for
    classification-based tasks like those from the GLUE benchmark.
    """
    model: ElmoClassifier
    dataset: QuoraDataset
    metrics: Dict[str, Metric]
    metric_averages: TensorRunningAverages

    def __init__(self, args: argparse.Namespace):
        assert args.model_name_or_path == "elmo" # TODO: Support choice of model via argparse.
        self.model = (
            ElmoClassifier(
                num_output_representations = 1, 
                requires_grad=True, 
                dropout=0
            )
        )
        self.dataset = QuoraDataset(
            para_dataset='quora', num_examples = args.num_examples,
            max_length=args.max_length, train_test_split=args.train_test_split,
            seed=args.random_seed
        )
        self._loss_fn = torch.nn.BCEWithLogitsLoss()
        self.metric_averages = TensorRunningAverages()
        self.metrics = {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
        }
    
    def compute_loss_and_update_metrics(self, preds: torch.Tensor, targets: torch.Tensor, metrics_key: str) -> torch.Tensor:
        """Computes loss. Updates running metric computations stored in `self.metric_averages`.

        Returns loss.
        """
        assert preds.shape == targets.shape
        loss = self._loss_fn(preds, targets)
        self.metric_averages.update(f'{metrics_key}_loss', loss.item())
        pred_classes = torch.sigmoid(preds.squeeze()).detach() # TODO should we round here for sure?

        for metric_name, metric_func in self.metrics.items():
            val = metric_func(pred_classes, targets)
            self.metric_averages.update(f'{metrics_key}_{metric_name}', val)

        return loss

