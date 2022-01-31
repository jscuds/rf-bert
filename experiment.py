import abc
import argparse

import torch

from torch.utils.data import Dataset

from dataloaders import ParaphraseDataset, QuoraDataset
from models import ElmoClassifier

class Experiment(abc.ABC):
    model: torch.nn.Module
    dataset: Dataset

    @abc.abstractmethod
    def __init__(self, args: argparse.Namespace):
        raise NotImplementedError()

    @abc.abstractmethod
    def compute_loss(self) -> torch.Tensor:
        """Computes loss for the current experiment."""
        raise NotImplementedError()



class RetrofitExperiment(Experiment):
    """Configures experiments with retrofitting loss."""
    model: ElmoClassifier
    dataset: ParaphraseDataset
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
        self.dataset = ParaphraseDataset(
            'quora',
            model_name='bert-base-uncased', num_examples=20000, 
            max_length=40, stop_words_file=f'stop_words_en.txt',
            r1=0.5, seed=42
        )

    def compute_loss(self, *args, **kwargs) -> torch.Tensor:
        # TODO: add some test that gradients are only applying to the M matrix
        # TODO: Implement retrofitting loss (both parts!)
        raise NotImplementedError()



class FinetuneExperiment(Experiment):
    """Configures experiments for fine-tuning models, typically for
    classification-based tasks like those from the GLUE benchmark.
    """
    model: ElmoClassifier
    dataset: QuoraDataset
    def __init__(self, args: argparse.Namespace):
        assert args.model_name_or_path == "elmo" # TODO: Support choice of model via argparse.
        self.model = (
            ElmoClassifier(
                num_output_representations = 1, 
                requires_grad=True, 
                dropout=0
            )
        )
        self.dataset = (
            QuoraDataset(para_dataset='quora', train_test_split=args.train_test_split)
        )
        self._loss_fn = torch.nn.BCEWithLogitsLoss()
    
    def compute_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._loss_fn(preds, targets)
