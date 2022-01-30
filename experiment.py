import abc
import argparse

import torch

from torch.utils.data import Dataset

from .datasets import ParaphraseDataset, QuoraDataset
from .models import ElmoClassifier, ElmoClassifierWithMatrixLayer

class Experiment(abc.ABC):
    @abc.abstractmethod
    def __init__(self, args: argparse.Namespace):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def model(self) -> torch.nn.Module:
        """The model for the current experiment."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def dataset(self) -> Dataset:
        """The dataset for the current experiment."""
        raise NotImplementedError()

    @abc.abstractmethod
    def compute_loss(self) -> torch.Tensor:
        """Computes loss for the current experiment."""
        raise NotImplementedError()



class RetrofitExperiment(Experiment):
    """Configures experiments with retrofitting loss."""
    def __init__(self, args: argparse.Namespace):
        assert args.model_name_or_path == "elmo" # TODO: Support choice of model via argparse.
        self.model = (
            ElmoClassifierWithMatrixLayer(
                num_output_representations = 1, 
                requires_grad=True, 
                dropout=0
            )
        )
        # TODO: pass proper args to ParaphaseDataset
        self.dataset = ParaphraseDataset(
            'quora',
            model_name='bert-base-uncased', num_examples=20000, 
            max_length=40, stop_words_file=f'{cwd}/stop_words_en.txt',
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
            QuoraDataset(para_dataset = 'quora', model_name = MODEL_NAME, 
                     num_examples = NUM_EXAMPLES, max_length = MAX_LENGTH,
                     train_size = TRAIN_SIZE, test_size = TEST_SIZE, seed = RNDM_SEED)
        )
        self._loss_fn = torch.nn.BCEWithLogitsLoss()
    
    def compute_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._loss_fn(preds, targets)
