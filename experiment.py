import abc
import argparse

import torch

from torch.utils.data import Dataset

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


# TODO: Support choice of model via argparse.

class RetrofitExperiment(Experiment):
    def __init__(self, args: argparse.Namespace):
        self.model = 
        self.dataset = (
            QuoraDatasetElmo(para_dataset = 'quora', model_name = MODEL_NAME, 
                     num_examples = NUM_EXAMPLES, max_length = MAX_LENGTH,
                     train_size = TRAIN_SIZE, test_size = TEST_SIZE, seed = RNDM_SEED)
        )


    def compute_loss(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()



class FinetuneExperiment(Experiment):
    def __init__(self):
        self.model = (
            model = ElmoClassifier(options_file = OPTIONS_FILE,
                       weight_file = WEIGHT_FILE, 
                       num_output_representations = 1, 
                       requires_grad=True, 
                       dropout=0)
        )
        self.dataset =
        self._loss_fn = torch.nn.BCEWithLogitsLoss()
    
    def compute_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._loss_fn(preds, targets)
