from typing import Callable, Dict, Tuple
import abc
import argparse
import functools
import logging

import torch

from torch.utils.data import Dataset

from dataloaders import ParaphraseDatasetElmo, QuoraDataset
from metrics import f1, accuracy, precision, recall
from models import ElmoClassifier, ElmoRetrofit
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

def retrofit_hinge_loss(
        word_rep_pos_1: torch.Tensor, word_rep_pos_2: torch.Tensor,
        word_rep_neg_1: torch.Tensor, word_rep_neg_2: torch.Tensor,
        gamma: float
    ) -> torch.Tensor:
    """L_H = sum_{w} [d_1(M w) - \gamma + d_2(M w)]_+
    
    Where d_1 is the distance between w's representations in a paraphrase pair (hopefully
    will be a small distance), and d_2 is the distance between w's representations in a
    non-paraphrase pair (hopefully will be a relatively large distance).

    And the []_+ operator is max(x, 0).
    """
    assert word_rep_pos_1.shape == word_rep_pos_2.shape
    assert word_rep_neg_1.shape == word_rep_neg_2.shape
    positive_pair_distance = torch.norm(word_rep_pos_1 - word_rep_pos_2, p=2) # TODO: need keepdim=True??
    negative_pair_distance = torch.norm(word_rep_neg_1 - word_rep_neg_2, p=2)
    loss = positive_pair_distance + gamma - negative_pair_distance
    # hinge loss: return max(0,x)
    return torch.max(
        loss, torch.tensor(0.0, dtype=torch.float32)
    )

def orthogonalization_loss(M: torch.Tensor) -> torch.Tensor:
    """L_o = ||I - M^T M||"""
    assert len(M.shape) == 2
    assert M.shape[0] == M.shape[1]
    I = torch.eye(M.shape[0], dtype=float).to(M.device) 
    return torch.norm(I - torch.matmul(M.T, M), p='fro')

class RetrofitExperiment(Experiment):
    """Configures experiments with retrofitting loss."""
    model: ElmoRetrofit
    dataset: ParaphraseDatasetElmo # TODO: make this abstract ParaphraseDataset class w/ inheritance for ELMO/BERT
    metrics: Dict[str, Metric]
    metric_averages: TensorRunningAverages


    def __init__(self, args: argparse.Namespace):
        assert args.model_name_or_path == "elmo" # TODO: Support choice of model via argparse.
        self.args = args
        self.model = (
            ElmoRetrofit(
                num_output_representations = 1, 
                requires_grad=False,
                dropout=0,
            )
        )
        # TODO: pass proper args to ParaphaseDataset
        self.dataset = ParaphraseDatasetElmo(
            'quora',
            model_name='elmo', num_examples=args.num_examples, 
            max_length=args.max_length, stop_words_file=f'stop_words_en.txt',
            r1=0.5, seed=args.random_seed
        )
        
        self.rf_lambda = self.args.rf_lambda
        self.rf_gamma = self.args.rf_gamma
        self.metric_averages = TensorRunningAverages()
        # TODO: implement retrofit metrics
        self.metrics = {}

    @property
    def M(self) -> torch.nn.Parameter:
        """The orthogonal matrix, M, used for retrofitting."""
        if self.args.model_name_or_path == 'elmo':
            return self.model.elmo._elmo_lstm._elmo_lstm.M
        else:
            raise ValueError(f'cannot get orthogonal matrix for model {self.model_name_or_path}')

    # TODO(jxm): Make compute_loss return dicts so we can log multiple losses independently

    def compute_loss(self, word_rep_pos_1: torch.Tensor, word_rep_pos_2: torch.Tensor,
                     word_rep_neg_1: torch.Tensor, word_rep_neg_2: torch.Tensor) -> torch.Tensor:
        """Retrofitting loss from Retrofitting Contextualized Word Embeddings with Paraphrases https://arxiv.org/pdf/1909.09700.pdf 
    
        Loss is defined in Section 3.2 of the paper.

        L_h: hinge loss between representations of the same word in two different contexts
        L_o: orthogonality constraint on matrix transformation M
        L = L_h + lambda * L_o
        """
        # TODO: how to get representations for each word? --> ANS: ElmoRetrofit.forward() outputs
        hinge_loss = retrofit_hinge_loss(
            word_rep_pos_1, word_rep_pos_2,
            word_rep_neg_1, word_rep_neg_2,
            self.rf_gamma
        )
        orth_loss = orthogonalization_loss(self.M)

        return hinge_loss + self.rf_lambda * orth_loss



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
        self.args = args
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

