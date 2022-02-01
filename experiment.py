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

def retrofit_hinge_loss(
        word_rep_pos_1: torch.Tensor, word_rep_pos_2: torch.Tensor,
        word_rep_neg_1: torch.Tensor, word_rep_neg_2: torch.Tensor,
        gamma: float
    ) -> torch.Tensor:
    """L_H = \sum_{w} [d_1(M w) - \gamma + d_2(M w)]_+
    
    Where d_1 is the distance between w's representations in a paraphrase pair (hopefully
    will be a small distance), and d_2 is the distance between w's representations in a
    non-paraphrase pair (hopefully will be a relatively large distance).

    And the []_+ operator is max(x, 0).
    """
    positive_pair_distance = torch.norm(word_rep_pos_1 - word_rep_pos_2, p=2)
    negative_pair_distance = torch.norm(word_rep_neg_1 - word_rep_neg_2, p=2)
    loss = positive_pair_distance + gamma - negative_pair_distance
    # hinge loss: return max(0,x)
    return torch.max(
        loss, torch.tensor(0.0, dtype=torch.float32)
    )

def orthogonalization_loss(M: torch.Tensor) -> torch.Tensor:
    """L_o = ||I - M^T M||"""
    I = torch.eye(M.size).to(M.device) 
    return torch.norm(I - torch.matmul(M.T, M), p='fro')

class RetrofitExperiment(Experiment):
    """Configures experiments with retrofitting loss."""
    model: ElmoClassifier
    dataset: ParaphraseDataset
    args: argparse.Namespace
    rf_gamma: float
    rf_lambda: float
    def __init__(self, args: argparse.Namespace):
        assert args.model_name_or_path == "elmo" # TODO: Support choice of model via argparse.
        self.args = args
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
            model_name='bert-base-uncased', num_examples=args.num_examples, 
            max_length=40, stop_words_file=f'stop_words_en.txt',
            r1=0.5, seed=42
        )
        self.rf_lambda = self.args.rf_lambda
        self.rf_gamma = self.args.rf_gamma

    @property
    def M(self) -> torch.nn.Parameter:
        """The orthogonal matrix, M, used for retrofitting."""
        if self.args.model_name_or_path == 'elmo':
            return self.model.elmo._elmo_lstm._elmo_lstm.M
        else:
            raise ValueError(f'cannot get orthogonal matrix for model {self.model_name_or_path}')

    # TODO(jxm): Make compute_loss return dicts so we can log multiple losses independently

    def compute_loss(self, *args, **kwargs) -> torch.Tensor:
        """Retrofitting loss from Retrofitting Contextualized Word Embeddings with Paraphrases https://arxiv.org/pdf/1909.09700.pdf 
    
        Loss is defined in Section 3.2 of the paper.

        L_h: hinge loss between representations of the same word in two different contexts
        L_o: orthogonality constraint on matrix transformation M
        L = L_h + lambda * L_o
        """
        # TODO: how to get representations for each word?
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
    args: argparse.Namespace
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
    
    def compute_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._loss_fn(preds, targets)
