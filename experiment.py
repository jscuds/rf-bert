from typing import Callable, Dict, Tuple
import abc
import argparse
import functools
import logging

import torch
from torch.utils.data import DataLoader, Dataset

from dataloaders import ParaphraseDatasetElmo, QuoraDataset
from metrics import f1, accuracy, precision, recall
from models import ElmoClassifier, ElmoRetrofit
from utils import TensorRunningAverages, log_wandb_histogram
from dataloaders.helpers import load_rotten_tomatoes, train_test_split


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
    
    def compute_and_reset_metrics(self, epoch: int) -> Dict[str, float]:
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
    
    @abc.abstractmethod
    def get_dataloaders() -> Tuple[DataLoader, DataLoader]:
        """Returns train and test dataloaders."""
        raise NotImplementedError()


class RetrofitExperiment(Experiment):
    """Configures experiments with retrofitting loss."""
    model: ElmoRetrofit
    metrics: Dict[str, Metric]
    metric_averages: TensorRunningAverages


    def __init__(self, args: argparse.Namespace):
        assert args.model_name == "elmo_single_sentence" # TODO: Support choice of model via argparse.
        self.args = args
        self.model = (
            ElmoRetrofit(
                num_output_representations = 1, 
                requires_grad=args.req_grad_elmo, #default = False --> Frozen
                dropout=0,
            )
        )
        self.rf_lambda = self.args.rf_lambda
        self.rf_gamma = self.args.rf_gamma
        self.metric_averages = TensorRunningAverages()
        # TODO: implement retrofit metrics
        #   --> I think loss is all we need for retrofitting -js
        self.metrics = {}

        # NOTE(js): added lists to hold every individual pos_dist and neg_dist for wandb histogram
        self.pos_dist_list = []
        self.neg_dist_list = []
        self.diff_dist_list = [] #pos_dist - neg_dist
        self.diff_dist_plus_margin_list = [] #pos_dist + gamma - neg_dist
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        # TODO: pass proper args to ParaphaseDataset
        dataset = ParaphraseDatasetElmo(
            'quora',
            model_name='elmo', num_examples=self.args.num_examples, 
            max_length=self.args.max_length, stop_words_file=f'stop_words_en.txt',
            r1=0.5, seed=self.args.random_seed
        )
        # Quora doesn't have a test split, so we have to do this?
        # @js - is this right? Otherwise we should be using the actual
        # test data from quora
        train_dataloader, test_dataloader = train_test_split(
            dataset, batch_size=self.args.batch_size, 
            shuffle=True, drop_last=self.args.drop_last, 
            train_split=self.args.train_test_split
        )
        return train_dataloader, test_dataloader

    @property
    def M(self) -> torch.nn.Parameter:
        """The orthogonal matrix, M, used for retrofitting."""
        if self.args.model_name == 'elmo_single_sentence':
            return self.model.elmo._elmo_lstm._elmo_lstm.M
        else:
            raise ValueError(f'cannot get orthogonal matrix for model {self.args.model_name}')

    # TODO(jxm): Make compute_loss return dicts so we can log multiple losses independently
    def _draw_histograms(self, epoch: int):
        # Create histograms from stats
        # only plot for training data (lists only populate if model is training)
        log_wandb_histogram(self.pos_dist_list, "Positive Pair Distance", epoch)
        log_wandb_histogram(self.neg_dist_list, "Negative Pair Distance", epoch)
        log_wandb_histogram(self.diff_dist_list, "Positive minus Negative Pair Distance", epoch)
        log_wandb_histogram(self.diff_dist_plus_margin_list, "Positive minus Negative Pair Dist plus Margin", epoch)
        # Reset stats
        logger.info('Resetting experiment.pos_dist_list, .neg_dist_list, .diff_dist_list, .diff_dist_plus_margin_list for new histograms')
        self.pos_dist_list = []
        self.neg_dist_list = []
        self.diff_dist_list = [] #pos_dist - neg_dist
        self.diff_dist_plus_margin_list = [] #pos_dist + gamma - neg_dist

    def compute_and_reset_metrics(self, epoch: int) -> Dict[str, float]:
        # Override this method to also draw histograms of distances.
        self._draw_histograms(epoch)
        # Then return the normal result.
        return super().compute_and_reset_metrics(epoch)

    def retrofit_hinge_loss(self,
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
        positive_pair_distance = torch.norm(word_rep_pos_1 - word_rep_pos_2, p=2, dim=1) # TODO: need keepdim=True??
        negative_pair_distance = torch.norm(word_rep_neg_1 - word_rep_neg_2, p=2, dim=1) # shape: (batch_size,) if keepdim=False

        loss = positive_pair_distance + gamma - negative_pair_distance
        assert loss.shape == (word_rep_pos_1.shape[0],) # ensure dimensions of loss is same as batch size.
        
        # if model is training, create distance lists for creating 4x wandb.plot.histogram() per epoch
        if self.model.training:
            self.pos_dist_list += positive_pair_distance.tolist()
            self.neg_dist_list += negative_pair_distance.tolist() 
            self.diff_dist_list += (positive_pair_distance - negative_pair_distance).tolist()
            self.diff_dist_plus_margin_list += loss.tolist()

        loss_pre_clamp = loss.detach().clone()
        loss = loss.clamp(min=0) # shape: (batch_size,)
        return loss.mean(), loss_pre_clamp.mean(), positive_pair_distance.mean(), negative_pair_distance.mean()
        
        
    def orthogonalization_loss(self, M: torch.Tensor) -> torch.Tensor:
        """L_o = ||I - M^T M||"""
        assert len(M.shape) == 2
        assert M.shape[0] == M.shape[1]
        I = torch.eye(M.shape[0], dtype=float).to(M.device) 
        return torch.norm(I - torch.matmul(M.T, M), p='fro')

    def compute_loss_and_update_metrics(self, word_rep_pos_1: torch.Tensor, word_rep_pos_2: torch.Tensor,
                     word_rep_neg_1: torch.Tensor, word_rep_neg_2: torch.Tensor, metrics_key: str) -> torch.Tensor:
        """Retrofitting loss from Retrofitting Contextualized Word Embeddings with Paraphrases https://arxiv.org/pdf/1909.09700.pdf 
    
        Loss is defined in Section 3.2 of the paper.

        L_h: hinge loss between representations of the same word in two different contexts
        L_o: orthogonality constraint on matrix transformation M
        L = L_h + lambda * L_o
        """
        # TODO: how to get representations for each word? --> ANS(js): ElmoRetrofit.forward() outputs
        hinge_loss, pre_clamp_hinge_loss, pos_pair_distance, neg_pair_distance = self.retrofit_hinge_loss(
            word_rep_pos_1, word_rep_pos_2,
            word_rep_neg_1, word_rep_neg_2,
            self.rf_gamma
        )
        orth_loss = self.orthogonalization_loss(self.M)
        loss = hinge_loss + self.rf_lambda * orth_loss
        self.metric_averages.update(f'{metrics_key}/Loss', loss.item())
        # NOTE(js): logging 3 additional metrics to troubleshoot loss
        self.metric_averages.update(f'{metrics_key}/Hinge_Loss', hinge_loss.item())
        self.metric_averages.update(f'{metrics_key}/Pre_Clamp_Hinge_Loss', pre_clamp_hinge_loss.item())
        self.metric_averages.update(f'{metrics_key}/Orthogonalization_Loss', orth_loss.item())
        self.metric_averages.update(f'{metrics_key}/pos_pair_dist_mean', pos_pair_distance.item())
        self.metric_averages.update(f'{metrics_key}/neg_pair_dist_mean', neg_pair_distance.item())
        return loss



class FinetuneExperiment(Experiment):
    """Configures experiments for fine-tuning models, typically for
    classification-based tasks like those from the GLUE benchmark.
    """
    model: ElmoClassifier
    dataset: QuoraDataset
    metrics: Dict[str, Metric]
    metric_averages: TensorRunningAverages

    def __init__(self, args: argparse.Namespace):
        assert args.model_name in {"elmo_single_sentence", "elmo_sentence_pair"} # TODO: Support choice of model via argparse.
        self.args = args
        self.model = (
            ElmoClassifier(
                num_output_representations = 1, 
                requires_grad=True, 
                dropout=0,
                sentence_pair=(args.model_name == "elmo_sentence_pair"),
                m_transform=args.finetune_rf,
            )
        )
        self._loss_fn = torch.nn.BCEWithLogitsLoss()
        self.metric_averages = TensorRunningAverages()
        self.metrics = {
            'F1': f1,
            'Precision': precision,
            'Recall': recall,
            'Acc': accuracy,
        }
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        if self.args.dataset_name == 'quora':
            dataset = QuoraDataset(
                para_dataset='quora', num_examples=self.args.num_examples,
                max_length=self.args.max_length
            )
            # Quora doesn't have a test split, so we have to do this?
            # @js - is this right? Otherwise we should be using the actual
            # test data from quora
            train_dataloader, test_dataloader = train_test_split(
                dataset, batch_size=self.args.batch_size, 
                shuffle=True, drop_last=self.args.drop_last, 
                train_split=self.args.train_test_split
            )
        elif self.args.dataset_name == 'rotten_tomatoes':
            train_dataloader, test_dataloader = load_rotten_tomatoes(
                max_length=self.args.max_length, batch_size=self.args.batch_size,
                num_examples=self.args.num_examples, drop_last=self.args.drop_last
            )
        else:
            raise ValueError(f'unrecognized fine-tuning dataset {self.args.dataset_name}')
        return train_dataloader, test_dataloader
    
    def compute_loss_and_update_metrics(self, preds: torch.Tensor, targets: torch.Tensor, metrics_key: str) -> torch.Tensor:
        """Computes loss. Updates running metric computations stored in `self.metric_averages`.

        Returns loss.
        """
        assert preds.shape == targets.shape
        loss = self._loss_fn(preds, targets)
        self.metric_averages.update(f'{metrics_key}/Loss', loss.item())
        pred_classes = torch.sigmoid(preds.squeeze()).detach() # TODO should we round here for sure?

        for metric_name, metric_func in self.metrics.items():
            val = metric_func(pred_classes, targets)
            self.metric_averages.update(f'{metrics_key}/{metric_name}', val)

        return loss

