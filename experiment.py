from typing import Callable, Dict, List, Tuple
import abc
import argparse
import functools
import logging

import torch
from torch.utils.data import DataLoader, Dataset

from dataloaders import ParaphraseDatasetElmo
from dataloaders.helpers import (
    load_rotten_tomatoes, load_qqp, load_sst2, train_test_split
)
from metrics import Metric, Accuracy, PrecisionRecallF1
from models import ElmoClassifier, ElmoRetrofit
from tablelog import TableLog
from utils import TensorRunningAverages, log_wandb_histogram


logger = logging.getLogger(__name__)

Metric = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

class Experiment(abc.ABC):
    model: torch.nn.Module
    dataset: Dataset
    metrics: List[Metric]
    metric_averages: TensorRunningAverages
    wb_table: TableLog

    @abc.abstractmethod
    def __init__(self, args: argparse.Namespace):
        raise NotImplementedError()

    @abc.abstractmethod
    def compute_loss_and_update_metrics(self) -> torch.Tensor:
        """Computes loss for the current experiment. Also updates
        metrics stored in `self.metric_averages`.
        """
        raise NotImplementedError()
    
    def compute_and_reset_metrics(self, step: int, epoch: int) -> Dict[str, float]:
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
        all_metrics_dict: Dict[str, float] = {}
        # Compute running averages of metrics and store in a dict
        for name in sorted(self.metric_averages.keys()):
            val = self.metric_averages.get(name)
            all_metrics_dict[name] = val
            # todo(jxm): use `logging` package here
            logger.info('\t%s = %f', name, val)
        # Clear all metric averages and return the averages
        self.metric_averages.clear_all()

        # Add metrics that can't be computed by simple averaging
        # (accuracy, precision, f1...)
        for metric in self.metrics:
            # Compute metrics like test/accuracy and train/accuracy
            # and reset count.
            metric_dict = metric.compute()
            # Print metrics and add to list of total metrics.
            for name, val in metric_dict.items():
                logger.info('\t%s = %f', name, val)
            all_metrics_dict.update(metric_dict)

        return all_metrics_dict
    
    @abc.abstractmethod
    def get_dataloaders() -> Tuple[DataLoader, DataLoader]:
        """Returns train and test dataloaders."""
        raise NotImplementedError()


class RetrofitExperiment(Experiment):
    """Configures experiments with retrofitting loss."""
    model: ElmoRetrofit
    metrics: List[Metric]
    metric_averages: TensorRunningAverages
    wb_table: TableLog

    def __init__(self, args: argparse.Namespace):
        assert args.model_name == "elmo_single_sentence" # TODO: Support choice of model via argparse.
        self.args = args
        self.model = (
            ElmoRetrofit(
                num_output_representations = 1, 
                requires_grad=args.req_grad_elmo, #default = False --> Frozen
                elmo_dropout=args.elmo_dropout,
            )
        )
        self.rf_lambda = self.args.rf_lambda
        self.rf_gamma = self.args.rf_gamma
        self.metric_averages = TensorRunningAverages()
        self.metrics = []
        self.wb_table = None

        # Lists to store things over batch for histograms
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
        #
        # @jxm - I think we decided to use quora for retrofitting, qqp for GLUE tasks
        train_dataloader, test_dataloader = train_test_split(
            dataset, batch_size=self.args.batch_size, 
            shuffle=True, drop_last=self.args.drop_last, 
            train_split=self.args.train_test_split
        )
        # if we're tracking examples for a table, setup the table configuration
        if self.args.num_table_examples is not None:
            self.wb_table = (
                TableLog(
                    num_table_examples=self.args.num_table_examples,
                    columns=["epoch", "index", "positive/negative", "sent1", "sent2", "shared_word", "word_distance", "hinge_loss", "split"]
                )
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
    def _draw_histograms(self, step: int, epoch: int):
        # Create histograms from stats
        # only plot for training data (lists only populate if model is training)
        log_wandb_histogram(self.pos_dist_list, "Positive Pair Distance", step, epoch)
        log_wandb_histogram(self.neg_dist_list, "Negative Pair Distance", step, epoch)
        log_wandb_histogram(self.diff_dist_list, "Pos - Neg Pair Distance", step, epoch)
        log_wandb_histogram(self.diff_dist_plus_margin_list, "Pos - Neg Pair Dist + Margin", step, epoch)
        # Reset stats
        logger.info('Resetting experiment.pos_dist_list, .neg_dist_list, .diff_dist_list, .diff_dist_plus_margin_list for new histograms')
        self.pos_dist_list = []
        self.neg_dist_list = []
        self.diff_dist_list = [] #pos_dist - neg_dist
        self.diff_dist_plus_margin_list = [] #pos_dist + gamma - neg_dist

    def compute_and_reset_metrics(self, step: int, epoch: int) -> Dict[str, float]:
        # Override this method to also draw histograms of distances.
        self._draw_histograms(step, epoch)
        # Then return the normal result.
        return super().compute_and_reset_metrics(step, epoch)

    def retrofit_hinge_loss(self,
            word_rep_pos_1: torch.Tensor, word_rep_pos_2: torch.Tensor,
            word_rep_neg_1: torch.Tensor, word_rep_neg_2: torch.Tensor,
            gamma: float, epoch: int
        ) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        """L_H = sum_{w} [d_1(M w) - \gamma + d_2(M w)]_+
        
        Where d_1 is the distance between w's representations in a paraphrase pair (hopefully
        will be a small distance), and d_2 is the distance between w's representations in a
        non-paraphrase pair (hopefully will be a relatively large distance).

        And the []_+ operator is max(x, 0).
        """
        assert word_rep_pos_1.shape == word_rep_pos_2.shape
        assert word_rep_neg_1.shape == word_rep_neg_2.shape
        positive_pair_distance = torch.norm(word_rep_pos_1 - word_rep_pos_2, p=2, dim=1)
        negative_pair_distance = torch.norm(word_rep_neg_1 - word_rep_neg_2, p=2, dim=1) # shape: (batch_size,) if keepdim=False

        loss = positive_pair_distance + gamma - negative_pair_distance
        assert loss.shape == (word_rep_pos_1.shape[0],) # ensure dimensions of loss is same as batch size.
        
        # If model is training, create distance lists for creating 4x wandb.plot.histogram() per epoch
        if self.model.training:
            self.pos_dist_list += positive_pair_distance.tolist()
            self.neg_dist_list += negative_pair_distance.tolist() 
            self.diff_dist_list += (positive_pair_distance - negative_pair_distance).tolist()
            self.diff_dist_plus_margin_list += loss.tolist()

        loss_pre_clamp = loss.detach().clone()
        loss = loss.clamp(min=0) # shape: (batch_size,)

        # If a batch hasn't been added from this epoch to wb_table add it for train/test
        if self.wb_table is not None:
            if self.model.training and (epoch not in self.wb_table.train_epochs_sampled):
                split = 'train'
                self._wb_table_add_loss_and_dists(epoch, split, loss, positive_pair_distance, negative_pair_distance)
                self.wb_table.train_epochs_sampled.append(epoch)

            elif not self.model.training and (epoch not in self.wb_table.test_epochs_sampled):
                split = 'validation'
                self._wb_table_add_loss_and_dists(epoch, split, loss, positive_pair_distance, negative_pair_distance)
                self.wb_table.test_epochs_sampled.append(epoch)            

        return loss.mean(), loss_pre_clamp.mean(), positive_pair_distance.mean(), negative_pair_distance.mean()
        
    def _wb_table_add_loss_and_dists(self, epoch: int, split: str, loss: torch.Tensor, 
                                         positive_pair_distance: torch.Tensor, negative_pair_distance: torch.Tensor) -> None:
        """Helper function for capturing individual example losses and distances for a W&B table."""
        for i in range(self.wb_table.num_table_examples):
            key = (epoch,i,split)
            self.wb_table.table_hinge_loss[key] = loss[i]
            self.wb_table.table_pos_word_dist[key] = positive_pair_distance[i]
            self.wb_table.table_neg_word_dist[key] = negative_pair_distance[i]
        
    def orthogonalization_loss(self, M: torch.Tensor) -> torch.Tensor:
        """L_o = ||I - M^T M||"""
        assert len(M.shape) == 2
        assert M.shape[0] == M.shape[1]
        I = torch.eye(M.shape[0], dtype=float).to(M.device) 
        return torch.norm(I - torch.matmul(M.T, M), p='fro')

    def compute_loss_and_update_metrics(self, epoch: int, word_rep_pos_1: torch.Tensor, word_rep_pos_2: torch.Tensor,
                     word_rep_neg_1: torch.Tensor, word_rep_neg_2: torch.Tensor, metrics_key: str) -> torch.Tensor:
        """Retrofitting loss from Retrofitting Contextualized Word Embeddings with Paraphrases https://arxiv.org/pdf/1909.09700.pdf 
    
        Loss is defined in Section 3.2 of the paper. emb_dim for ELMO is 1024.

        L_h: hinge loss between representations of the same word in two different contexts
        L_o: orthogonality constraint on matrix transformation M
        L = L_h + lambda * L_o

        Args:
            word_rep_pos_1 (float torch.Tensor): representation of word 1 in positive pair, shape [B, emb_dim]
            word_rep_pos_2 (float torch.Tensor): representation of word 2 in positive pair, shape [B, emb_dim]
            word_rep_neg_1 (float torch.Tensor): representation of word 1 in negative pair, shape [B, emb_dim]
            word_rep_neg_2 (float torch.Tensor): representation of word 2 in negative pair, shape [B, emb_dim]
            metric_key (str): prefix for metric-logging, like 'Train'
        Returns:
            loss (float torch.Tensor): loss for batch, a scalar
        """
        hinge_loss, pre_clamp_hinge_loss, pos_pair_distance, neg_pair_distance = self.retrofit_hinge_loss(
            word_rep_pos_1, word_rep_pos_2,
            word_rep_neg_1, word_rep_neg_2,
            self.rf_gamma, epoch
        )
        orth_loss = self.orthogonalization_loss(self.M)
        loss = hinge_loss + self.rf_lambda * orth_loss
        # Compute and store losses and related metrics.
        self.metric_averages.update(f'{metrics_key}/Loss', loss.item())
        self.metric_averages.update(f'{metrics_key}/Hinge_Loss', hinge_loss.item())
        self.metric_averages.update(f'{metrics_key}/Pre_Clamp_Hinge_Loss', pre_clamp_hinge_loss.item())
        self.metric_averages.update(f'{metrics_key}/Orthogonalization_Loss', orth_loss.item())
        # Compute and store average distances between word pairs.
        self.metric_averages.update(f'{metrics_key}/pos_pair_dist_mean', pos_pair_distance.item())
        self.metric_averages.update(f'{metrics_key}/neg_pair_dist_mean', neg_pair_distance.item())
        # Compute and store norms of word representations.
        word_rep_pos = torch.cat([word_rep_pos_1, word_rep_pos_2], dim=0)
        word_rep_pos_norm = torch.norm(word_rep_pos, p=2, dim=1).mean()
        self.metric_averages.update(f'{metrics_key}/pos_word_emb_norm', word_rep_pos_norm.item())
        word_rep_neg = torch.cat([word_rep_neg_1, word_rep_neg_2], dim=0)
        word_rep_neg_norm = torch.norm(word_rep_neg, p=2, dim=1).mean()
        self.metric_averages.update(f'{metrics_key}/neg_word_emb_norm', word_rep_neg_norm.item())
        avg_norm = (word_rep_pos_norm + word_rep_neg_norm) / 2.0
        self.metric_averages.update(f'{metrics_key}/avg_word_emb_norm', avg_norm.item())
        return loss



class FinetuneExperiment(Experiment):
    """Configures experiments for fine-tuning models, typically for
    classification-based tasks like those from the GLUE benchmark.
    """
    model: ElmoClassifier
    metrics: List[Metric]
    metric_averages: TensorRunningAverages

    def __init__(self, args: argparse.Namespace):
        assert args.model_name in {"elmo_single_sentence", "elmo_sentence_pair"}
        self.args = args

        # ELMO should never be frozen during finetuning.
        self.model = (
            ElmoClassifier(
                num_output_representations = 1, 
                requires_grad=False, 
                ft_dropout=args.ft_dropout,
                sentence_pair=(args.model_name == "elmo_sentence_pair"),
                m_transform=args.finetune_rf,
                m_transform_requires_grad=False,
                elmo_dropout=args.elmo_dropout
            )
        )
        self._loss_fn = torch.nn.BCELoss()
        self.metric_averages = TensorRunningAverages()
        self.metrics = [Accuracy(), PrecisionRecallF1()]
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        logger.warn('Loading a fine-tuning dataset with a pre-defined test set so ignoring --train_test_split arg if set.')

        if self.args.num_examples:
            logger.warn('--num_examples set so restricting dataset sizes to %d', self.args.num_examples)

        if self.args.dataset_name == 'qqp':
            train_dataloader, test_dataloader = load_qqp(
                max_length=self.args.max_length, batch_size=self.args.batch_size,
                num_examples=self.args.num_examples, drop_last=self.args.drop_last,
                random_seed=self.args.random_seed
            )
        elif self.args.dataset_name == 'rotten_tomatoes':
            train_dataloader, test_dataloader = load_rotten_tomatoes(
                max_length=self.args.max_length, batch_size=self.args.batch_size,
                num_examples=self.args.num_examples, drop_last=self.args.drop_last,
                random_seed=self.args.random_seed
            )
        elif self.args.dataset_name == 'sst2':
            train_dataloader, test_dataloader = load_sst2(
                max_length=self.args.max_length, batch_size=self.args.batch_size,
                num_examples=self.args.num_examples, drop_last=self.args.drop_last,
                random_seed=self.args.random_seed
            )
        else:
            raise ValueError(f'unrecognized fine-tuning dataset {self.args.dataset_name}')
        return train_dataloader, test_dataloader
    
    def compute_loss_and_update_metrics(self, preds: torch.Tensor, targets: torch.Tensor, metrics_key: str) -> torch.Tensor:
        """Computes loss. Updates running metric computations stored in `self.metric_averages`.

        Args:
            preds (torch.Tensor): y_pred to use for loss computation
            targets (torch.Tensor): y_true to use for loss computation.
            metric_key (str): prefix for storing metric, probably looks like 'Test' or 'Train'

        Returns loss as float torch.Tensor of shape ().
        """
        assert preds.shape == targets.shape
        loss = self._loss_fn(preds, targets)
        self.metric_averages.update(f'{metrics_key}/Loss', loss.item())
        for metric in self.metrics:
            metric.update(metrics_key, preds, targets)

        return loss

