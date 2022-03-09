from typing import Dict, List, Tuple
import torch
import wandb


from utils import elmo_sentence_decode

class TableLog:
    """A class to log W&B tables for specific examples over each epoch of retrofitting."""
    
    def __init__(self, num_table_examples: int, columns:List[str]):

        self.num_table_examples: int = num_table_examples
        self.train_epochs_sampled: List[int] = [] # store a list of epochs where a batch was sampled so only 1 batch is sampled per epoch
        self.test_epochs_sampled: List[int] = [] # store a list of epochs where a batch was sampled so only 1 batch is sampled per epoch

        # self.final_table will be what is logged to W&B
        self.final_table = wandb.Table(columns=columns)

        # For all dicts, key is the tuple `(epoch, index, split)``
        #    where `index` is arbitrarily assigned `for i in range(num_table_examples)`
        #    and   `split` is 'train' or 'validation' based on whether the model.training == True/False}
        self.table_pos_sent1_str: Dict[Tuple[int,int,str], str] = {}
        self.table_pos_sent2_str: Dict[Tuple[int,int,str], str] = {}
        self.table_neg_sent1_str: Dict[Tuple[int,int,str], str] = {}
        self.table_neg_sent2_str: Dict[Tuple[int,int,str], str] = {}
        self.table_pos_target_word: Dict[Tuple[int,int,str], str] = {}
        self.table_neg_target_word: Dict[Tuple[int,int,str], str] = {}
        self.table_pos_word_dist: Dict[Tuple[int,int,str], float] = {}
        self.table_neg_word_dist: Dict[Tuple[int,int,str], float] = {}
        self.table_hinge_loss: Dict[Tuple[int,int,str], float] = {}
    

    def get_sample_batch(self, epoch: int, model_training: bool, 
                         pos_sent_1: torch.Tensor, pos_sent_2: torch.Tensor,
                         neg_sent_1: torch.Tensor, neg_sent_2: torch.Tensor,
                         pos_token_1: torch.Tensor, pos_token_2: torch.Tensor,
                         neg_token_1: torch.Tensor, neg_token_2: torch.Tensor) -> None:
        """ Get one batch of data and log `num_table_examples` of the batch in a W&B Table."""

        # Get num_table_examples examples from batch
        # NOTE: previously key of dictionary corresponded to a particular index of the dataset;
        #     After refactoring, the key is just a tuple `(epoch, range(num_table_examples), split)``
        if model_training: 
            split = 'train'
        else:
            split = 'validation'

        for i in range(self.num_table_examples):
            key = (epoch,i,split)
            # Populate positive example dictionaries
            self.table_pos_sent1_str[key], self.table_pos_sent2_str[key] = elmo_sentence_decode(pos_sent_1[i]), elmo_sentence_decode(pos_sent_2[i]) # get string representations of sentences
            assert self.table_pos_sent1_str[key].split(' ')[pos_token_1[i].item()] == self.table_pos_sent2_str[key].split(' ')[pos_token_2[i].item()] # assert that the word is the same in both pos_sent1 & pos_sent2
            self.table_pos_target_word[key] = self.table_pos_sent1_str[key].split(' ')[pos_token_1[i].item()]

            # Populate negative example dictionaries
            self.table_neg_sent1_str[key], self.table_neg_sent2_str[key] = elmo_sentence_decode(neg_sent_1[i]), elmo_sentence_decode(neg_sent_2[i]) # get string representations of sentence 2
            assert self.table_neg_sent1_str[key].split(' ')[neg_token_1[i].item()] == self.table_neg_sent2_str[key].split(' ')[neg_token_2[i].item()] # assert that the word is the same in both pos_sent1 & pos_sent2
            self.table_neg_target_word[key] = self.table_neg_sent1_str[key].split(' ')[neg_token_1[i].item()]


    def update_wandb_table(self) -> None:
        """Update rows of example table at the end of training."""
        # columns ["epoch", "index", "positive/negative", "sent1", "sent2", "shared_word", "word_distance", "hinge_loss", "split"]
        # columns [epoch, index, positive, sent1, sent2, pos_word, pos_distance, hinge_loss, split]
        #         [epoch, index, negative, sent1, sent2, neg_word, neg_distance, hinge_loss, split] 
        
        # Each example will have 2 rows: Row A = Positive, Row B = Negative
        for epoch, i, split in sorted(self.table_pos_sent1_str.keys()):
            key = (epoch, i, split)
            # Row A
            self.final_table.add_data(
                epoch, i, 'positive', self.table_pos_sent1_str[key], self.table_pos_sent2_str[key],
                self.table_pos_target_word[key], self.table_pos_word_dist[key],
                self.table_hinge_loss[key], split
            )
            # Row B
            self.final_table.add_data(
                epoch, i, 'negative', self.table_neg_sent1_str[key], self.table_neg_sent2_str[key],
                self.table_neg_target_word[key], self.table_neg_word_dist[key],
                self.table_hinge_loss[key], split
            )