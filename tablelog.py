import argparse
import copy
import glob
import json
import logging
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch
import tqdm
import wandb
from torch.utils.data import DataLoader


from utils import elmo_sentence_decode

class TableLog():
    """A class to log W&B tables for specific examples over each epoch of retrofitting."""
    
    def __init__(self, num_table_examples:int, train_dataloader: DataLoader, test_dataloader: DataLoader, columns:List[str]):

        # Sample k examples from the indices in the train/test dataset, respectively
        self.num_table_examples = num_table_examples
        self.table_train_indices: List[int] = random.sample(train_dataloader.sampler.indices, k=self.num_table_examples)
        self.table_test_indices: List[int] = random.sample(test_dataloader.sampler.indices, k=self.num_table_examples)
        self.id_to_sent: Dict[int, Tuple[Tuple[int,...],...]] = train_dataloader.dataset._id_to_sent # dict; lookup sentence given sentence_id
        self.para_tuples: List[Tuple[int,int,int,int]] = train_dataloader.dataset._para_tuples # list of tuples; (s1_id, s2_id, idx1, idx2)
        
        # self.final_table will be what is logged to W&B
        self.final_table = wandb.Table(columns=columns)

        # For all dicts, key is the integer representing the positive example in the paraphrase dataset
        #     ex. the return of quora[10]; 10 is the key for all dictionaries.  This keeps tracking the example
        #         consistent across all epochs with batch shuffling.
        self.batch_indices: Dict[int,int] = {}
        self.table_split: Dict[int,str] = {} # 'train' or 'validation
        self.table_pos_sent1: Dict[int, torch.Tensor] = {}
        self.table_pos_sent2: Dict[int, torch.Tensor] = {}
        self.table_pos_word1: Dict[int, int] = {}
        self.table_pos_word2: Dict[int, int] = {}
        self.table_pos_sent1_str: Dict[int, str] = {}
        self.table_pos_sent2_str: Dict[int, str] = {}
        self.table_neg_sent1_str: Dict[int, str] = {}
        self.table_neg_sent2_str: Dict[int, str] = {}
        self.table_neg_word1: Dict[int, int] = {} # TODO: remove?
        self.table_neg_word2: Dict[int, int] = {} # TODO: remove?
        self.table_pos_target_word: Dict[int, str] = {}
        self.table_neg_target_word: Dict[int, str] = {}
        self.table_pos_word_dist: Dict[int, float] = {}
        self.table_neg_word_dist: Dict[int, float] = {}
        self.table_hinge_loss: Dict[int, float] = {}
    

    def get_tracked_examples(self) -> None:
        """Get information for sampled examples.  Called after dataloaders instantiated and before the training loop."""
        for i in self.table_train_indices: self.table_split[i] = 'train'
        for i in self.table_test_indices: self.table_split[i] = 'validation'
        
        for i in (self.table_train_indices+self.table_test_indices): 

            # Pull sentence/overlap from dataset._para_tuples
            pos_sent1, pos_sent2, token1, token2 = self.para_tuples[i] # (s1_id, s2_id, idx1, idx2)
            pos_sent1, pos_sent2 = torch.tensor(self.id_to_sent[pos_sent1]), torch.tensor(self.id_to_sent[pos_sent2]) # convert sent_id to sentence tensor; shape: [40,50]
            
            pos_sent1_str, pos_sent2_str = elmo_sentence_decode(pos_sent1), elmo_sentence_decode(pos_sent2) # get string representations of sentence

            # populate dictionaries
            self.table_pos_sent1[i] = pos_sent1
            self.table_pos_sent2[i] = pos_sent2
            self.table_pos_sent1_str[i] = pos_sent1_str
            self.table_pos_sent2_str[i] = pos_sent2_str
            self.table_pos_word1[i] = token1
            self.table_pos_word2[i] = token2
            assert pos_sent1_str.split(' ')[token1] == pos_sent2_str.split(' ')[token2] # assert that the word is the same in both pos_sent1 & pos_sent2
            self.table_pos_target_word[i] = pos_sent1_str.split(' ')[token1]

    # TODO: called every time a batch is passed in the train/val loop from the respective dataloader
    def check_tracked_indices(self, pos_sent_1: torch.Tensor, pos_sent_2: torch.Tensor,
                              neg_sent_1: torch.Tensor, neg_sent_2: torch.Tensor,
                              pos_token_1: torch.Tensor, pos_token_2: torch.Tensor,
                              neg_token_1: torch.Tensor, neg_token_2: torch.Tensor) -> Dict[int,int]:
        """Given a retrofit batch, determine if any of the examples are the ones we are sampling for a W&B Table.  """
        
        # Reset self.batch_indices each time it's called
        self.batch_indices = {}
        batch_size = pos_sent_1.shape[0]

        # compare dictionary of sentences we're tracking to see if they're present in a batch.
        for b in range(batch_size):
            for i in self.table_pos_sent1.keys():
                #  if sentence pair matches the paraphrase tuple, add batch_index as a value corresponding to the tracked_index key
                if (torch.equal(self.table_pos_sent1[i], pos_sent_1[b]) and torch.equal(self.table_pos_sent2[i], pos_sent_2[b]) and 
                    self.table_pos_word1[i] == pos_token_1[b].item() and self.table_pos_word2[i] == pos_token_2[b].item()):

                    self.batch_indices[i] = b
                    neg_sent1_str, neg_sent2_str = elmo_sentence_decode(neg_sent_1[b]), elmo_sentence_decode(neg_sent_2[b])
                    self.table_neg_sent1_str[i] = neg_sent1_str
                    self.table_neg_sent2_str[i] = neg_sent2_str
                    self.table_neg_word1[i] = neg_token_1[b].item()
                    self.table_neg_word2[i] = neg_token_2[b].item()
                    assert neg_sent1_str.split(' ')[neg_token_1[b].item()] == neg_sent2_str.split(' ')[neg_token_2[b].item()] # assert that the word is the same in both pos_sent1 & pos_sent2
                    self.table_neg_target_word[i] = neg_sent1_str.split(' ')[neg_token_1[b].item()]

                else: continue

        return self.batch_indices

    def update_wandb_table(self, epoch:int) -> None:
        """Update rows of example table at the end of each epoch."""
        # columns ["index", "epoch", "positive/negative", "sent1", "sent2", "shared_word", "word_distance", "hinge_loss", "split"]
        # columns [index, epoch, sent1, sent2, pos_word, positive, pos_distance, hinge_loss]
        #         [index, epoch, sent1, sent2, neg_word, negative, neg_distance, hinge_loss] 
        
        # Each example will have 2 rows: Row A = Positive, Row B = Negative
        for i in sorted(self.table_split.keys()):
            # Row A
            self.final_table.add_data(
                i, epoch, 'positive', self.table_pos_sent1_str[i], self.table_pos_sent2_str[i],
                self.table_pos_target_word[i], self.table_pos_word_dist[i],
                self.table_hinge_loss[i], self.table_split[i]
            )
            # Row B
            self.final_table.add_data(
                i, epoch, 'negative', self.table_neg_sent1_str[i], self.table_neg_sent2_str[i],
                self.table_neg_target_word[i], self.table_neg_word_dist[i],
                self.table_hinge_loss[i], self.table_split[i]
            )