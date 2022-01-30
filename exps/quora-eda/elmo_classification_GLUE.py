


# WandB parameters
wandb_config_dict = dict(learning_rate=LR,
                         epochs=NUM_EPOCHS,
                         batch_size=BATCH_SIZE,
                         num_examples=NUM_EXAMPLES,
                         train_size=TRAIN_SIZE,
                         test_size=TEST_SIZE,
                         seed=RNDM_SEED,
                         check_duplicate=CHECK_DUP,
                         model_name=MODEL_NAME,
                         max_length=MAX_LENGTH,
                         eval_method=EVAL_METHOD,
                         drop_last=DROP_LAST)

wandb_init_dict = dict(name=TITLE,
                       project='rf-bert',
                       entity='jscuds',
                       tags=['quora','EDA',MODEL_NAME,'cluster'],
                       notes=None,
                       config=wandb_config_dict)




OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
WEIGHT_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


import copy
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Dict, List, Set, Tuple
import warnings

import datasets

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch
from torch import Tensor, nn, optim
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, Dataset

import wandb

import allennlp
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer

# https://github.com/luismsgomes/mosestokenizer
from mosestokenizer import MosesTokenizer

# import nltk
# nltk.download('perluniprops')
# nltk.download('nonbreaking_prefixes')
# from nltk.tokenize.moses import MosesTokenizer

# from allennlp.data.tokenizers.token_class import Token

#js taken from setup of code example at following URL:
#    https://guide.allennlp.org/representing-text-as-features#6
# from allennlp.data import Token, Vocabulary, TokenIndexer, Tokenizer
# from allennlp.data.fields import ListField, TextField
# from allennlp.data.token_indexers import (
#     SingleIdTokenIndexer,
#     TokenCharactersIndexer,
#     ELMoTokenCharactersIndexer,
#     PretrainedTransformerIndexer,
#     PretrainedTransformerMismatchedIndexer,
# )
# from allennlp.data.tokenizers import (
#     CharacterTokenizer,
#     PretrainedTransformerTokenizer,
#     SpacyTokenizer,
#     WhitespaceTokenizer,
# )
# from allennlp.modules.seq2vec_encoders import CnnEncoder
# from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
# from allennlp.modules.token_embedders import (
#     Embedding,
#     TokenCharactersEncoder,
#     ElmoTokenEmbedder,
#     PretrainedTransformerEmbedder,
#     PretrainedTransformerMismatchedEmbedder,
# )
# from allennlp.nn import util as nn_util

warnings.filterwarnings("ignore")

#########################################################
################# DATASET CLASS W/ ELMO #################
#########################################################



