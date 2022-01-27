#########################################################
################### GLOBALS & IMPORTS ###################
#########################################################
import time
day = time.strftime(f'%Y-%m-%d')

NUM_EXAMPLES = 100000 #20000 #100000  404290   #EXP CHANGE 1
EVAL_METHOD = 'base' #base #GLUE
METHOD = f'SIZE_{NUM_EXAMPLES//1000}k_{EVAL_METHOD}' #100k #20k  #full  #50-50  #no-repeat  #filter
TITLE = f'quora-{METHOD}_{day}'

MODEL_NAME = "ELMo-allennlp"
MAX_LENGTH = 40*2 # multiplied by 2 because now passing two paraphrases together

TRAIN_SIZE = 0.8
TEST_SIZE = 0.2
RNDM_SEED = 42
STRAT = False #EXP CHANGE 2
CHECK_DUP = False #EXP CHANGE 3

LR = 1e-4
NUM_EPOCHS = 6 # elmo_change
BATCH_SIZE = 512
DROP_LAST = False
PUNC = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~—“”'


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
                       tags=['quora','EDA',MODEL_NAME, 'cluster'],
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

class QuoraDatasetElmo(Dataset):
    def __init__(self, para_dataset: str = 'quora', model_name: str = 'ELMo-allennlp', 
                 num_examples: int = 20000, max_length: int = 40, 
                 train_size: float = 0.8, test_size: float = 0.2,seed: int = None):
        
        self.para_dataset = para_dataset
        self.model_name = model_name
        self.num_examples = num_examples
        self.max_length = max_length
        self.seed = seed # set to None for random

        self.sentence_id_pair_list = [] # index of this list corresponds to index of tokenized_sentences
        self.tokenized_sentences = [] # lists of [CLS] <s1> [SEP] <s2> [SEP]; where <s1> and <s2> are tokenized sentences
        self.labels = [] # list of labels as ints: 0,1

        if CHECK_DUP:
            self.sent_id_list = []
            self.dup = 0

        #elmo_change
        # https://github.com/luismsgomes/mosestokenizer
        self.tokenizer = MosesTokenizer('en') 
        

        self.bad_words = ["-LSB-", "\\", "``", "-LRB-", "????", "n/a", "'"] #, "//" #js add if you want to filter out some URLs

        # load quora, mrpc, etc.
        print(f'Loading {para_dataset} dataset...\n')
        self._load_dataset()
        self._train_test_split(train_size, test_size, STRAT)
        self.split('train')

    def split(self, train_or_test: str = 'train'):
        self._split = train_or_test
        print(f'{self.para_dataset} split set to {self._split}')

    def __len__(self) -> int:
        if self._split == 'train':
            return len(self.train_labels) 

        if self._split == 'test':
            return len(self.test_labels)
        # return len(self.labels) #changed from ParaDataset

    # based on Shi's gen_batch
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        if self._split == 'train':
            return torch.tensor(self.train_sents[idx].squeeze()), torch.tensor(self.train_labels[idx]) 
            #elmo_change ^^ had to sqeeze() to pass shape [T,50] instead of [1,T,50] for batching

        if self._split == 'test':
            return torch.tensor(self.test_sents[idx].squeeze()), torch.tensor(self.test_labels[idx]) 

        # return torch.tensor(self.tokenized_sentences[idx].ids), torch.tensor(self.labels[idx])
    
    # TODO add MRPC, PAN...
    def _load_dataset(self):
        """
        Loads relevant paraphrase dataset based on argument passed to constructor and creates attributes for model training.
        """

        # load quora from HuggingFace
        if self.para_dataset == 'quora':
            quora = datasets.load_dataset(self.para_dataset)
            
            quora_shuffled = quora.shuffle(seed=self.seed) #js holdover from ParaDataset; ensures if you're using less than total dataset, you are at least shuffling it first.
            count = 0
            if STRAT:
                pos_label_count = 0
                neg_label_count = 0
                print('STRAT==True')

            bad_count = 0              #js added for testing
            sentence_id_pair_list = [] #js added for testing
            label_list = []            #js added for testing

            for pair in quora_shuffled['train']:
                # count += 1 #NOTE:  incrementing here means you don't actually get 20k examples. It's what Shi did.
                if count >= self.num_examples:
                    break

                label = pair['is_duplicate']
                s1_id = pair['questions']['id'][0]
                s2_id = pair['questions']['id'][1]
                s1 = pair['questions']['text'][0]
                s2 = pair['questions']['text'][1]
                
                exist_bad_word = False
                for i in self.bad_words:
                    if (i in s1 or i in s2):
                        exist_bad_word = True

                if exist_bad_word: 
                    bad_count += 1  #js added for testing
                    continue # skip pair if it contains element from self.bad_words

                #js experiment 50-50 split
                if STRAT:
                    if label:
                        pos_label_count +=1
                        if pos_label_count > self.num_examples//2:
                            continue #if half of examples are positive, go to top of loop to find more negative
                    else:
                        neg_label_count +=1
                        if neg_label_count > self.num_examples//2:
                            continue #if half of examples of negative, go to top of loop to find more positive

                #js experiment no duplicate questions
                
                if CHECK_DUP:
                    if s1_id in self.sent_id_list:
                        self.dup += 1
                        continue #if either sentence is in the set, go to top of loop
                    elif s2_id in self.sent_id_list:
                        self.dup += 1
                        continue
                    else:  #otherwise add to sent_id_list
                        self.sent_id_list.append(s1_id) 
                        self.sent_id_list.append(s2_id)

                #### CODE FOR TESTING HOW MANY EXAMPLES ARE POSITIVE AND NEGATIVE ####
                self.sentence_id_pair_list.append((s1_id,s2_id)) #js added
                label_list.append(label)                    #js added
                #### CODE FOR TESTING HOW MANY EXAMPLES ARE POSITIVE AND NEGATIVE ####

                #elmo_change thru `count += 1`
                # tokenize two sentences; add sentence padding tokens <S>,</S>; convert into ELMo character_ids
                # concatentate the sentences, tokenize, and turn into tensor of ELMo character_ids
                concat_sents = ' '.join([s1,s2])
                s1_tokenized = self.tokenizer(s1)
                s2_tokenized = self.tokenizer(s2)
                # self.tokenizer.close()
                concat_sents = ['<S>'] + s1_tokenized + ['</S>'] + ['<S>'] + s2_tokenized + ['</S>']
                if count ==1 : print(concat_sents)
                interim_tokenized = batch_to_ids([concat_sents]) # shape: (1,seq length, 50)
                
                # add padding to self.max_length && truncate if number of tokens > self.max_length
                full_tokenized = torch.zeros(1, self.max_length, 50, dtype=torch.int64) # shape: (one example, self.max_length, 50 for ELMo character encoding)

                full_tokenized[:,:interim_tokenized.shape[1],:] = torch.clone(interim_tokenized[:,:self.max_length,:])
                assert torch.all(torch.eq(full_tokenized[:,:interim_tokenized.shape[1],:], interim_tokenized[:,:self.max_length,:]))

                self.tokenized_sentences.append(full_tokenized)       

                count += 1  # NOTE: count should be incremented here to get num_examples


            #### PRINT STATEMENTS FOR TESTING ####
            #print(f'Total time: {(time.time() - t0):.2f} seconds')
            print('\n#### PRINT STATEMENTS FOR TESTING ####')
            print(f'Total pairs of questions: {len(self.sentence_id_pair_list)}')
            print(f'`count` is {count}, should be {self.num_examples}')
            print(f'`bad_count` is {bad_count}')
            print(f'Number of paraphrases = {sum(label_list)}, {sum(label_list)/len(self.sentence_id_pair_list)*100:.2f}% of total')
            print(f'Number of NON-paraphrases = {len(label_list) - sum(label_list)}, {(len(label_list) - sum(label_list))/len(self.sentence_id_pair_list)*100:.2f}% of total')
            print('#### PRINT STATEMENTS FOR TESTING ####\n')
            #### PRINT STATEMENTS FOR TESTING ####

            # map boolean list of labels to list of 0,1 ints
            self.labels = list(map(int, label_list))
            if STRAT:
              print(f'pos_label_count = {pos_label_count}')
              print(f'neg_label_count = {neg_label_count}')


    def _train_test_split(self, train_size: float =0.8, test_size: float = 0.2, strat=False):
        """
        If called, creates train/test splits of sents/labels.
        """
        if strat:
            self.train_sents, self.test_sents, self.train_labels, self.test_labels = train_test_split(
                self.tokenized_sentences, self.labels, train_size=train_size, test_size=test_size, 
                random_state=self.seed, shuffle = True, stratify = self.labels)

        else: 
            self.train_sents, self.test_sents, self.train_labels, self.test_labels = train_test_split(
                self.tokenized_sentences, self.labels, train_size=train_size, test_size=test_size, 
                random_state=self.seed, shuffle = True, stratify = None
            ) #stratify=self.labels, TODO: use this to get an even distribution of paraphrases and non-paraphrases
        print(f'Training Size: {len(self.train_sents)}')
        print(f'Testing Size: {len(self.test_sents)}')



#########################################################
################## DATASET & DATALOADER #################
#########################################################

quora = QuoraDatasetElmo(para_dataset = 'quora', model_name = MODEL_NAME, 
                     num_examples = NUM_EXAMPLES, max_length = MAX_LENGTH, 
                     train_size = TRAIN_SIZE, test_size = TEST_SIZE, seed = RNDM_SEED)

# js NOTE: HAD TO COPY QUORA BECAUSE CHANGING THE SPLIT CHANGED THE DATALOADERS.
quora.split('train')
train_quora = copy.copy(quora)
train_dataloader = DataLoader(train_quora, batch_size=BATCH_SIZE, shuffle=True, drop_last=DROP_LAST, pin_memory=True)
quora.split('test')
test_quora = copy.copy(quora)
test_dataloader = DataLoader(test_quora, batch_size=BATCH_SIZE, shuffle=True, drop_last=DROP_LAST, pin_memory=True)

print('\n***CHECK DATASET LENGTHS:***')
print(f'len(train_dataloader.dataset) = {len(train_dataloader.dataset)}')
print(f'len(test_dataloader.dataset) = {len(test_dataloader.dataset)}')

#########################################################
################## ElmoClassifier Class #################
#########################################################


class ElmoClassifier(torch.nn.Module):

    def __init__(self, options_file: str, weight_file: str, num_output_representations: int=1, requires_grad: bool=True, dropout: float=0):
        super().__init__()
        self.elmo = Elmo(options_file = options_file, weight_file = weight_file, 
                         num_output_representations = num_output_representations, requires_grad=requires_grad, dropout=dropout)
        self.hidden_size = self.elmo.get_output_dim()
        self.linear = torch.nn.Linear(in_features=self.hidden_size, out_features=1) # shape [1024, 1]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        elmo_out = self.elmo(inputs)  # shape [batch, max_length, 1024] 
        average_elmo = elmo_out['elmo_representations'][0].mean(dim=1) #average across all words / shape [batch, 1024] 
        logits = self.linear(average_elmo) # shape [batch, 1] 

        return logits
    
    #TODO:
    # 1. based on num_output_representations...should we average between those?  Use just the "top" layer?  Is that index 0 or index -1? Right now I'm grabbing elmo_out['elmo_representations'][0]

    # https://github.com/allenai/allennlp/blob/main/allennlp/modules/elmo.py


#########################################################
###################### TRAINING LOOP ####################
#########################################################


# WandB init and config (based on argument dictionaries in imports/globals cell)
wandb.init(**wandb_init_dict)
wandb.config = wandb_config_dict

# def compute_n_correct(preds, targets):
#     return torch.sum((torch.sigmoid(preds.squeeze()).round() == targets)).cpu().item()

# train on gpu if availble, set `device` as global variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


torch.manual_seed(RNDM_SEED)


model = ElmoClassifier(options_file = OPTIONS_FILE,
                       weight_file = WEIGHT_FILE, 
                       num_output_representations = 1, 
                       requires_grad=True, 
                       dropout=0)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = BCEWithLogitsLoss()

train_losses = []
test_losses = []
train_accs = []
test_accs = []
train_f1 = []
test_f1 = []
train_precision = []
test_precision = []
train_recall = []
test_recall = []


t0 = time.time()
for epoch in range(NUM_EPOCHS):
    running_loss = 0
    n_correct = 0
    preds_accum = []
    targets_accum = []

    for batch, targets in tqdm(train_dataloader, leave=False):
    # for batch, targets in train_dl:
        targets_accum.append(targets)
        batch, targets = batch.to(device), targets.to(device)
        preds = model(batch)

        loss = loss_fn(preds.squeeze(), targets.float())

        preds_sigmoid = torch.sigmoid(preds.squeeze())
        preds_accum.append(preds_sigmoid.cpu().detach().numpy().round()) 
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.cpu().item()


    # https://discuss.pytorch.org/t/calculating-f1-score-over-batched-data/83348/2    
    preds_accum = np.concatenate(preds_accum)
    targets_accum = np.concatenate(targets_accum)
    epoch_f1 = f1_score(targets_accum, preds_accum, average='binary')
    epoch_precision = precision_score(targets_accum, preds_accum, average='binary')
    epoch_recall = recall_score(targets_accum, preds_accum, average='binary')
    epoch_accuracy = accuracy_score(targets_accum, preds_accum)


    train_losses.append(running_loss / len(train_dataloader.dataset))
    train_accs.append(epoch_accuracy)
    train_f1.append(epoch_f1)
    train_precision.append(epoch_precision)
    train_recall.append(epoch_recall)

    t1 = time.time()
    print("="*20)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} Train Loss: {train_losses[-1]:>0.5f}")
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} Train Accuracy: {train_accs[-1]*100:>0.2f}%" )
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} Train F1: {train_f1[-1]*100:>0.2f}" )
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} Train Prec: {train_precision[-1]:>0.2f}" )
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} Train Recall: {train_recall[-1]:>0.2f}" )


    running_loss = 0
    n_correct = 0
    preds_accum = []
    targets_accum = []

    with torch.no_grad():
        for batch, targets in tqdm(test_dataloader, leave=False):
        # for batch, targets in test_dl:
            targets_accum.append(targets)
            batch, targets = batch.to(device), targets.to(device)
            preds = model(batch)
            loss = loss_fn(preds.squeeze(), targets.float())
            preds_sigmoid = torch.sigmoid(preds.squeeze())
            preds_accum.append(preds_sigmoid.cpu().round())

            running_loss += loss.cpu().item()


    preds_accum = np.concatenate(preds_accum)
    targets_accum = np.concatenate(targets_accum)
    epoch_f1 = f1_score(targets_accum, preds_accum, average='binary')
    epoch_precision = precision_score(targets_accum, preds_accum, average='binary')
    epoch_recall = recall_score(targets_accum, preds_accum, average='binary')
    epoch_accuracy = accuracy_score(targets_accum, preds_accum)

    test_losses.append(running_loss / len(test_dataloader.dataset))
    test_accs.append(epoch_accuracy)
    test_f1.append(epoch_f1)
    test_precision.append(epoch_precision)
    test_recall.append(epoch_recall)

    t1 = time.time()
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} Test Loss: {test_losses[-1]:>0.5f}")
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} Test Accuracy: {test_accs[-1]*100:>0.2f}%" )
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} Test F1: {test_f1[-1]*100:>0.2f}" )
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} Test Prec: {test_precision[-1]:>0.2f}" )
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} Test Recall: {test_recall[-1]:>0.2f}" )
    
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} Total EPOCH Time: {(t1-t0)/60:>0.2f} min")
    

    # WandB log loss, accuracy, and F1 at end of each epoch
    wandb.log({"Train/Loss": train_losses[-1],
               "Train/Acc": train_accs[-1],
               "Train/F1": train_f1[-1],
               "Train/Recall": train_recall[-1],
               "Train/Precision": train_precision[-1],
               "Test/Loss": test_losses[-1],
               "Test/Acc": test_accs[-1],
               "Test/F1": test_f1[-1],
               "Test/Recall": test_recall[-1],
               "Test/Precision": test_precision[-1],
               "Epoch Time": (t1-t0)/60})

    t0 = time.time()
