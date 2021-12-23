# Globals and Imports

LOCAL = False
STOP_WORDS_LOC = 'stop_words_en.txt' #'NLTK_stop_words_en.txt'

# MODEL_NAME = "bert-base-uncased"
# MAX_LENGTH = 40
PUNC = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~—“”'
MODEL_CACHE_DIR = "/mnt/d/huggingface/transformers"
DATASET_CACHE_DIR = "/mnt/d/huggingface/datasets"

import pickle
import random
import re
import time
from typing import Dict, List, Set, Tuple

import datasets
from transformers import AutoTokenizer

import numpy as np
#import pandas as pd
import torch
from torch import Tensor, nn, optim
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)
from torch.utils.data import DataLoader, Dataset

# TODO future: ParaDataset to work for all datasets?
class ParaDataset(Dataset):
    def __init__(self, para_dataset: str = 'quora', model_name: str = 'bert-base-uncased', 
                 num_examples: int = 20000, max_length: int = 40, stop_words_file: str = 'stop_words_en.txt',
                 r1: float=0.5, seed: int = None):
        
        self.para_dataset = para_dataset
        self.model_name = model_name
        self.num_examples = num_examples
        self.max_length = max_length
        self.r1 = r1 # negative sample ratio
        self.seed = seed # set to None for random

        # TODO do these only apply to 'quora'? if so --> move into self._load_dataset
        # TODO combine self._token_pair_to_neg_tuples and self._neg_tuples because the former references the latter?

        self._id_to_sent: Dict[int, Tensor] = {} # maps question/sentence IDs to token tensors
        self._sent_to_id: Dict[Tuple[int,...], int] = {} # reverse lookup: token tensors (cast to ) to question/sentence IDs
        self._paraphrase_sets: Set[Tuple[int,int]] = set([]) # a set of {(sent1_id, sent2_id), (sent2_id, sent1_id) ...} to check whether two sentences are paraphrases or not.
        self._para_tuples: List[Tuple[int,int,int,int]] = [] # [(sent1_id, sent2_id, sent1_index_of_an_overlapping/synonym_token, sent2_index_of_an_overlapping/synonym_token), ... ]
        self._neg_tuples:  List[Tuple[int,int,int,int]] = [] # [(sent1_id, sent2_id, sent1_index_of_an_overlapping/synonym_token, sent2_index_of_an_overlapping/synonym_token), ... ]
        self._token_pair_to_neg_tuples: Dict[Tuple[int,int],Set[int]] = {} # maps pair of word_token_ids to their index position in self._neg_tuples; {(token_id, token_id) : set([idx of self._neg_tuples, ...])}
        self._token_to_sents: Dict[int,Set[Tuple[int,int]]] = {} # reverse index of sentences given tokens. This is a map {token_id : set([(sent_id, index_of_the_token_in_the_sentence), ...]) }
        
        if LOCAL: self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = MODEL_CACHE_DIR)
        else:     self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # load stop words from file
        self.bad_words = ["-LSB-", "\\", "``", "-LRB-", "????", "n/a", "'"] #, "//" #js add if you want to filter out some URLs
        self.stop_words_set = set([])
        print(f'Generating stop words from {stop_words_file}...\n')
        self._gen_stop_words(stop_words_file)

        # load quora, mrpc, etc.
        print(f'Loading {para_dataset} dataset...\n')
        self._load_dataset()

    def __len__(self) -> int:
        return len(self._para_tuples)

    # based on Shi's gen_batch
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, int, int, int, int]:
        
        neg_tuple = self._gen_neg_tuple(self._para_tuples[idx])
        random.seed(self.seed)
        if (random.uniform(0,1) > self.r1) or (neg_tuple is None):
            neg_tuple = self._corrupt(self._para_tuples[idx])
            

        sent1 = self._id_to_sent[ self._para_tuples[idx][0] ] # tensor of token_ids
        sent2 = self._id_to_sent[ self._para_tuples[idx][1] ] 
        nsent1 = self._id_to_sent[ neg_tuple[0] ] 
        nsent2 = self._id_to_sent[ neg_tuple[1] ] 
        token1 = self._para_tuples[idx][2] # token_id for a particular word
        token2 = self._para_tuples[idx][3]
        ntoken1 = neg_tuple[2]
        ntoken2 = neg_tuple[3]
        return torch.tensor(sent1), torch.tensor(sent2), torch.tensor(nsent1), torch.tensor(nsent2), token1, token2, ntoken1, ntoken2
    
    # TODO add MRPC, PAN...
    def _load_dataset(self):
        """
        Loads relevant paraphrase dataset based on argument passed to constructor and creates attributes for model training.
        """

        # load quora from HuggingFace
        if self.para_dataset == 'quora':
            if LOCAL: quora = datasets.load_dataset(self.para_dataset, cache_dir = MODEL_CACHE_DIR)
            else:     quora = datasets.load_dataset(self.para_dataset)
            
            quora_shuffled = quora.shuffle(seed=self.seed)
            count = 0

            bad_count = 0              #js added for testing
            sentence_id_pair_list = [] #js added for testing
            label_list = []            #js added for testing

            # TODO: Tokenize things beforehand!
            for pair in quora_shuffled['train']:
                #count += 1 #TODO incrementing here means you don't actually get 20k examples. It's what Shi did.
                if count >= self.num_examples:
                    break
                label = pair['is_duplicate']
                s1_id = pair['questions']['id'][0]
                s2_id = pair['questions']['id'][1]
                s1 = pair['questions']['text'][0]
                s2 = pair['questions']['text'][1]
                
                # TODO: better way? .map() functionality of quora dataset object? map bad words to [UNK]?
                exist_bad_word = False
                for i in self.bad_words:
                    if (i in s1 or i in s2):
                        exist_bad_word = True

                if exist_bad_word: 
                    bad_count += 1  #js added for testing
                    continue # skip pair if it contains element from self.bad_words

                #### CODE FOR TESTING HOW MANY EXAMPLES ARE POSITIVE AND NEGATIVE ####
                sentence_id_pair_list.append((s1_id,s2_id)) #js added
                label_list.append(label)                    #js added
                #### CODE FOR TESTING HOW MANY EXAMPLES ARE POSITIVE AND NEGATIVE ####

                # use (1) `.ids` for ints or (2) `.tokens` for wordpiece strings
                s1_tokenized = self.tokenizer(s1,truncation=True, max_length = self.max_length, padding= 'max_length')[0] 
                s2_tokenized = self.tokenizer(s2,truncation=True, max_length = self.max_length, padding= 'max_length')[0]

                #Shi uses a list for tokenized sentences; where the index is equivalent to the updated id; 
                #   js: I'm going to just use a dictionary
                
                #if not s1_tokenized in sent2id: shouldn't be necessary due to id2sent.update() functionality
                self._id_to_sent.update({s1_id : s1_tokenized.ids})
                self._id_to_sent.update({s2_id : s2_tokenized.ids})
                self._sent_to_id.update({tuple(s1_tokenized.ids) : s1_id}) #js NOTE have to use Tuples because lists aren't hashable
                self._sent_to_id.update({tuple(s2_tokenized.ids) : s2_id})

                total_index_pairs = self._overlap(s1_tokenized.ids, s2_tokenized.ids)

                #js if `is_duplicate`== True; append both combinations s1&s2 ++ s2&s1 to `self.paraphrases`
                if label:
                    self._paraphrase_sets.add((s1_id, s2_id))
                    self._paraphrase_sets.add((s2_id, s1_id))
                    for p in total_index_pairs:             #js for every pair of overlap words:
                        sent_tuple = (s1_id,s2_id,p[0],p[1]) #js (s1_idx, s2_idx, s1_overlap_word_index, s2_overlap_word_idx) **SO THERE COULD BE MULTIPLE TUPLES STARTING WITH  `s1_idx, s2_idx`** if they have multiple overlapping words
                        self._para_tuples.append(sent_tuple) #js a list of tuples in the format from the line above
                else: #js this from NON-PARAPHRASE questions
                    for p in total_index_pairs:
                        sent_tuple = (s1_id, s2_id, p[0], p[1])
                        self._neg_tuples.append(sent_tuple)
                        w1 = s1_tokenized.ids[p[0]] #js I think w1 == w2; except when synonyms are involved
                        w2 = s2_tokenized.ids[p[1]]
                        if w1 in self.stop_words_set or w2 in self.stop_words_set: #js filter out stopwords from negative sentences
                            continue

                        self._token_pair_to_neg_tuples.setdefault((w1, w2), set()).add(len(self._neg_tuples)-1)
                        #js ^ dict((token_id\, token_id) : set([neg_tuple_id, ...]))
                        #   `.setdefault()` adds the (w1,w2) key if it's not present and if the (w1,w2) key _is_ present, `.setdefault()`
                        #   will add the new index of the neg_tuple id to the set of dictionary values
                        #   EX. a = {(52, 53): {4}}
                        #       a.setdefault((20,21),set()).add(3);  a == {(20, 21): {3}, (52, 53): {4}}
                        #       a.setdefault((20,21),set()).add(5);  a == {(20, 21): {3, 5}, (52, 53): {4}}            

                # update self._token_to_sents
                # js self._token_to_sents is a dict {token_id: {(updated_s1_id, new_index)}}  
                #   *NOTE* `index` is the index of the word in the tokenized sentence (`s1_tokenized`)
                
                for index,token_id in enumerate(s1_tokenized.ids):
                    if token_id in self.stop_words_set:
                        continue
                    sid_index = (s1_id, index)
                    self._token_to_sents.setdefault(token_id, set()).add(sid_index) 
                for index,token_id in enumerate(s2_tokenized.ids):
                    if token_id in self.stop_words_set:
                        continue
                    sid_index = (s2_id, index)
                    self._token_to_sents.setdefault(token_id, set()).add(sid_index)                              

                count += 1  # TODO count should be incremented here to get 20k examples


            #### PRINT STATEMENTS FOR TESTING ####
            #print(f'Total time: {(time.time() - t0):.2f} seconds')
            print('\n#### PRINT STATEMENTS FOR TESTING ####')
            print(f'Total pairs of questions: {len(sentence_id_pair_list)}')
            print(f'`count` is {count}, should be 20000')
            print(f'`bad_count` is {bad_count}')
            print(f'Number of paraphrases = {sum(label_list)}, {sum(label_list)/len(sentence_id_pair_list)*100:.2f}% of total')
            print(f'Number of NON-paraphrases = {len(label_list) - sum(label_list)}, {(len(label_list) - sum(label_list))/len(sentence_id_pair_list)*100:.2f}% of total')
            print('#### PRINT STATEMENTS FOR TESTING ####\n')
            #### PRINT STATEMENTS FOR TESTING ####


    def _overlap(self, s1: List[int], s2: List[int]) -> List[List[int]]:
        """ 
        Finds overlapping non-stopwords of two sentences and returns the words' index in each sentence
        IN:     s1 and s2 are tokenized sentences of token_ids
        RETURN: a list of lists; inner lists are [overlap_word_index_sent_1, overlap_word_index_sent_2]
        """
        # check intersection  
        # create dict mapping words to their indices in a given sentence
        s1_dict = dict((k, i) for i, k in enumerate(s1)) 
        s2_dict = dict((k, i) for i, k in enumerate(s2))
        word_pairs = []
        inter = set(s1_dict.keys()).intersection(set(s2_dict.keys())) #find all the common words between sentences

        inter.difference_update(self.stop_words_set) #js removes all token_ids of stopwords TO INCLUDE [CLS],[SEP],[PAD]
        #js NOTE: removed ~6 lines of individual filtering                   
        
        # check for digits or starts with '-'
        #TODO: quicker to make a list of token_ids that correspond to integers?
        #   it seems like it would be easer than coverting back to a string with the tokenizer...
        for i in inter.copy():                      #js convert back to strings; remove digits and words beginning with '-'
            if self.tokenizer.decode(i).isdigit():
                inter.remove(i)
            if self.tokenizer.decode(i).startswith('-'):
                inter.remove(i)

        for w in inter:
            w1_id = s1_dict[w]
            w2_id = s2_dict[w]
            word_pairs.append([w1_id, w2_id])

        #js NOTE: removed ~15 lines of references to synonyms/synonym_pairs because synonyms.tsv wasn't provided
        return word_pairs

    def _corrupt(self, para_tuple, target_sent=None):    
        """
        Corrupt para_tuple into a negative sample. 
        Return (sent_id, sent_id, index_of_an_overlapping/synonym_token, index_of_an_overlapping/synonym_token) for a negative sample.
        """
        
        #js assigns appropriate variable to the elements of `para_tuple = (s1_idx, s2_idx, s1_overlap_word_index, s2_overlap_word_idx)`
        s1_id = para_tuple[0]  
        s1_index = para_tuple[2]
        s2_id = para_tuple[1]
        s2_index = para_tuple[3]
        #js randomly 1 or 2 to pick whether or 'corrupt' s1_id or s2_id
        if target_sent == None:                     
            target_sent = random.randint(1,2)

        if target_sent == 1: 
            target_sent_id    = s1_id
            target_sent_index = s1_index
        if target_sent == 2: 
            target_sent_id    = s2_id
            target_sent_index = s2_index

        # token_id of ONE OVERLAP WORD; INDEXING: finds s1_id in `_id_to_sent` Tensor, then uses the overlap word index to get the token_id of that word
        #    then using `token`, find all possible sentences with that token: set(sent_id, index_of_token)
        token = self._id_to_sent[target_sent_id][target_sent_index]  
        sents_list = self._token_to_sents[token]  #TODO: use copy.copy()
        
        # remove s1 and s2 out of the set of possible sentences with token
        if (s1_id, s1_index) in sents_list:      
            sents_list.remove((s1_id, s1_index)) #js TODO: test to see if this modifies the actual self._token_to_sents dictionary
        if (s2_id, s2_index) in sents_list:
            sents_list.remove((s2_id, s2_index))

        # if the only overlap was between the two paraphrase sentences; *RANDOMLY RETURN A NEGATIVE TUPLE??* 
        #   TODO there has to be a better way; should just skip this example then.
        if (len(sents_list) == 0):
            return random.choice(self._neg_tuples)
        else:
            corrupt_s = random.choice(list(sents_list)) #find a sentence in the remaining set of overlap sentences (after casting to a list)
        
        # a 2nd check to see if the randomly selected set of overlap sentences is a set of paraphrases
        #   if the randomly selected sentence is a paraphrase, try 10 times to find a non-paraphrase 
        #   TODO: `.remove()` the paraphrased sentence pair if `.is_paraphrase() = True` and find overlap with negative tuple?
        ind = 0
        while self._is_paraphrase(corrupt_s[0], target_sent_id):     
            corrupt_s = random.choice(list(sents_list)) 
            ind += 1
            if ind > 10:
                # print("ind", ind)
                random.choice(self._neg_tuples)     # if there isn't a non-paraphrase: randomly return a negative tuple:
                break                                   
        return (corrupt_s[0],target_sent_id,corrupt_s[1], target_sent_index)


    def _gen_stop_words(self, filename: str) -> Set[int]:
        """
        Generates a set of token_ids given a text file of stopwords.
        """

        stop_words = np.genfromtxt(filename, dtype='str')
        tokenized_stop_words = self.tokenizer(" ".join(stop_words))[0]
        tokenized_punc = self.tokenizer(PUNC)[0] #TODO: move PUNC assignment within this method?

        stop_words_set = set.union(set(tokenized_stop_words.ids), set(tokenized_punc.ids))
        stop_words_set.add(self.tokenizer.pad_token_id) #add [PAD] == 0

        self.stop_words_set = stop_words_set

    def _gen_neg_tuple(self, para_tuple):
        """
        Given a paraphrase tuple, find a non-paraphrase pair of sentences that share the same word.
        NOTE: this is different than _corrupt() because corrupt() finds a random other sentence that shares a word.
        """

        s1_id = para_tuple[0]
        s1_index = para_tuple[2]
        s2_id = para_tuple[1]
        s2_index = para_tuple[3]
        s1_token = self._id_to_sent[s1_id][s1_index]
        s2_token = self._id_to_sent[s2_id][s2_index]
        if (s1_token, s2_token) in self._token_pair_to_neg_tuples:
            neg_tuple_id = random.choice(list(self._token_pair_to_neg_tuples[(s1_token, s2_token)]))
            neg_tuple = self._neg_tuples[neg_tuple_id]
            return neg_tuple
        else:
            return None

    def _is_paraphrase(self, s1_id, s2_id):
        return (s1_id,s2_id) in self._paraphrase_sets


    #js copied directly from Shi
    def save(self, filename):
        f = open(filename,'wb')
        #self.desc_embed = self.desc_embed_padded = None
        pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        print("Save data object as", filename)

    #js copied directly from Shi
    def load(self, filename):
        f = open(filename,'rb')
        tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        print("Loaded data object from", filename)
        print("=====================\nCaution: need to reload desc embeddings.\n=====================")



# FURTHER TODO:
# 1. When finding intersection of words, remove token_ids of '?', '[CLS]', '[SEP]', ... maybe all symbols & punctuation?
# 2. is_synonym and corrupt_n() not included from Shi's original Data class
# 3. might need to pass ._para_tuples and ._neg_tuples for testing purposes...that's the only thing linking the q_ids to the actual sentences
#    in the __getitem__() return.