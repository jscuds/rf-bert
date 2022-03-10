import copy
import logging
import pickle
import random
import re
import time
from typing import Dict, List, Optional, Set, Tuple

import datasets
import numpy as np
import torch
import tqdm

from allennlp.modules.elmo import Elmo, batch_to_ids
from mosestokenizer import MosesTokenizer
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

STOP_WORDS_LOC = 'stop_words_en.txt' #'NLTK_stop_words_en.txt'

# MODEL_NAME = "bert-base-uncased"
# MAX_LENGTH = 40
PUNC = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~—“”'

# TODO future:  (1) ParaDataset to work for all datasets?
#               (2) ELMo and BERT in same Dataset?  Would require changes to tokenizer, token_ids vs token_tuples len=50...

class ParaphraseDatasetBert(Dataset):
    def __init__(self, para_dataset: str = 'quora', model_name: str = 'bert-base-uncased', 
                 num_examples: int = None, max_length: int = 40, stop_words_file: str = 'stop_words_en.txt',
                 r1: float=0.5, seed: int = None):
        
        self.para_dataset = para_dataset
        self.model_name = model_name
        self.num_examples = num_examples
        self.max_length = max_length
        self.r1 = r1 # negative sample ratio
        self.seed = seed # set to None for random
        random.seed(self.seed)

        # TODO do these only apply to 'quora'? if so --> move into self._load_dataset
        # TODO combine self._token_pair_to_neg_tuples and self._neg_tuples because the former references the latter?

        self._id_to_sent: Dict[int, Tensor] = {} # maps question/sentence IDs to token tensors
        self._sent_to_id: Dict[Tuple[int,...], int] = {} # reverse lookup: token tensors (cast to tuple) to question/sentence IDs
        self._paraphrase_sets: Set[Tuple[int,int]] = set([]) # a set of {(sent1_id, sent2_id), (sent2_id, sent1_id) ...} to check whether two sentences are paraphrases or not.
        self._para_tuples: List[Tuple[int,int,int,int]] = [] # [(sent1_id, sent2_id, sent1_index_of_an_overlapping/synonym_token, sent2_index_of_an_overlapping/synonym_token), ... ]
        self._neg_tuples:  List[Tuple[int,int,int,int]] = [] # [(sent1_id, sent2_id, sent1_index_of_an_overlapping/synonym_token, sent2_index_of_an_overlapping/synonym_token), ... ]
        self._token_pair_to_neg_tuples: Dict[Tuple[int,int],Set[int]] = {} # maps pair of word_token_ids to their index position in self._neg_tuples; {(token_id, token_id) : set([idx of self._neg_tuples, ...])}
        self._token_to_sents: Dict[int,Set[Tuple[int,int]]] = {} # reverse index of sentences given tokens. This is a map {token_id : set([(sent_id, index_of_the_token_in_the_sentence), ...]) }
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # load stop words from file
        self.bad_words = ["-LSB-", "\\", "``", "-LRB-", "????", "n/a", "'"] #, "//" #js add if you want to filter out some URLs
        self.stop_words_set = set([])
        logger.info('Generating stop words from {stop_words_file}...\n')
        self._gen_stop_words(stop_words_file)

        # load quora, mrpc, etc.
        logger.info('Loading %s dataset with r1=%d...\n', para_dataset, r1)
        self._load_dataset()

    def __len__(self) -> int:
        return len(self._para_tuples)

    # based on Shi's gen_batch
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, int, int, int, int]:
        # find a non-paraphrase pair of sentences that share the same word.
        neg_tuple = self._gen_neg_tuple(self._para_tuples[idx])
        if (random.uniform(0,1) > self.r1) or (neg_tuple is None):
            # find a random sentence in the dataset that also shares the same word.
            neg_tuple = self._corrupt(self._para_tuples[idx])
            
        sent1 = self._id_to_sent[self._para_tuples[idx][0]] # tensor of char representations of tokens / shape [1,max_length,50]
        sent2 = self._id_to_sent[self._para_tuples[idx][1]] 
        nsent1 = self._id_to_sent[neg_tuple[0]] 
        nsent2 = self._id_to_sent[neg_tuple[1]]
        token1 = self._para_tuples[idx][2] # shared word index in sent1
        token2 = self._para_tuples[idx][3] # shared word index in sent2
        ntoken1 = neg_tuple[2] # shared word index in nsent1
        ntoken2 = neg_tuple[3] # shared word index in nsent2
        return torch.tensor(sent1), torch.tensor(sent2), torch.tensor(nsent1), torch.tensor(nsent2), token1, token2, ntoken1, ntoken2
    
    # TODO add MRPC, PAN...
    def _load_dataset(self):
        """
        Loads relevant paraphrase dataset based on argument passed to constructor and creates attributes for model training.
        """

        # load quora from HuggingFace
        if self.para_dataset == 'quora':
            quora = datasets.load_dataset(self.para_dataset)
            
            quora_shuffled = quora.shuffle(seed=self.seed)
            count = 0

            bad_count = 0              #js added for testing
            sentence_id_pair_list = [] #js added for testing
            label_list = []            #js added for testing

            # TODO: Tokenize things beforehand!
            for pair in quora_shuffled['train']:
                #count += 1 #TODO incrementing here means you don't actually get 20k examples. It's what Shi did.
                if (self.num_examples is not None) and count >= self.num_examples:
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
                #js TODO elmo_change using batch_to_ids and tokenizer
                s1_tokenized = self.tokenizer(s1.lower(),truncation=True, max_length = self.max_length, padding= 'max_length')[0] 
                s2_tokenized = self.tokenizer(s2.lower(),truncation=True, max_length = self.max_length, padding= 'max_length')[0]

                #Shi uses a list for tokenized sentences; where the index is equivalent to the updated id; 
                #   js: I'm going to just use a dictionary
                
                #if not s1_tokenized in sent2id: shouldn't be necessary due to id2sent.update() functionality
                #js TODO elmo_change to s1_tokenized
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
                        intersection_tuple = (s1_id,s2_id,p[0],p[1]) #js (s1_idx, s2_idx, s1_overlap_word_index, s2_overlap_word_idx) **SO THERE COULD BE MULTIPLE TUPLES STARTING WITH  `s1_idx, s2_idx`** if they have multiple overlapping words
                        self._para_tuples.append(intersection_tuple) #js a list of tuples in the format from the line above
                else: #js this from NON-PARAPHRASE questions
                    for p in total_index_pairs:
                        intersection_tuple = (s1_id, s2_id, p[0], p[1])
                        self._neg_tuples.append(intersection_tuple)
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
                # js self._token_to_sents is a dict {token_id: {(sent_id, index of token corresponding to token_id)}} 
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
            #logger.info(f'Total time: {(time.time() - t0):.2f} seconds')
            logger.info('\n#### PRINT STATEMENTS FOR TESTING ####')
            logger.info(f'Total pairs of questions: {len(sentence_id_pair_list)}')
            logger.info(f'`count` is {count}, should be {self.num_examples}')
            logger.info(f'`bad_count` is {bad_count}')
            logger.info(f'Number of paraphrases = {sum(label_list)}, {sum(label_list)/len(sentence_id_pair_list)*100:.2f}% of total')
            logger.info(f'Number of NON-paraphrases = {len(label_list) - sum(label_list)}, {(len(label_list) - sum(label_list))/len(sentence_id_pair_list)*100:.2f}% of total')
            logger.info('#### PRINT STATEMENTS FOR TESTING ####\n')
            #### PRINT STATEMENTS FOR TESTING ####

            # map boolean list of labels to list of 0,1 ints
            self.labels = list(map(int, label_list))


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
        
        #js TODO: elmo_change: this won't work with elmo because I don't have a way to go from token_ids --> strings
        for i in inter.copy():                      #js convert back to strings; remove digits and words beginning with '-'
            if self.tokenizer.decode(i).isdigit():
                inter.remove(i)
            if self.tokenizer.decode(i).startswith('-'): #TODO ELMO_CHANGE: ord('-') == 45; 46 after batch_to_ids() b/c of off-by-one issue
                inter.remove(i)                          #   so you can find things that start with 259

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
        sents_set = copy.copy(self._token_to_sents[token])  #copy.copy() o/w it modifies the actual self._token_to_sents dictionary
        
        # remove s1 and s2 out of the set of possible sentences with token
        if (s1_id, s1_index) in sents_set:      
            sents_set.remove((s1_id, s1_index)) 
        if (s2_id, s2_index) in sents_set:
            sents_set.remove((s2_id, s2_index))

        # if the only overlap was between the two paraphrase sentences; *RANDOMLY RETURN A NEGATIVE TUPLE??* 
        #   TODO there has to be a better way; should just skip this example then.
        if (len(sents_set) == 0):
            return random.choice(self._neg_tuples)
        else:
            corrupt_s = random.choice(list(sents_set)) #find a sentence in the remaining set of overlap sentences (after casting to a list)
        
        # a 2nd check to see if the randomly selected set of overlap sentences is a set of paraphrases
        #   if the randomly selected sentence is a paraphrase, try 10 times to find a non-paraphrase 
        #   TODO: `.remove()` the paraphrased sentence pair if `.is_paraphrase() = True` and find overlap with negative tuple?
        ind = 0
        while self._is_paraphrase(corrupt_s[0], target_sent_id):     
            corrupt_s = random.choice(list(sents_set)) 
            ind += 1
            if ind > 10:
                # logger.info("ind", ind)
                random.choice(self._neg_tuples)     # if there isn't a non-paraphrase: randomly return a negative tuple:
                break                                   
        return (corrupt_s[0], target_sent_id,corrupt_s[1], target_sent_index)


    def _gen_stop_words(self, filename: str) -> Set[int]:
        """
        Generates a set of token_ids given a text file of stopwords.
        """

        stop_words = np.genfromtxt(filename, dtype='str')
        stop_words = stop_words.tolist()
        # have a variant of all stopwords where they are capitalized (use case: beginning of sentence)
        stop_words = [token.capitalize() for token in stop_words] + stop_words

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
        logger.info("Save data object as", filename)

    #js copied directly from Shi
    def load(self, filename):
        f = open(filename,'rb')
        tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        logger.info("Loaded data object from", filename)
        logger.info("=====================\nCaution: need to reload desc embeddings.\n=====================")

class ParaphraseDatasetElmo(Dataset):
    """
    Creates a Dataset object for retrofitting ELMo with paraphrase examples.
    """    
    def __init__(self, para_dataset: str = 'quora', model_name: str = 'elmo', 
                 num_examples: int = 25000, max_length: int = 40, stop_words_file: str = 'stop_words_en.txt',
                 r1: float=0.5, seed: int = None, split: str = 'train'):
        
        self.para_dataset = para_dataset
        self.model_name = model_name
        self.num_examples = num_examples
        self.max_length = max_length
        self.r1 = r1 # negative sample ratio
        self.seed = seed # set to None for random
        self._split = split

        # TODO do these only apply to 'quora'? if so --> move into self._load_dataset
        # TODO combine self._token_pair_to_neg_tuples and self._neg_tuples because the former references the latter?

        self._id_to_sent: Dict[int, Tuple[Tuple[int,...],...]] = {} # maps question/sentence IDs to tuple of token tuples
        self._sent_to_id: Dict[Tuple[Tuple[int,...],...], int] = {} # reverse lookup: token tensors (cast to tuple of tuples) to question/sentence IDs
        self._paraphrase_sets: Set[Tuple[int,int]] = set([]) # a set of {(sent1_id, sent2_id), (sent2_id, sent1_id) ...} to check whether two sentences are paraphrases or not.
        self._para_tuples: List[Tuple[int,int,int,int]] = [] # [(sent1_id, sent2_id, sent1_index_of_an_overlapping/synonym_token, sent2_index_of_an_overlapping/synonym_token), ... ]
        self._neg_tuples:  List[Tuple[int,int,int,int]] = [] # [(sent1_id, sent2_id, sent1_index_of_an_overlapping/synonym_token, sent2_index_of_an_overlapping/synonym_token), ... ]
        
        #elmo_change: any token_id is an `int` in BERT, but a `tensor` w/  shape [50]  cast to a tuple w/ len=50
        self._token_pair_to_neg_tuples: Dict[Tuple[Tuple[int,...],Tuple[int, ...]],Set[int]] = {} # maps pair of word_token_ids to their index position in self._neg_tuples; {(token_id, token_id) : set([idx of self._neg_tuples, ...])}
        self._token_to_sents: Dict[Tuple[int,...],Set[Tuple[int,int]]] = {} # reverse index of sentences given tokens. This is a map {token_id : set([(sent_id, index_of_the_token_in_the_sentence), ...]) }
        
        #elmo_change
        self.moses_tokenizer = MosesTokenizer('en', no_escape=True)
        self.tokenizer = lambda s: self.moses_tokenizer(s.lower())

        # load stop words from file
        self.bad_words = ["-LSB-", "\\", "``", "-LRB-", "????", "n/a", "'"] #, "//" #js add if you want to filter out some URLs
        self.stop_words_set = set([])
        logger.info(f'Generating stop words from {stop_words_file}...\n')
        self._gen_stop_words(stop_words_file)

        # load quora, mrpc, etc.
        logger.info('Loading %s dataset with r1=%d...\n', para_dataset, r1)
        self._load_dataset()

    def __len__(self) -> int:
        return len(self._para_tuples)

    # based on Shi's gen_batch
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, int, int, int, int]:
        neg_tuple = self._gen_neg_tuple(self._para_tuples[idx])
        if (random.uniform(0,1) > self.r1) or (neg_tuple is None):
            neg_tuple = self._corrupt(self._para_tuples[idx])
            
        #TODO: instead of token_ids, these will be tensors of length 50...problematic?
        sent1 = self._id_to_sent[ self._para_tuples[idx][0] ] # tuple of char ids / len [self.max_length*50]
        sent2 = self._id_to_sent[ self._para_tuples[idx][1] ] 
        nsent1 = self._id_to_sent[ neg_tuple[0] ] 
        nsent2 = self._id_to_sent[ neg_tuple[1] ] 
        token1 = self._para_tuples[idx][2] # index location of shared word in sent1
        token2 = self._para_tuples[idx][3] 
        ntoken1 = neg_tuple[2] # index location of shared word in nsent1
        ntoken2 = neg_tuple[3]
        return (torch.tensor(sent1).reshape(self.max_length,50), 
                torch.tensor(sent2).reshape(self.max_length,50), 
                torch.tensor(nsent1).reshape(self.max_length,50), 
                torch.tensor(nsent2).reshape(self.max_length,50), 
                token1, token2, ntoken1, ntoken2)

    # TODO add MRPC, PAN...
    def _load_dataset(self):
        """
        Loads relevant paraphrase dataset based on argument passed to constructor and creates attributes for model training.
        """

        # load quora or mrpc from HuggingFace
        if self.para_dataset == 'quora':
            dataset = datasets.load_dataset('glue', 'qqp')
            dataset_shuffled = dataset.shuffle(seed=self.seed)
            self._process_dataset(dataset_shuffled)
        elif self.para_dataset == 'mrpc':
            dataset = datasets.load_dataset('glue', 'mrpc')
            dataset_shuffled = dataset.shuffle(seed=self.seed)
            self._process_dataset(dataset_shuffled)

    def _process_dataset(self, dataset_shuffled: datasets.DatasetDict):
        count = 0

        bad_count = 0              #js added for testing
        self.sentence_id_pair_list = [] #js added for testing
        label_list = []            #js added for testing

        for pair in tqdm.tqdm(dataset_shuffled[self._split], desc=f'Processing {self.para_dataset} paraphrases from {self._split} split'):
            #count += 1 #TODO incrementing here means you don't actually get 20k examples. It's what Shi did.
            if (self.num_examples is not None) and count >= self.num_examples:
                break
            if self.para_dataset == 'quora':
                label = pair['label']
                s1_id = pair['idx']*10
                s2_id = pair['idx']*10 + 1
                s1 = pair['question1']
                s2 = pair['question2']

            elif self.para_dataset == 'mrpc':
                label = pair['label']
                # mrpc dataset index is for the sentence pair, so I created new indices for each sentence 
                # by multiplying pair index *10 + 0 for s1  and + 1 for s2
                s1_id = pair['idx']*10
                s2_id = pair['idx']*10 + 1
                s1 = pair['sentence1']
                s2 = pair['sentence2']

            # TODO: better way? .map() functionality of quora/mrpc dataset object? map bad words to [UNK]?
            exist_bad_word = False
            for i in self.bad_words:
                if (i in s1 or i in s2):
                    exist_bad_word = True

            if exist_bad_word: 
                bad_count += 1  #js added for testing
                continue # skip pair if it contains element from self.bad_words

            # Look for newlines (they will break the Moses tokenizer).
            if '\n' in s1:
                logger.warn(f'Replacing newlines in s1: "{s1}"')
                s1 = s1.replace('\n', ' ')
            if '\n' in s2:
                logger.warn(f'Replacing newlines in s2: "{s2}"')
                s2 = s2.replace('\n', ' ')

            #### CODE FOR TESTING HOW MANY EXAMPLES ARE POSITIVE AND NEGATIVE ####
            self.sentence_id_pair_list.append((s1_id,s2_id)) #js added
            label_list.append(label)                    #js added
            #### CODE FOR TESTING HOW MANY EXAMPLES ARE POSITIVE AND NEGATIVE ####


            #elmo_change using batch_to_ids and tokenizer
            s1_interim_tokenized = batch_to_ids([self.tokenizer(s1.lower())]) # shape: (1,seq_length, 50)
            s2_interim_tokenized = batch_to_ids([self.tokenizer(s2.lower())]) # shape: (1,seq_length, 50) 

            # Skip if one of the inputs has no tokens (this is a real issue).
            if (not s1_interim_tokenized.numel()) or (not s2_interim_tokenized.numel()):
                logger.warn(f'Skipping pair with empty question: {pair}')
                continue
            

            # add padding to self.max_length && truncate if number of tokens > self.max_length
            s1_tokenized = torch.zeros(1, self.max_length, 50, dtype=torch.int64) # shape: (1, self.max_length, 50 for ELMo character encoding)
            s2_tokenized = torch.zeros(1, self.max_length, 50, dtype=torch.int64)

            # Truncate to self.max_length
            s1_tokenized[:,:s1_interim_tokenized.shape[1],:] = torch.clone(s1_interim_tokenized[:,:self.max_length,:])
            s2_tokenized[:,:s2_interim_tokenized.shape[1],:] = torch.clone(s2_interim_tokenized[:,:self.max_length,:])     
            
            # Convert sentences to tuple of tuples for dictionaries
            #   tuple of tuples: len(outer_tuple) = self.max_length; len(outer_tuple[<any index>]) = 50
            s1_tuple = self._tensor_to_token_tuples(s1_tokenized)
            s2_tuple = self._tensor_to_token_tuples(s2_tokenized)

            #Shi uses a list for tokenized sentences; where the index is equivalent to the updated id; 
            #   js: I'm going to just use a dictionary
            
            #if not s1_tokenized in sent_to_id: shouldn't be necessary due to id_to_sent.update() functionality
            #js NOTE have to use Tuples because tensors aren't hashable
            self._id_to_sent.update({s1_id : s1_tuple})
            self._id_to_sent.update({s2_id : s2_tuple})
            self._sent_to_id.update({s1_tuple : s1_id})
            self._sent_to_id.update({s2_tuple : s2_id})
            
            #TODO: ELMO_CHANGE: ensure overlap returns what you expect
            total_index_pairs = self._overlap(s1_tuple, s2_tuple)
            
            #js if `is_duplicate`== True; append both combinations s1&s2 ++ s2&s1 to `self.paraphrases`
            if label:
                self._paraphrase_sets.add((s1_id, s2_id))
                self._paraphrase_sets.add((s2_id, s1_id))
                for p in total_index_pairs:             #js for every pair of overlap words:
                    intersection_tuple = (s1_id,s2_id,p[0],p[1]) #js (s1_idx, s2_idx, s1_overlap_word_index, s2_overlap_word_idx) **SO THERE COULD BE MULTIPLE TUPLES STARTING WITH  `s1_idx, s2_idx`** if they have multiple overlapping words
                    self._para_tuples.append(intersection_tuple) #js a list of tuples in the format from the line above
            else: #js this from NON-PARAPHRASE questions
                for p in total_index_pairs:
                    intersection_tuple = (s1_id, s2_id, p[0], p[1])
                    self._neg_tuples.append(intersection_tuple)

                    #js NOTE I think w1 == w2; except if/when synonyms are involved
                    w1 = self._tensor_to_token_tuples(s1_tokenized[:,p[0],:]) # original w1 & w2 shape: [1,1,50]
                    w2 = self._tensor_to_token_tuples(s2_tokenized[:,p[1],:]) # new w1 & w2 len = 50

                    # TODO: we should lowercase words before checking stopwords
                    #js filter out stopwords from negative sentences
                    #  stopwords are filtered out of pos sentences in self.overlap_()
                    if w1 in self.stop_words_set or w2 in self.stop_words_set:
                        continue                                               

                    self._token_pair_to_neg_tuples.setdefault((w1, w2), set()).add(len(self._neg_tuples)-1)
                    #js ^ dict((token_id\, token_id) : set([neg_tuple_id, ...]))
                    #   `.setdefault()` adds the (w1,w2) key if it's not present and if the (w1,w2) key _is_ present, `.setdefault()`
                    #   will add the new index of the neg_tuple id to the set of dictionary values
                    #   EX. a = {(52, 53): {4}}
                    #       a.setdefault((20,21),set()).add(3);  a == {(20, 21): {3}, (52, 53): {4}}
                    #       a.setdefault((20,21),set()).add(5);  a == {(20, 21): {3, 5}, (52, 53): {4}}            

            # update self._token_to_sents
            # js self._token_to_sents is a dict {token_tuple: {(sent_id, index of token corresponding to token_tuple)}}  
            #   *NOTE* `index` is the index of the word in the tokenized sentence (`s1_tokenized`)

            for index, token_tuple in enumerate(s1_tuple):
                if token_tuple in self.stop_words_set:
                    continue
                sid_index = (s1_id, index)
                self._token_to_sents.setdefault(token_tuple, set()).add(sid_index) 
            for index, token_tuple in enumerate(s2_tuple):
                if token_tuple in self.stop_words_set:
                    continue
                sid_index = (s2_id, index)
                self._token_to_sents.setdefault(token_tuple, set()).add(sid_index)                              

            count += 1  # NOTE: count should be incremented here to get correct num_examples

        #### PRINT STATEMENTS FOR TESTING ####
        #logger.info(f'Total time: {(time.time() - t0):.2f} seconds')
        logger.info('\n#### PRINT STATEMENTS FOR TESTING ####')
        logger.info(f'Total pairs of questions: {len(self.sentence_id_pair_list)}')
        logger.info(f'`count` is {count}, should be {self.num_examples}')
        logger.info(f'`bad_count` is {bad_count}')
        logger.info(f'Number of paraphrases = {sum(label_list)}, {sum(label_list)/len(self.sentence_id_pair_list)*100:.2f}% of total')
        logger.info(f'Number of NON-paraphrases = {len(label_list) - sum(label_list)}, {(len(label_list) - sum(label_list))/len(self.sentence_id_pair_list)*100:.2f}% of total')
        logger.info('#### PRINT STATEMENTS FOR TESTING ####\n')
        #### PRINT STATEMENTS FOR TESTING ####

        # map boolean list of labels to list of 0,1 ints
        self.labels = list(map(int, label_list))


    def _overlap(self, s1_tuple: Tuple[Tuple[int,...],...], s2_tuple: Tuple[Tuple[int,...],...]) -> List[List[int]]:
        """ 
        Finds overlapping non-stopwords of two sentences and returns the words' index in each sentence
        IN:     s1 and s2 are tuples of token tuples
        RETURN: a list of lists; inner lists are [overlap_word_index_sent_1, overlap_word_index_sent_2]
        """
         
        # create dict mapping words (keys) to their indices (values) in a given sentence
        #    where a word is a tuple of length 50
        assert len(s1_tuple) == len(s2_tuple) == self.max_length
        s1_dict = dict((k,i) for i, k in enumerate(s1_tuple))
        s2_dict = dict((k,i) for i, k in enumerate(s2_tuple))
        word_pairs = []
        
        # check intersection: find all the common words between sentences
        inter = set(s1_dict.keys()).intersection(set(s2_dict.keys()))

        #removes all token_ids of stopwords TO INCLUDE [0]*50 (padding)
        inter.difference_update(self.stop_words_set) 
        
        #elmo_change: same effect as filtering out all words that begin with '-' or digits 0-9
        #   index 0 of a chracter_id token tuple (length 50) is always 259
        #   index 1 is the first character of the word, so this is what we're checking
        #   elmo char_ids are based on ord()+1, so ord('-') = 45, but elmo char_id of '-' is 46
        #   elmo char_ids for digits '0' thru '9' are in range(49,59)
        #   setting the list in range(46,59) filters out all words beginning with '-' or digits.
        for i in inter.copy():
            if i[1] in list(range(46,59)):
                inter.remove(i)

        for w in inter:
            w1_id = s1_dict[w]
            w2_id = s2_dict[w]
            word_pairs.append([w1_id, w2_id])

        #js NOTE: removed ~15 lines of references to synonyms/synonym_pairs because synonyms.tsv wasn't provided
        return word_pairs

    def _corrupt(self, para_tuple: Tuple[int,int,int,int], target_sent: Optional[int]=None) -> Tuple[int,int,int,int]:
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
        sents_set = copy.copy(self._token_to_sents[token])  #copy.copy() o/w it modifies the actual self._token_to_sents dictionary
        
        # remove s1 and s2 out of the set of possible sentences with token
        if (s1_id, s1_index) in sents_set:      
            sents_set.remove((s1_id, s1_index)) 
        if (s2_id, s2_index) in sents_set:
            sents_set.remove((s2_id, s2_index))

        # if the only overlap was between the two paraphrase sentences; *RANDOMLY RETURN A NEGATIVE TUPLE??* 
        #   TODO there has to be a better way; should just skip this example then.
        if (len(sents_set) == 0):
            return random.choice(self._neg_tuples)
        else:
            corrupt_s = random.choice(list(sents_set)) #find a sentence in the remaining set of overlap sentences (after casting to a list)
        
        # a 2nd check to see if the randomly selected set of overlap sentences is a set of paraphrases
        #   if the randomly selected sentence is a paraphrase, try 10 times to find a non-paraphrase 
        #   TODO: `.remove()` the paraphrased sentence pair if `.is_paraphrase() = True` and find overlap with negative tuple?
        ind = 0
        while self._is_paraphrase(corrupt_s[0], target_sent_id):     
            corrupt_s = random.choice(list(sents_set)) 
            ind += 1
            if ind > 10:
                # logger.info("ind", ind)
                random.choice(self._neg_tuples)     # if there isn't a non-paraphrase: randomly return a negative tuple:
                break                                   
        return (corrupt_s[0],target_sent_id,corrupt_s[1], target_sent_index)


    def _gen_stop_words(self, filename: str) -> Set[int]:
        """
        Generates a set of token_ids given a text file of stopwords.
        *NOTE*: doesn't include `<S>` or `</S>` in list of stopwords
        """

        stop_words = np.genfromtxt(filename, dtype='str')
        stop_words = stop_words.tolist()
        # have a variant of all stopwords where they are capitalized (use case: beginning of sentence)
        stop_words = [token.capitalize() for token in stop_words] + stop_words

        tokenized_stop_words = self.tokenizer(" ".join(stop_words))
        tokenized_punc = self.tokenizer(PUNC)[0] #TODO: move PUNC assignment within this method?

        stop_words_set = set.union(set(tokenized_stop_words), set(tokenized_punc))
        # squeezes tokens of shape [1,1,50] --> [50]  --> list --> tuple because it's hashable
        stop_words_set = set([tuple(batch_to_ids([[word]]).squeeze().tolist()) for word in stop_words_set])
        # add padding to stop_words_set
        stop_words_set.add(tuple([0]*50))

        # a set of of tuples representing the length-50 ELMo char_ids
        self.stop_words_set = stop_words_set

    def _gen_neg_tuple(self, para_tuple: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int]]:
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

    def _tensor_to_token_tuples(self, sent: Tensor) -> Tuple[Tuple[int,...]]:
        '''
        Converts a tensor to either 
            (1) a tuple of tuples where each outer tuple entry represents a word
        as a tuple with length 50.
            (2) a single tuple representing a word
        '''
        # convert a sentence tensor of shape [1,40,50]
        assert sent.shape[-1] == 50
        if len(sent.shape) == 3:
            assert sent.shape[0] == 1
            assert sent.shape[1] == self.max_length    
            return tuple(tuple(token.tolist()) for token in sent.squeeze())
        
        # convert a word tensor of shape [1,50]
        if len(sent.shape) == 2:
            assert sent.shape[0] == 1
            return tuple(sent.squeeze().tolist())

    # TODO(js): ability to save the dataset after it's created?  The below code copied from Shi doesn't work
    
    # #js copied directly from Shi
    # def save(self, filename):
    #     f = open(filename,'wb')
    #     #self.desc_embed = self.desc_embed_padded = None
    #     pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
    #     f.close()
    #     print("Save data object as", filename)

    # #js copied directly from Shi
    # def load(self, filename):
    #     f = open(filename,'rb')
    #     tmp_dict = pickle.load(f)
    #     self.__dict__.update(tmp_dict)
    #     print("Loaded data object from", filename)
    #     print("=====================\nCaution: need to reload desc embeddings.\n=====================")
