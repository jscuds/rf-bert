from typing import Tuple

import datasets
import sklearn.model_selection
import torch

from allennlp.modules.elmo import batch_to_ids

from torch.utils.data import Dataset

from mosestokenizer import MosesTokenizer

# import allennlp

# # https://github.com/luismsgomes/mosestokenizer
# from mosestokenizer import MosesTokenizer

class QuoraDataset(Dataset):
    def __init__(self, para_dataset: str = 'quora', num_examples: int = 20000, max_length: int = 40, 
                 train_test_split: float = 0.8, seed: int = None):
        
        self.para_dataset = para_dataset
        self.num_examples = num_examples
        self.max_length = max_length
        self.seed = seed # set to None for random

        self.sentence_id_pair_list = [] # index of this list corresponds to index of tokenized_sentences
        # elmo_change
        # self.tokenized_sentences = [] # lists of [CLS] <s1> [SEP] <s2> [SEP]; where <s1> and <s2> are tokenized sentences
        self.s1_tokenized = [] # lists of sentences of token_ids from s1
        self.s2_tokenized  = [] # lists of sentences of token_ids from s2
        self.labels = [] # list of labels as ints: 0,1

        #elmo_change
        # https://github.com/luismsgomes/mosestokenizer
        self.tokenizer = MosesTokenizer('en') 
        self.bad_words = ["-LSB-", "\\", "``", "-LRB-", "????", "n/a", "'"] #, "//" #js add if you want to filter out some URLs

        # load quora, mrpc, etc.
        print(f'Loading {para_dataset} dataset...\n')
        self._load_dataset()
        self._train_test_split(train_test_split)
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
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._split == 'train':
            # self.train_sents has shape B x 2 x self.max_length x word_length==50
            return self.train_sents[idx,:,:,:], torch.tensor(self.train_labels[idx])

            # return torch.tensor(self.train_sents[idx].squeeze()), torch.tensor(self.train_labels[idx]) 
            # #elmo_change ^^ had to squeeze() to pass shape [T,50] instead of [1,T,50] for batching

        if self._split == 'test':
            # self.test_sents has shape B x 2 x self.max_length x word_length==50

            return self.test_sents[idx,:,:,:], torch.tensor(self.test_labels[idx]) 

            # return torch.tensor(self.test_sents[idx].squeeze()), torch.tensor(self.test_labels[idx]) 
            # #elmo_change ^^ had to squeeze() to pass shape [T,50] instead of [1,T,50] for batching

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

                #js experiment no duplicate questions

                #### CODE FOR TESTING HOW MANY EXAMPLES ARE POSITIVE AND NEGATIVE ####
                self.sentence_id_pair_list.append((s1_id,s2_id)) #js added
                label_list.append(label)                    #js added
                #### CODE FOR TESTING HOW MANY EXAMPLES ARE POSITIVE AND NEGATIVE ####

                #elmo_change thru `count += 1` 
                # tokenize the sentences and turn into tensor of ELMo character_ids
                # concat_sents = ' '.join([s1,s2])
                # print(concat_sents)
                # interim_tokenized = batch_to_ids([self.tokenizer.tokenize(concat_sents, escape=False)]) # shape: (1,seq length, 50)
                # TODO: initialize tokenizer in a smarter way
                s1_interim_tokenized = batch_to_ids([self.tokenizer(s1)]) # shape: (1,seq length, 50)
                s2_interim_tokenized = batch_to_ids([self.tokenizer(s2)]) # shape: (1,seq length, 50) 

                # add padding to self.max_length && truncate if number of tokens > self.max_length
                # # full_tokenized = torch.zeros(1, self.max_length, 50, dtype=torch.int64) # shape: (one example, self.max_length, 50 for ELMo character encoding)
                s1_full_tokenized = torch.zeros(1, self.max_length, 50, dtype=torch.int64)
                s2_full_tokenized = torch.zeros(1, self.max_length, 50, dtype=torch.int64)

                # Truncate to self.max_length
                # # full_tokenized[:,:interim_tokenized.shape[1],:] = torch.clone(interim_tokenized[:,self.max_length,:]) #previously a BUG for 2x WandB ELMo runs; THERE IS NO `:` before self.max_length...so it just copied the self.max_length cell across the whole tensor
                # s1_full_tokenized[:,:s1_interim_tokenized.shape[1],:] = torch.clone(s1_interim_tokenized[:,:self.max_length,:])
                # s2_full_tokenized[:,:s2_interim_tokenized.shape[1],:] = torch.clone(s2_interim_tokenized[:,:self.max_length,:])
                s1_full_tokenized[:,:s1_interim_tokenized.shape[1],:] = torch.clone(s1_interim_tokenized[:,:self.max_length,:])
                s2_full_tokenized[:,:s2_interim_tokenized.shape[1],:] = torch.clone(s2_interim_tokenized[:,:self.max_length,:])               
                
                # Ensure values and shapes match
                assert torch.all(torch.eq(s1_full_tokenized[:,:s1_interim_tokenized.shape[1],:], s1_interim_tokenized[:,:self.max_length,:]))
                assert torch.all(torch.eq(s2_full_tokenized[:,:s2_interim_tokenized.shape[1],:], s2_interim_tokenized[:,:self.max_length,:]))
                assert s1_full_tokenized.shape == (1,self.max_length,50)
                assert s2_full_tokenized.shape == (1,self.max_length,50)

                # self.tokenized_sentences.append(full_tokenized)
                self.s1_tokenized.append(s1_full_tokenized)
                self.s2_tokenized.append(s2_full_tokenized)      


                count += 1  # NOTE: count should be incremented here to get 20k examples


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

    def _train_test_split(self, train_test_split: float=0.8):
        """
        If called, creates train/test splits of sents/labels.
        """
        # q = torch.stack((torch.cat(s1_tokenized),torch.cat(s2_tokenized)),dim=1)
        # SHAPE: num_examples x (s1,s2) x max_length x char length (50)
        # SHAPE: num_examples x 2       x 40         x 50
        self.tokenized_sentences = torch.stack((torch.cat(self.s1_tokenized), 
                                                torch.cat(self.s2_tokenized)), 
                                                dim=1)
        try:
            assert self.tokenized_sentences.shape == (self.num_examples,2,40,50)
        except AssertionError:
            print(f'self.tokenized_sentences.shape: {self.tokenized_sentences.shape}\nintended shape: {(self.num_examples,2,40,50)}')


        self.train_sents, self.test_sents, self.train_labels, self.test_labels = (
            sklearn.model_selection.train_test_split(
                self.tokenized_sentences, self.labels, train_size=train_test_split, test_size=1-train_test_split, 
                random_state=self.seed, shuffle = True, stratify = None
            )
        )
        print(f'Training Size: {len(self.train_sents)}')
        print(f'Testing Size: {len(self.test_sents)}')
