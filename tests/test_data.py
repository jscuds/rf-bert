import sys
from pathlib import Path

import numpy as np
import torch
from dataloaders import ParaphraseDatasetBert, ParaphraseDatasetElmo
from mosestokenizer import MosesTokenizer

from dataloaders.helpers import load_rotten_tomatoes, load_qqp, load_sst2, train_test_split

# Instantiate ParaphraseDataset class variants for BERT and ELMo

class TestQuoraBert:
    quora = ParaphraseDatasetBert(para_dataset = 'quora', model_name = 'bert-base-uncased', num_examples = 20000, 
                    max_length = 40, stop_words_file = './stop_words_en.txt', r1 =0.5, seed = 42)


    def test_id_to_sent(self):
        # sentence, but pad to length 40
        sent = [101, 2129, 2079, 1045, 3857, 2026, 6337, 2005, 2327, 1038, 1011, 2816, 1029, 102] + ([0] * 26)
        assert (
            self.quora._id_to_sent[213590] == sent
        )

    def test_sent_to_id(self):
        idx = 213590
        tokenized_sentence_tuple = tuple(self.quora._id_to_sent[idx])
        print('id:', id)
        assert self.quora._sent_to_id[tokenized_sentence_tuple] == idx
    
    def test_paraphrase_sets(self):
        paraphrase_ids = sorted(list(self.quora._paraphrase_sets))[0]
        assert paraphrase_ids == (32, 1101) #[136860, 136861] #[67843, 105500]
        print(f'paraphrase_sets[0]:\t{paraphrase_ids}')      # (136860, 136861) OR (105500, 67843) OR...because it's a set

    def test_para_tuples(self):
        assert self.quora._para_tuples[0] == (213590, 149174, 6, 6)
        assert len(self.quora._para_tuples) == 20_062 # TODO: CHECK; 17049 colab vs. 17029 local

    def test_neg_tuples(self):
        assert self.quora._neg_tuples[0] == (416022, 416023, 18, 4)
        assert len(self.quora._neg_tuples) == 23099 # TODO: CHECK; 19806 colab vs 19773 local
    
    def test_token_pair_to_neg_tuples(self):
        assert self.quora._token_pair_to_neg_tuples[(12952, 12952)] == {7712, 12931, 19173, 11, 21644, 15791, 11824, 20402, 2611, 19732}
        # token 12952 == '##pal'
        # TODO CHECK; {11, 2614, 7723, 11841, 12950, 15814, 19206, 19765, 20436, 21679}   
    
    def test_token_to_sents(self):
        assert len(self.quora._token_to_sents.keys()) == 14_543# TODO: check - 13848??

    
    def test_list_types_and_len(self):
        assert type(self.quora._para_tuples) == list
        assert type(self.quora._neg_tuples) == list
        assert len(self.quora) == len(self.quora._para_tuples)

    def test_quora_example_types(self):
        test_sent1, test_sent2, test_nsent1, test_nsent2, test_token1, test_token2, test_ntoken1, test_ntoken2 = self.quora[0]
        assert isinstance(test_sent1, torch.Tensor)
        assert isinstance(test_sent2, torch.Tensor)
        assert isinstance(test_nsent1, torch.Tensor)
        assert isinstance(test_nsent2, torch.Tensor)
        assert isinstance(test_token1, int)
        assert isinstance(test_token2, int)
        assert isinstance(test_ntoken1, int)
        assert isinstance(test_ntoken2, int)


class TestQuoraElmo:
    quora = ParaphraseDatasetElmo(para_dataset = 'quora', model_name = 'elmo', num_examples = 20000, 
                    max_length = 40, stop_words_file = './stop_words_en.txt', r1 =0.5, seed = 42, use_synonyms= False) # all of these are seed dependent

    def test_length(self):
        assert len(self.quora) == 16881
    
    def test_id_to_sent(self):
        # sentence tuple for "How do I get rid of my fat rolls?"
        sent = ((259, 73, 112, 120, 260, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261), (259, 101, 112, 260, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261), (259, 74, 260, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261), (259, 104, 102, 117, 260, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261), (259, 115, 106, 101, 260, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261), (259, 112, 103, 260, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261), (259, 110, 122, 260, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261), (259, 103, 98, 117, 260, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261), (259, 115, 112, 109, 109, 116, 260, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261), (259, 64, 260, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        assert (
            self.quora._id_to_sent[423021] == sent
        )

    def test_sent_to_id(self):
        idx = 213590
        tokenized_sentence_tuple = tuple(self.quora._id_to_sent[idx])
        print('id:', id) # TODO: what does this do?
        assert self.quora._sent_to_id[tokenized_sentence_tuple] == idx
    
    def test_paraphrase_sets(self):
        paraphrase_ids = sorted(list(self.quora._paraphrase_sets))[0]
        paraphrase_sets = self.quora._paraphrase_sets
        assert paraphrase_ids == (32, 1101) #[67843, 105500]
        assert (136860, 136861) in paraphrase_sets #harry potter
        assert (105500, 67843) in paraphrase_sets # general contractor
        print(f'sorted(paraphrase_sets)[0]:\t{paraphrase_ids}')      # (136860, 136861) OR (105500, 67843) OR...because it's a set

    def test_para_tuples(self):
        assert sorted(self.quora._para_tuples)[0] == (32, 1101, 3, 2)
        assert (24523, 60274, 10, 12) in self.quora._para_tuples
        assert len(self.quora._para_tuples) == 16_881 # NOTE: 20_062 for BERT likely because of subwords vs. stopwords...

    def test_neg_tuples(self):
        assert self.quora._neg_tuples[0] == (416022, 416023, 4, 11)
        assert len(self.quora._neg_tuples) == 19_643 
    
    def test_token_pair_to_neg_tuples(self):
        # tuple representation of 'Zealand'
        Zealand = (259, 91, 102, 98, 109, 98, 111, 101, 260, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261)
        assert self.quora._token_pair_to_neg_tuples[(Zealand, Zealand)] in [{0, 11653}, {0, 11654}] #{0, 11654} # NOTE: this is system dependent (cluster -> 11653, local -> 11654); it references the index of the list self._neg_tuples; so the index changes based on how the list was created
    
    def test_token_to_sents(self):
        assert len(self.quora._token_to_sents.keys()) == 23_975

    def test_list_types_and_len(self):
        assert type(self.quora._para_tuples) == list
        assert type(self.quora._neg_tuples) == list
        assert len(self.quora) == len(self.quora._para_tuples)

    def test_quora_example_types(self):
        test_sent1, test_sent2, test_nsent1, test_nsent2, test_token1, test_token2, test_ntoken1, test_ntoken2 = self.quora[0]
        assert isinstance(test_sent1, torch.Tensor)
        assert isinstance(test_sent2, torch.Tensor)
        assert isinstance(test_nsent1, torch.Tensor)
        assert isinstance(test_nsent2, torch.Tensor)
        assert isinstance(test_token1, int)
        assert isinstance(test_token2, int)
        assert isinstance(test_ntoken1, int)
        assert isinstance(test_ntoken2, int)

    def test_train_test_split_with_paraphrase(self):
        batch_size = 4
        train_split = 0.8
        np.random.seed(42)
        torch.manual_seed(42)

        train_dl, test_dl = train_test_split(dataset=self.quora, batch_size=batch_size, shuffle=True,
                                              drop_last=False, train_split=train_split)
        dataset_length = len(self.quora) #16881 from test_length() above
        assert len(train_dl) == int(np.floor(np.floor(dataset_length*train_split)/batch_size))
        assert len(test_dl) == int(np.ceil(np.ceil(dataset_length*(1-train_split))/batch_size))

        train_sent1, train_sent2, train_nsent1, train_nsent2, train_token1, train_token2, train_ntoken1, train_ntoken2 = next(iter(train_dl))
        test_sent1, test_sent2, test_nsent1, test_nsent2, test_token1, test_token2, test_ntoken1, test_ntoken2 = next(iter(test_dl))
        assert train_token1[0].item() == 2
        assert train_token2[0].item() == 2
        assert test_token1[0].item() == 18
        assert test_token2[0].item() == 13

        # TODO: check other train_dl and test_dl return tensors
        # TODO: check other train_dl and test_dl return tensors

class TestClassificationDatasets:
    def test_rotten_tomatoes_elmo(self):
        batch_size = 19
        num_examples = 100
        max_length = 173
        train_dataloader, test_dataloader = load_rotten_tomatoes(
            batch_size=batch_size, max_length=max_length,
            num_examples=num_examples, drop_last=False 
        )
        item = next(iter(train_dataloader))
        batch, labels = item
        assert batch.shape == (batch_size, max_length, 50)
        assert labels[0].item() in {0, 1}

    def test_rotten_tomatoes_elmo_full(self):
        # Loads the full RT dataset (num examples is None).
        # Checks the length of both datasets.
        batch_size = 1
        max_length = 16
        num_examples = None
        train_dataloader, test_dataloader = load_rotten_tomatoes(
            batch_size=batch_size, max_length=max_length,
            num_examples=num_examples, drop_last=False 
        )
        assert len(train_dataloader) == 8530
        assert len(test_dataloader) == 1066
    
    def test_qqp_elmo(self):
        batch_size = 7
        num_examples = 19
        max_length = 173
        train_dataloader, test_dataloader = load_qqp(
            batch_size=batch_size, max_length=max_length,
            num_examples=num_examples, drop_last=False
        )
        item = next(iter(train_dataloader))
        s1, s2, labels = item
        # Make sure sentences are the proper shape
        assert s1.shape == (batch_size, max_length, 50)
        assert s2.shape == (batch_size, max_length, 50)
        # make sure all labels in batch are in {0, 1}
        assert torch.logical_or(labels == 0, labels == 1).all()
    
    def test_sst2_elmo(self):
        batch_size = 17
        num_examples = 50
        max_length = 20
        train_dataloader, test_dataloader = load_sst2(
            batch_size=batch_size, max_length=max_length,
            num_examples=num_examples, drop_last=False
        )
        item = next(iter(train_dataloader))
        s1, labels = item
        # Make sure sentences are the proper shape
        assert s1.shape == (batch_size, max_length, 50)
        # make sure all labels in batch are in {0, 1}
        assert torch.logical_or(labels == 0, labels == 1).all()