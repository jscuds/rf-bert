from pathlib import Path
import sys
import torch

from dataloaders import ParaphraseDataset

# Instantiate ParaDataset class

class TestQuora:
    quora = ParaphraseDataset(para_dataset = 'quora', model_name = 'bert-base-uncased', num_examples = 20000, 
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
        paraphrase_ids = sorted(list(self.quora._paraphrase_sets)[0])
        assert paraphrase_ids == [67843, 105500]
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


#TODO: check number ._paraphrase_sets; len(DATA_NAME._pararphrase_sets)/2 == 6526 when count is incremented at top and seed = 42


#TODO: NEED TO CHECK
# _overlap()
# _corrupt()
# _gen_neg_tuple()

# __len__()
# __getitem__() with DataLoader


#TODO: models.py tests