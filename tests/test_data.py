import sys
#import pytest
import torch

sys.path.append('../')

from data import ParaDataset

# Instantiate ParaDataset class

quora = ParaDataset(para_dataset = 'quora', model_name = 'bert-base-uncased', num_examples = 20000, 
                    max_length = 40, stop_words_file = '../stop_words_en.txt', r1 =0.5, seed = 42)

# check number ._paraphrase_sets; len(DATA_NAME._pararphrase_sets)/2 == 6526 when count is incremented at top and seed = 42


# print some attribute metrics
print('\n#### PRINT ATTRIBUTE METRICS FOR TESTING ####')
print(f'id_to_sent[213590]:\t{quora._id_to_sent[213590]}')      # [101, 2129, 2079, 1045, 3857, 2026, 6337, 2005, 2327, 1038, 1011, 2816, 1029, 102] + [PAD] to 40 places
print(f'sent_to_id check:\t{quora._sent_to_id[tuple(quora._id_to_sent[213590])]}') # 213590
print(f'paraphrase_sets[0]:\t{(list(quora._paraphrase_sets)[0])}')      # (136860, 136861) OR (105500, 67843) OR...because it's a set
print(f'para_tuples[0]:\t{quora._para_tuples[0]}')                  # tensor([213590, 149174, 6, 6]) 
print(f'len(para_tuples):\t{len(quora._para_tuples)}')                # TODO: CHECK; 17049 colab vs. 17029 local
print(f'neg_tuples[0]:\t{quora._neg_tuples[0]}')                   # tensor([416022, 416023, 18, 4])
print(f'len(neg_tuples):\t{len(quora._neg_tuples)}')                 # TODO: CHECK; 19806 colab vs 19773 local

# token 12952 == '##pal'
print(f'token_pair_to_neg_tuples[##pal,##pal]:\t{quora._token_pair_to_neg_tuples[(12952, 12952)]}')  # TODO CHECK; {11, 2614, 7723, 11841, 12950, 15814, 19206, 19765, 20436, 21679}   
print(f'len(token_to_sents.keys():\t{len(quora._token_to_sents.keys())}')           # 13848
print('#### PRINT ATTRIBUTE METRICS FOR TESTING ####')


assert type(quora._para_tuples) == torch.Tensor
assert type(quora._neg_tuples) == torch.Tensor

# NEED TO CHECK
# _overlap()
# _corrupt()
# _gen_neg_tuple()

# __len__()
# __getitem__() with DataLoader