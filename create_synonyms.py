import nltk
import numpy as np

nltk.download('omw-1.4')
nltk.download('wordnet')
from typing import List, Set, Tuple

from nltk.corpus import wordnet as wn

from dataloaders import ParaphraseDatasetElmo
from utils import elmo_word_decode

NUM_EXAMPLES = 25_000 # None

# Adapted from https://github.com/QData/TextAttack/blob/master/textattack/transformations/word_swaps/word_swap_wordnet.py

def _get_wordnet_synonyms(word: str) -> List[str]:
    """Returns a list containing all synonyms of the given word using WordNet synsets"""
    synonyms = set()
    for syn in wn.synsets(word, lang='eng'):
        for syn_word in syn.lemma_names(lang='eng'):
            if (
                (syn_word != word)
                and ("_" not in syn_word)
            #     and (textattack.shared.utils.is_one_word(syn_word)) #TODO(js) is this `is_one_word` function useful?? 
            ):
                # WordNet can suggest phrases that are joined by '_' but we ignore phrases.
                synonyms.add(syn_word)
    return list(synonyms)


# 1. Create a ParaphraseDatasetElmo object in order to obtain all the overlap tokens.

# js to create quora paraphrase dataset w/ 25k examples or full quora:
quora = ParaphraseDatasetElmo(num_examples=NUM_EXAMPLES, seed=42)


# 2. Obtain the string representations of the overlap tokens using `elmo_word_decode`
quora_token_list = []
for i in  quora._token_to_sents.keys():
    quora_token_list.append(elmo_word_decode(i))

assert len(quora._token_to_sents.keys()) == len(quora_token_list)

if NUM_EXAMPLES == 25_000:
    np.savetxt('25k_tokens.csv',quora_token_list,delimiter=',',fmt='%s')
elif NUM_EXAMPLES is None:
    np.savetxt('full_tokens.csv',quora_token_list,delimiter=',',fmt='%s')


# 3. Create a 2 column csv where column 1 is a list of tokens used in the quora dataset
big_token_list = []
big_synonym_list = []
for token in quora_token_list:
    syn_list = []
    syn_list = _get_wordnet_synonyms(token)

    big_token_list+= [token]*len(syn_list)
    big_synonym_list +=syn_list

assert len(big_token_list) == len(big_synonym_list)


a = np.vstack([big_token_list, big_synonym_list]).T
if NUM_EXAMPLES == 25_000:
    np.savetxt("25k_syn_list.csv", a, delimiter=",", fmt="%s")
elif NUM_EXAMPLES is None:
    np.savetxt("synonyms.csv", a, delimiter=",", fmt="%s")

# TODO / IDEAS:
# 0. Make this cleaner if it's used.
# 1. Limit the full synonym list to just reference other overlap tokens already encompassed in the ParaphraseDataset.
#        Ex. bigger/larger instead of _any_ synonymous word in WordNet
# 2. Remove hyphens '-'
# 3. *REACH* ensure synonyms are used in the same part of speech
