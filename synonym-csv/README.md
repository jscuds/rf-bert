# `csv` Table of Contents
_last update: 2022-03-09_

* `25k_syn_list.csv` - list of words and their wordnet synonyms created from all overlap words in `quora` paraphrase dataset limited to 25k examples
* `25k_tokens.csv` - list of words created from all overlap words in `quora` paraphrase dataset limited to 25k examples
* `antonyms_all_synsets_wordnet.csv` - antonyms using `wn.all_synsets()`; method dervied from [2022-03-05 BERT Synonym Test.ipynb](https://github.com/jscuds/rf-bert/blob/103171b27cf1742e9fb1b22a05b05dd2d900661f/notebooks/2022-03-05%20BERT%20Synonym%20Test.ipynb)
* `create_synonyms_2022_03_02.py` - running `python create_synonyms_2022_03_02.py` will create:
    * `25k_tokens.csv` and `25k_syn_list.csv` if `NUM_EXAMPLES` global variable in file is set to `25_000`
    * `full_tokens.csv` and `full_syn_list.csv` if `NUM_EXAMPLES` global variable in file is set to `None`
* `full_syn_list.csv` - list of words and their wordnet synonyms created from all overlap words in full `quora` paraphrase dataset
* `full_tokens.csv` - list of words created from all overlap words in full `quora` paraphrase dataset
* `synonyms_all_synsets_wordnet.csv` - synonyms using `wn.all_synsets()`; method dervied from [2022-03-05 BERT Synonym Test.ipynb](https://github.com/jscuds/rf-bert/blob/103171b27cf1742e9fb1b22a05b05dd2d900661f/notebooks/2022-03-05%20BERT%20Synonym%20Test.ipynb)
* `synonyms_shi.csv` - original synonym file from  [Retrofitting Contextualized Word Embeddings with Paraphrases](https://aclanthology.org/D19-1113)
* `wordnet_synonyms_2022_03_08.py` - running `python wordnet_synonyms_2022_03_08.py` will create:
    * `synonyms_all_synsets_wordnet.csv` and `antonyms_all_synsets_wordnet.csv`