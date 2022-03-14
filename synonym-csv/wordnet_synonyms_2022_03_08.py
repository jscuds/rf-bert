from typing import List, Tuple
import nltk
nltk.download('omw-1.4')
nltk.download('wordnet')

from nltk.corpus import wordnet as wn

def populate_synonyms_antonyms(limit: bool = None) -> Tuple[List[Tuple[str,str]],List[Tuple[str,str]]]:
    """Use wordnet to populate full synonyms and antonyms lists."""
    synonym_pairs = []
    antonym_pairs = []
    for ss in wn.all_synsets():
        word_name = ss.lemmas()[0].name()
        for lemma in ss.lemmas():
            if lemma.name() != word_name:
                synonym_pairs.append((word_name, lemma.name()))
            if lemma.antonyms():
                for ant_lemma in lemma.antonyms():
                    antonym_pairs.append((word_name, ant_lemma.name()))
        if (limit is not None) and (len(synonym_pairs) > limit) and (len(antonym_pairs) > limit): break
    
    # sort by first item in each pair
    synonym_pairs.sort()
    antonym_pairs.sort()
    
    return synonym_pairs, antonym_pairs


def write_pairs_to_csv(filename: str, pair_list: List[Tuple[str,str]]) -> None:
    textfile = open(filename, "w")
    for pair in pair_list:
        if (
            (pair[0] != pair[1])
            and ("_" not in pair[0])
            and ("_" not in pair[1])
        ):
            textfile.write(pair[0] + ',' + pair[1] + "\n")
    textfile.close()

if __name__ == '__main__':

    synonym_pairs, antonym_pairs = populate_synonyms_antonyms(limit=None)


    print('\n\nSynonyms:', synonym_pairs[:10] + ['...'])
    print('\n\nAntonyms:', antonym_pairs[:10] + ['...'])

    write_pairs_to_csv('synonyms_all_synsets_wordnet.csv',synonym_pairs)
    write_pairs_to_csv('antonyms_all_synsets_wordnet.csv',antonym_pairs)



# TODO / IDEAS:
# 0. Make this cleaner if it's used.
# 1. REMOVE DIGITS
# 2. Limit the full synonym list to just reference other overlap tokens already encompassed in the ParaphraseDataset.
#        Ex. bigger/larger instead of _any_ synonymous word in WordNet
# 3. Remove hyphens '-'
# 4. *REACH* ensure synonyms are used in the same part of speech