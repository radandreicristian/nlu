from operator import itemgetter

import rowordnet as rwn
from rowordnet import Synset

from util.tools import unique, remove_phrases


def get_cross_synset_pairs(src_synset: Synset, dst_synset: Synset) -> list:
    src_literals = src_synset.literals
    dst_literals = dst_synset.literals
    return unique([tuple(sorted((w1, w2), key=itemgetter(0))) for w1 in src_literals for w2 in dst_literals])


def get_synset_pairs(synset: Synset) -> list:
    literals = remove_phrases(synset.literals)
    return unique([tuple(sorted((w1, w2), key=itemgetter(0))) for w1 in literals for w2 in literals if not w1 == w2])


def generate_rwn_synonyms(word: str) -> list:
    pairs = list()
    wordnet = rwn.RoWordNet()
    synset_ids = wordnet.synsets(word)
    for synset_id in synset_ids:
        # Compute pairs
        synset = wordnet.synset(synset_id)
        pairs.append(get_synset_pairs(synset))
    return pairs


def generate_conjugated_pairs(synonyms: list) -> list:
    pass


def generate_rwn_antonyms(word: str) -> list:
    pairs = []
    wordnet = rwn.RoWordNet()
    synset_ids = wordnet.synsets(word)
    for synset_id in synset_ids:
        pass
    return []


def main():
    wordnet = rwn.RoWordNet()
    # Enforce lemmas vs use language model to lemmatize. First is good for now
    words = ['aprinde']
    valid_time_moods = [
        ('Conditional', 'Conditional perfect', '2'),
        ('Conditional', 'Conditional prezent', '2'),
        ('Conjunctiv', 'Conjunctiv perfect', '2'),
        ('Conjunctiv', 'Conjunctiv prezent', '2'),
        ('Imperativ', 'Imperativ', '2'),
        ('Infinitiv', 'Infinitiv', '2'),
        ('Prezent', 'Prezent', '2'),
        ('Viitor', 'Viitor', '2')
    ]
    for word in words:
        synonyms = generate_rwn_synonyms(word)
        for _ in synonyms:
            print(f"{_}\n")
            pass


if __name__ == '__main__':
    main()
