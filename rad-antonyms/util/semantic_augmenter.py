import rowordnet as rwn
from mlconjug import mlconjug
from rowordnet import RoWordNet

from util.tools import unique, process_pair, get_cross_synset_pairs, get_synset_pairs


def generate_rwn_synonyms(wordnet: RoWordNet, word: str) -> list:
    pairs = list()
    synset_ids = wordnet.synsets(word)

    # For each synset that contains the target word
    for synset_id in synset_ids:
        synset = wordnet.synset(synset_id)

        # Append the list of pairs of the current synset and add them the list
        pairs.append(get_synset_pairs(synset))
    return pairs


def generate_rwn_antonyms(wordnet: RoWordNet, word: str) -> list:
    pairs = list()
    synset_ids = wordnet.synsets(word)

    # For each synset that contains the target word
    for synset_id in synset_ids:
        synset = wordnet.synset(synset_id)

        # Get antonym outbound relations
        antonym_relations = filter(lambda x: x[1] == 'near_antonym', wordnet.outbound_relations(synset_id))

        # For each relation, extend the pairs list with the cross-synset cartesian product
        for relation in antonym_relations:
            target_synset = wordnet.synset(relation[0])
            pairs.append(get_cross_synset_pairs(synset, target_synset))
    return pairs


def generate_conjugated_pairs(word: str, wordnet: RoWordNet, valid_time_moods: list, conjugator: mlconjug.Conjugator,
                              mode: str) -> list:
    # Check the mode
    if mode == 'syn':
        synsets = generate_rwn_synonyms(wordnet, word)
    elif mode == 'ant':
        synsets = generate_rwn_antonyms(wordnet, word)
    else:
        raise ValueError('Invalid Mode')

    pairs = list()
    for synset in synsets:
        for pair in synset:
            pairs.append(pair)
    pairs = unique(pairs)

    conjugated_pairs = list()

    for pair in pairs:
        for time in valid_time_moods:
            try:
                conjugated_first = conjugate(conjugator, pair[0], time)
                conjugated_second = conjugate(conjugator, pair[1], time)
                conjugated_pairs.append((conjugated_first, conjugated_second))
            except ValueError:
                print(f"Missing verb from {pair}")

    return conjugated_pairs


def generate_valid_conjugations() -> list:
    valid_personal_times = [
        ('Conditional', 'Conditional perfect'),
        ('Conditional', 'Conditional prezent'),
        ('Conjunctiv', 'Conjunctiv perfect'),
        ('Conjunctiv', 'Conjunctiv prezent'),
        ('Imperfect', 'Imperfect'),
        ('Mai', 'Mai mult ca perfect'),
        ('Perfect', 'Perfect simplu'),
        ('Perfect', 'Perfect compus'),
        ('Prezent', 'Prezent'),
        ('Viitor', 'Viitor I')
    ]

    valid_persons = ['2s', '2p', '3s', '3p']

    impersonal_valid_times = [
        ('Imperativ', 'Imperativ'),
        ('Infinitiv', 'Infinitiv'),
    ]

    valid_times = unique(
        list([(time[0], time[1], person) for time in valid_personal_times for person in
              valid_persons]))

    valid_times.extend(impersonal_valid_times)

    print(valid_times, sep="\n")
    return valid_times


def conjugate(conjugator: mlconjug.Conjugator, verb: str, time: tuple):
    # Time is a triplet of time, mode, person (personal verb) -> conjug_info takes 3 indices
    if len(time) == 3:
        return conjugator.conjugate(verb).conjug_info[time[0]][time[1]][time[2]]

    # Time is a tuple of time, mode (impersonal verb) -> conjug_info takes 2 indices (third is default to '')
    if len(time) == 2:
        return conjugator.conjugate(verb).conjug_info[time[0]][time[1]]['']


def augment_synonym_verbs() -> list:
    wordnet = rwn.RoWordNet()
    conjugator = mlconjug.Conjugator(language='ro')

    # Enforce lemmas vs use language model to lemmatize. First choice should suffice for now.
    # Todo: Compare advantages/disadvantages, maybe implement second
    words = ['aprinde', 'stinge', 'porni', 'opri', 'crește', 'scădea']
    global_synonym_pairs = list()
    valid_time_moods = generate_valid_conjugations()

    for word in words:
        global_synonym_pairs.extend(generate_conjugated_pairs(word, wordnet, valid_time_moods, conjugator, 'syn'))

    valid_synonym_pairs = unique(list(filter(None, list(map(lambda x: process_pair(x), global_synonym_pairs)))))
    return valid_synonym_pairs


def augment_antonym_verbs() -> list:
    wordnet = rwn.RoWordNet()
    conjugator = mlconjug.Conjugator(language='ro')

    # Enforce lemmas vs use language model to lemmatize. First choice should suffice for now.
    # Todo: Compare advantages/disadvantages, maybe implement second
    words = ['aprinde', 'stinge', 'porni', 'opri', 'crește', 'scădea']
    global_antonym_pairs = list()
    valid_time_moods = generate_valid_conjugations()

    for word in words:
        global_antonym_pairs.extend(generate_conjugated_pairs(word, wordnet, valid_time_moods, conjugator, 'ant'))

    valid_antonym_pairs = unique(list(filter(None, list(map(lambda x: process_pair(x), global_antonym_pairs)))))
    return valid_antonym_pairs
