from typing import Optional

import rowordnet as rwn
import spacy
from mlconjug import mlconjug
from rowordnet import RoWordNet

from util.tools import unique, process_pair, get_cross_synset_pairs, get_synset_pairs


def generate_rwn_synonyms(wordnet: RoWordNet, word: str) -> list:
    """
    Generates all synonyms of a word.
    :param wordnet: An instance of the RoWordNet lexicon.
    :param word: The target word.
    :return: A list of word's synonyms.
    """
    pairs = list()
    synset_ids = wordnet.synsets(word)

    # For each synset that contains the target word
    for synset_id in synset_ids:
        synset = wordnet.synset(synset_id)

        # Append the list of pairs of the current synset and add them the list
        pairs.append(get_synset_pairs(synset))
    return pairs


def generate_rwn_antonyms(wordnet: RoWordNet, word: str) -> list:
    """
    Generates all antonyms of a word.
    :param wordnet: An instance of the RoWordNet lexicon.
    :param word: The target word.
    :return: A list of word's antonyms.
    """
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


def generate_conjugated_pairs(lemma: str, wordnet: RoWordNet, conjugator: mlconjug.Conjugator,
                              valid_verbal_times: list, mode: str) -> list:
    """
    Generates the conjugated pairs for a verb.
    :param lemma: The lemma (infinitive form) of the target word.
    :param wordnet: An instance of the RoWordNet lexicon.
    :param conjugator: An instance of the mlConjug conjugator.
    :param valid_verbal_times: A list of valid verb times.
    :param mode: Specifies whether the function returns the list of antonym pairs or synonym pairs.
    :return: A list of conjugated pairs, according to the constraint specified by mode.
    """

    # Check the mode
    if mode == 'syn':
        relationships = generate_rwn_synonyms(wordnet, lemma)
    elif mode == 'ant':
        relationships = generate_rwn_antonyms(wordnet, lemma)
    else:
        raise ValueError('Invalid Mode')

    pairs = list()
    for relationship in relationships:
        for pair in relationship:
            pairs.append(pair)
    pairs = unique(pairs)

    pairs = list(filter(None, list(map(lambda x: process_pair(x), pairs))))

    conjugated_pairs = list()

    for pair in pairs:
        for time in valid_verbal_times:
            try:
                conjugated_first = conjugate(pair[0], conjugator, time)
                conjugated_second = conjugate(pair[1], conjugator, time)
                if not (conjugated_first is None and conjugated_second is None):
                    conjugated_pairs.append((conjugated_first, conjugated_second))
            except ValueError:
                # Well, no big deal if we are unable to conjugate the pairs. Probably the correct place to "solve"
                # the error of intruders by simply passing to the next conjugation step and filtering only
                # valid elements in the list.
                print(f"Value Error when conjugating {pair}")
                pass

    return conjugated_pairs


def conjugate(lemma: str, conjugator: mlconjug.Conjugator, time: tuple) -> Optional[str]:
    """
    Conjugates a verb.
    :param conjugator: An instance of the mlConjug conjugator.
    :param lemma: The lemma (infinitive form of a verb)
    :param time: A tuple containing information about the time/mode of the requested conjugation.
    :return: The conjugated form, if the conjugator is able to process it, None otherwise.
    """
    # Time is a triplet of time, mode, person (personal verb) -> conjug_info takes 3 indices
    if len(time) == 3:
        try:
            conjugated_form = conjugator.conjugate(lemma).conjug_info[time[0]][time[1]][time[2]]
        except KeyError:
            print(f"Unable to conjugate {lemma} at {time}.")
            return None
    # Time is a tuple of time, mode (impersonal verb) -> conjug_info takes 2 indices (third is default to '')
    elif len(time) == 2:
        try:
            conjugated_form = conjugator.conjugate(lemma).conjug_info[time[0]][time[1]]['']
        except KeyError:
            print(f"Unable to conjugate {lemma} at {time}.")
            return None
    else:
        raise IndexError("Insufficient/Incorrect number of time parameters. Expected either 2 or 3")

    return conjugated_form


def augment_synonym_verbs(lemmas: list, times: list) -> list:
    """
    Computes the list of valid synonym pairs.
    :param lemmas: A list of verb lemmas.
    :param times: A list of verbal times.
    :return: The list of synonym verb pairs generated from the provided lemmas and verbal times.
    """
    wordnet = rwn.RoWordNet()
    conjugator = mlconjug.Conjugator(language='ro')

    # Enforce lemmas vs use language model to lemmatize. First choice should suffice for now.
    # Todo: Compare advantages/disadvantages, maybe implement second
    global_synonym_pairs = list()

    for lemma in lemmas:
        global_synonym_pairs.extend(generate_conjugated_pairs(lemma, wordnet, conjugator, times, 'syn'))

    valid_synonym_pairs = unique(list(
        [pair for pair in global_synonym_pairs if pair[0] is not None and pair[1] is not None]))

    return valid_synonym_pairs


def augment_antonym_verbs(lemmas: list, times: list) -> list:
    """
      Computes the list of valid antonym pairs.
      :param lemmas: A list of verb lemmas.
      :param times: A list of verbal times.
      :return: The list of antonym verb pairs generated from the provided lemmas and verbal times.
    """
    wordnet = rwn.RoWordNet()
    conjugator = mlconjug.Conjugator(language='ro')

    # Enforce lemmas vs use language model to lemmatize. First choice should suffice for now.
    # Todo: Compare advantages/disadvantages, maybe implement second
    global_antonym_pairs = list()

    for lemma in lemmas:
        global_antonym_pairs.extend(generate_conjugated_pairs(lemma, wordnet, conjugator, times, 'ant'))

    valid_antonym_pairs = unique(list(
        [pair for pair in global_antonym_pairs if pair[0] is not None and pair[1] is not None]))

    return valid_antonym_pairs


def valid_verb(token, aux_lemmas: list, excluded_forms: list) -> bool:
    if token.tag_[0] != 'V':
        return False
    if token.lemma_ in aux_lemmas:
        return False
    for ef in excluded_forms:
        if ef in token.tag_:
            return False
    return True


def extract_verb_lemmas(base_langauge_model: str, sentences: list) -> tuple:
    verb_lemmas = list()
    verb_times = list()
    conjugator = mlconjug.Conjugator(language='ro')

    # Manually exclude the auxilliary verbs. If the to-do below is solved, this is no longer required
    auxilliary_verb_lemmas = ['vrea', 'avea', 'fi']
    excluded_verb_forms = ['Vmp--sf', 'Vmp--sm', 'Vmp--pm', 'Vmp--pf']

    # Load the SpaCy Language Model
    lang_model = spacy.load(base_langauge_model)
    for sentence in sentences:
        doc = lang_model(sentence)
        for token in doc:
            # For now this takes all the verbs regardless of their conjugation.
            # Todo: Use https://universaldependencies.org/tagset-conversion/ro-multext-uposf.html
            if valid_verb(token, auxilliary_verb_lemmas, excluded_verb_forms):
                if token.lemma_ not in verb_lemmas:
                    verb_lemmas.append(token.lemma_)

                    # Figure the possible conjugations of the lemma which are equal to the token
                    try:
                        target_conjugations = conjugator.conjugate(token.lemma_)

                        # For each conjugation of the target word
                        for conjugation in target_conjugations.iterate():

                            # Unpack the time, mood, person and value from the tuple
                            time, mood, person, value = conjugation

                            if value == token.text:
                                # Found a possible candidate, append the time/mood/person to the list
                                verb_times.append((time, mood, person))

                    except ValueError:
                        print(f"Unable to conjugate, possibly mistagged verb {token.text}")

    verb_lemmas = unique(verb_lemmas)
    verb_times = unique(verb_times)
    print("Verbs extracted", verb_lemmas)
    print("Verb times extracted:", verb_times)
    return verb_lemmas, verb_times
