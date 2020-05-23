import configparser
import errno
import os
import sys
from datetime import datetime

import rowordnet as rwn

from util.tools import unique, remove_phrases, process_pair, get_cross_synset_pairs, get_synset_pairs, \
    save_pairs_to_file


class SettingConfig:

    def __init__(self, config_path):

        print("Initializing configuration for relations generator...")
        # Read the config file
        self.config = configparser.RawConfigParser()
        try:
            self.config.read(config_path)
        except OSError:
            print("Unable to read config, aborting.")
            return

        # Load and split the POS parameter into a list

        # Create a mapping from each possible POS value in the config to the corresponding rwn component
        mapping = {"verb": rwn.Synset.Pos.VERB,
                   "noun": rwn.Synset.Pos.NOUN,
                   "adverb": rwn.Synset.Pos.ADVERB,
                   "adjective": rwn.Synset.Pos.ADJECTIVE}

        # Keep in the map of PoS : Rwn.Pos only the specified parts of speech
        self.pos = mapping

        # Load the root of the folders containing constraints
        self.constraints_root_path = self.config.get("paths", "CONSTRAINTS_ROOT_PATH")

        vocab_path = self.config.get("paths", "VOCAB_PATH")

        self.vocabulary = list()
        with open(file=vocab_path, mode="r", encoding="utf-8") as vocab_file:
            for line in vocab_file:
                self.vocabulary.append(line.strip())
        self.vocabulary = set(unique(self.vocabulary))
        print(f"Loaded {len(self.vocabulary)} words from {vocab_path}")
        print(
            f"Finished initializing config for relations generator. Will output results to {self.constraints_root_path}")


def postprocess_pairs(raw_pairs: list, config: SettingConfig) -> list:
    """
    Processes a list of pairs to remove reflexive forms, pairs with duplicate elements, proper nouns, etc.
    :param raw_pairs: List of tuples representing the initial pairs.
    :param config: Configuration of the current run.
    :return: The filtered list of pairs from the initial pairs.
    """
    processed_pairs = list()

    for raw_pair in raw_pairs:
        # Preprocess each line

        processed_line = process_pair(raw_pair)

        # If the processed line is not empty (meaning we have 2 different words separated by a space)
        if processed_line:
            # Split the words
            w1, w2 = processed_line

            # Check if both are in the dictionary
            if w1 in config.vocabulary and w2 in config.vocabulary:
                processed_pairs.append((w1, w2))
    return unique(processed_pairs)


def write_pairs(pairs: list, root_path: str, pos: str, constraint_type: str) -> None:
    """
    Computes the location where the pairs should be stored based on their PoS and constraint type and writes them there.
    :param pairs: Linguistic constraints, as pairs.
    :param root_path: Root to the location where constraint files are saved.
    :param pos: Part of speech.
    :param constraint_type: Category of constraints. Synonym or antonym are accepted values, but feel free to use
    whatever fits.
    :return: None.
    """
    print(f"Writing {pos} pairs to file")
    dir_path = os.path.join(root_path, pos)

    try:
        os.mkdir(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    constraints_path = os.path.join(dir_path, constraint_type + ".txt")

    save_pairs_to_file(pairs, constraints_path)


def generate_antonym_pairs(config: SettingConfig) -> dict:
    """
    Generates antonym pairs from RoWordNet.
    :param config: Configuration of the current run.
    :return: A dictionary where keys are strings representing parts of speech and values are lists of pairs
    corresponding to synonyms / antonyms from that category.
    """
    print(f"Generating initial antonym pairs from RoWordNet @ {datetime.now()}")
    wn = rwn.RoWordNet()

    # Create the output dictionary that will be of type dict(str : set(pair(str, str)) where the key is
    # the PoS and the value is a set of pairs of words of PoS specified by the key
    pairs = dict()

    # Iterate over the selected parts of speech
    for part_of_speech in config.pos.values():

        pos_pairs = list()

        # Return all synsets corresponding to the PoS
        synset_ids = wn.synsets(pos=part_of_speech)

        # Iterate all the synsets for the current PoS
        for synset_id in synset_ids:

            # Get the synset object specified by synset_id
            synset = wn.synset(synset_id)

            # Get the outbound relations of type antonym from
            outbound_relations = filter(lambda x: x[1] == 'near_antonym', wn.outbound_relations(synset_id))

            # Get the literals
            current_literals = remove_phrases(synset.literals)

            # Iterate outbound relations
            for relation in outbound_relations:
                # Get the synset corresponding to the target of the outbound relation
                target_synset = wn.synset(relation[0])

                # Get all the pairs, sort them by first word to keep set entries unique
                current_iteration_pairs = get_cross_synset_pairs(synset, target_synset)

                # Add the current set of pairs
                pos_pairs.extend(current_iteration_pairs)

        # Get corresponding key in pos dictionary and add the pair to the resulting dictionary
        for key, value in config.pos.items():
            if value == part_of_speech:
                pairs[key] = unique(pos_pairs)

    # Return the whole dictionary
    print(f"Successfully generated antonym paris @ {datetime.now()}")
    return pairs


def generate_synonym_pairs(config: SettingConfig) -> dict:
    """
    Generates synonym pairs from RoWordNet.
    :param config: Configuration of the current run.
    :return: A dictionary where keys are strings representing parts of speech and values are lists of pairs
    corresponding to synonyms / antonyms from that category.
    """
    print(f"Generating initial synonym pairs from RoWordNet @ {datetime.now()}")
    wn = rwn.RoWordNet()

    # Create the output dictionary that will be of type dict(str : set(pair(str, str)) where the key is
    # the PoS and the value is a set of pairs of words of PoS specified by the key
    pairs = dict()

    # Iterate over the selected parts of speech
    for part_of_speech in config.pos.values():

        pos_pairs = list()

        # Return all synsets corresponding to the PoS
        synset_ids = wn.synsets(pos=part_of_speech)

        # Iterate all the synsets for the current PoS
        for synset_id in synset_ids:
            # Get the synset object specified by synset_id
            synset = wn.synset(synset_id)

            # Get all the pairs, sort them by first word to keep set entries unique
            current_iteration_pairs = get_synset_pairs(synset)

            # Append all pairs from the current PoS to the global set
            pos_pairs.extend(current_iteration_pairs)

        # Get corresponding key in pos dictionary and add the pair to the resulting dictionary
        for key, value in config.pos.items():
            if value == part_of_speech:
                pairs[key] = unique(pos_pairs)

    print(f"Successfully generated synonym pairs {datetime.now()}")
    return pairs


def antonyms_pipeline(config: SettingConfig) -> None:
    """
    Synonyms generation pipeline driver function.
    :param config: Configuration of the current run.
    :return: None
    """
    raw_antonym_pairs = generate_antonym_pairs(config)
    for pos in config.pos.keys():
        processed_synonym_pairs = postprocess_pairs(raw_antonym_pairs[pos], config)
        write_pairs(processed_synonym_pairs, config.constraints_root_path, pos, "antonyms")


def synonyms_pipeline(config: SettingConfig) -> None:
    """
    Synonym generation pipeline driver function.
    :param config: Configuration of the current run.
    :return: None
    """
    raw_synonym_pairs = generate_synonym_pairs(config)
    for pos in config.pos.keys():
        processed_synonym_pairs = postprocess_pairs(raw_synonym_pairs[pos], config)
        write_pairs(processed_synonym_pairs, config.constraints_root_path, pos, "synonyms")


def main():
    try:
        config_filepath = sys.argv[1]
    except IndexError:
        print("\nUsing the default config file: parameteres.cfg")
        config_filepath = "parameters.cfg"
    config = SettingConfig(config_filepath)
    synonyms_pipeline(config)
    antonyms_pipeline(config)


if __name__ == '__main__':
    main()
