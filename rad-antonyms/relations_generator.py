import configparser
import errno
import io
import os
import re
import sys
from datetime import datetime
from operator import itemgetter
from typing import Optional

import rowordnet as rwn

from util.tools import unique


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


def process_line(raw_line: str) -> Optional[str]:
    # Replace all reflexive forms
    to_replace = ["[se]", "|se|", "[-și]", "[o]", "|-și|", "|și|", "[-i]", "[i]", "[și]", "a "]
    for sub in to_replace:
        raw_line = raw_line.replace(sub, "")

    # Replace multiple spaces, strip beginning / ending spaces
    processed_line = re.sub('\s{2,}', ' ', raw_line).strip()

    words = processed_line.split(' ')

    # Return the pair as a string "word1 word2"
    # Or the empty string if the words are the same or contain each other, or ar capital nouns
    if len(words) != 2:
        return None
    if words[1] in words[0] or words[0] in words[1]:
        return None
    if words[1][0].isupper() or words[0][0].isupper():
        return None
    return " ".join(words)


def postprocess_pairs(raw_pairs: list, config: SettingConfig) -> list:
    processed_pairs = list()

    for raw_pair in raw_pairs:
        # Preprocess each line
        raw_line = " ".join(raw_pair)
        processed_line = process_line(raw_line)

        # If the processed line is not empty (meaning we have 2 different words separated by a space)
        if processed_line:
            # Split the words
            w1, w2 = processed_line.split(" ")

            # Check if both are in the dictionary
            if w1 in config.vocabulary and w2 in config.vocabulary:
                processed_pairs.append((w1, w2))
    return unique(processed_pairs)


def write_pairs(pairs: list, root_path: str, pos: str, name: str) -> str:
    print(f"Writing {pos} pairs to file")
    dir_path = os.path.join(root_path, pos)

    try:
        os.mkdir(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    constraints_path = os.path.join(dir_path, name + ".txt")

    with io.open(file=constraints_path, mode="w", encoding='utf-8') as out:
        # Wipe the file
        out.truncate(0)

        # Write each pair to the file
        for pair in pairs:
            out.write(f"{pair[0]} {pair[1]}\n")

        out.close()

    return constraints_path


def generate_raw_antonym_pairs(config: SettingConfig) -> dict:
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
            current_literals = synset.literals

            # Iterate outbound relations
            for relation in outbound_relations:

                # Get the synset corresponding to the target of the outbound relation
                target_synset = wn.synset(relation[0])

                # Get the literals in the synset above
                target_literals = target_synset.literals

                # Get all the pairs, sort them by first word to keep set entries unique
                current_iteration_pairs = unique(
                    [tuple(sorted((w1, w2), key=itemgetter(0))) for w1 in current_literals for w2 in target_literals])

                # Add the current set of pairs
                for pair in current_iteration_pairs:
                    pos_pairs.append(pair)

        # Get corresponding key in pos dictionary and add the pair to the resulting dictionary
        for key, value in config.pos.items():
            if value == part_of_speech:
                pairs[key] = unique(pos_pairs)

    # Return the whole dictionary
    print(f"Successfully generated antonym paris @ {datetime.now()}")
    return pairs


def generate_raw_synonym_pairs(config: SettingConfig) -> dict:
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

            literals = synset.literals

            # Get all the pairs, sort them by first word to keep set entries unique
            current_iteration_pairs = unique(
                [tuple(sorted((w1, w2), key=itemgetter(0))) for w1 in literals for w2 in literals if not w1 == w2])

            # Append all pairs from the current PoS to the global set
            for pair in current_iteration_pairs:
                pos_pairs.append(pair)

        # Get corresponding key in pos dictionary and add the pair to the resulting dictionary
        for key, value in config.pos.items():
            if value == part_of_speech:
                pairs[key] = unique(pos_pairs)

    print(f"Successfully generated synonym pairs {datetime.now()}")
    return pairs


def antonyms_pipeline(config: SettingConfig) -> None:
    # raw_synonym_pairs : dict(str, set(pair(str, str))
    raw_antonym_pairs = generate_raw_antonym_pairs(config)
    for pos in config.pos.keys():
        processed_synonym_pairs = postprocess_pairs(raw_antonym_pairs[pos], config)
        write_pairs(processed_synonym_pairs, config.constraints_root_path, pos, "antonyms")


def synonyms_pipeline(config: SettingConfig) -> None:
    # raw_synonym_pairs : dict(str, set(pair(str, str))
    raw_synonym_pairs = generate_raw_synonym_pairs(config)
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
