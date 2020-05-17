import io
import json
import os
import re
import string
from datetime import datetime
from math import sqrt
from operator import itemgetter
from os.path import isfile, isdir
from typing import Optional

import gensim.models as gs
import numpy as np
from rowordnet import Synset


def unique(l: list) -> list:
    return list(dict.fromkeys(l))


def process_pair(words: tuple) -> Optional[tuple]:
    # Replace all reflexive forms
    to_replace = ["[se]", "|se|", "[-și]", "[o]", "|-și|", "|și|", "[-i]", "[i]", "[și]", "a "]
    raw_line = " ".join(words)
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
    return tuple(words)


def remove_phrases(words: list) -> list:
    # We need to sanitize synsets from phrasal expressions that contain "_"
    phrasal_expression_slices = set()
    phrasal_expressions = set()

    # Get all phrasal expressions (that contain '_')
    for word in words:
        if '_' in word:
            split_word = word.split("_")
            for w in split_word:
                phrasal_expression_slices.add(w)
            phrasal_expressions.add(word)

    valid_members = list()
    # Get all the words that are in the synset but not part of the phrasal expression:
    for word in words:
        if word not in phrasal_expression_slices and word not in phrasal_expressions:
            valid_members.append(word)
    return valid_members


def load_vectors(src_path: str, vocab: set) -> (Optional[str], dict):
    print(f"Started loading vectors from {src_path} @ {datetime.now()}")
    print(f"No. of words in vocabulary: {len(vocab)}")
    words = dict()
    try:
        with open(file=src_path, mode="r", encoding="utf-8") as source_file:
            # Get the first line. Check if there's only 2 space-separated strings (hints a dimension)
            dimensions = str(next(source_file))
            if len(dimensions.split(" ")) == 2:
                # We have a dimensions line. Keep it in the variable, continue with the next lines
                pass
            else:
                # We do not have a dimensions line
                line = dimensions.split(' ', 1)
                key = line[0]
                if key in vocab:
                    words[key] = np.fromstring(line[1], dtype="float32", sep=' ')
                dimensions = None
            for line in source_file:
                line = line.split(' ', 1)
                key = line[0]
                if key in vocab:
                    words[key] = np.fromstring(line[1], dtype="float32", sep=' ')
    except:
        print("Unable to read word vectors, aborting.")
        return {}
    print(f"Finished loading a total of {len(words)} vectors @ {datetime.now()}")
    return dimensions, normalise(words)


def load_vectors_novocab(src_path: str) -> (Optional[str], dict):
    print(f"Started loading vectors from {src_path} @ {datetime.now()}")
    words = dict()
    try:
        with open(file=src_path, mode="r", encoding="utf-8") as source_file:
            # Get the first line. Check if there's only 2 space-separated strings (hints a dimension)
            dimensions = str(next(source_file))
            if len(dimensions.split(" ")) == 2:
                # We have a dimensions line. Keep it in the variable, continue with the next lines
                pass
            else:
                # We do not have a dimensions line
                line = dimensions.split(' ', 1)
                key = line[0]
                words[key] = np.fromstring(line[1], dtype="float32", sep=' ')
                dimensions = None
            for line in source_file:
                line = line.split(' ', 1)
                key = line[0]
                words[key] = np.fromstring(line[1], dtype="float32", sep=' ')
    except OSError:
        print("Unable to read word vectors, aborting.")
        return {}
    print(f"Finished loading a total of {len(words)} vectors @ {datetime.now()}")
    return dimensions, normalise(words)


def store_vectors(dimens: str, dst_path: str, vectors: dict) -> None:
    print(f"Storing a total of {len(vectors)} counter-fitted vectors in {dst_path} @ {datetime.now()}")
    with open(file=dst_path, mode="w", encoding="utf-8") as destination_file:
        if dimens:
            destination_file.write(dimens)
        keys = vectors.keys()
        for key in keys:
            destination_file.write(key + " " + " ".join(map(str, np.round(vectors[key], decimals=4))) + "\n")
    print(f"Finished storing vectors @ {datetime.now()}")


def normalise(words: dict) -> dict:
    # Safe norm of a dictionary
    for word in words:
        words[word] /= sqrt((words[word] ** 2).sum() + 1e-6)
    return words


def distance(v1, v2, normalised=True):
    # Distance between two vectors, optimized as per counterfitting implementation
    return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) if not normalised else 1 - np.dot(v1, v2)


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def partial_gradient(u, v, normalised=True):
    # Computes partial derivative of the cosine distance with respect to u
    if normalised:
        return u * np.dot(u, v) - v
    else:
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        return (u * np.dot(u, v) - v * np.power(norm_u, 2)) / (norm_v * np.power(norm_u, 3))


def compute_dictionary_difference(initial_vectors: dict, counterfit_vectors: dict) -> dict:
    # Make sure they are the same length
    assert len(initial_vectors) == len(counterfit_vectors)
    difference = dict()
    for key in initial_vectors.keys():
        if not np.array_equal(initial_vectors.get(key), counterfit_vectors.get(key)):
            difference[key] = counterfit_vectors.get(key)
    print(f"Total of different word vectors: {len(difference)}")
    return difference


def convert_vec_to_binary(vec_path: str, bin_path: str) -> None:
    model = gs.KeyedVectors.load_word2vec_format(fname=vec_path, binary=False)
    model.save_word2vec_format(fname=bin_path, binary=True)


def parse_vocabulary_from_file(path: str) -> list:
    with io.open(file=path, mode="r", encoding='utf-8') as file:
        content = json.load(file)

        # Obtain the wrapped object
        data = content["rasa_nlu_data"]

        # Obtain the list of sentences
        common_examples = data["common_examples"]

        vocab = list

        # For each example, parse its text and return the list containing all the words in the sentence
        for example in common_examples:
            sentence = example['text']
            punctuation = string.punctuation.replace("-", "")
            new_sentence = re.sub('[' + punctuation + ']', '', sentence)
            vocab = vocab + unique(new_sentence.split())

        file.close()
    return unique(vocab)


def compute_vocabulary(path: str) -> list:
    vocab = list()
    scenario_folders = [os.path.join(path, f) for f in os.listdir(path) if isdir(os.path.join(path, f))]
    for scenario_folder in scenario_folders:
        # Compute the path for the scenario folder
        files = [os.path.join(scenario_folder, f) for f in os.listdir(scenario_folder) if
                 isfile(os.path.join(scenario_folder, f))]
        for file in files:
            file_vocab = parse_vocabulary_from_file(file)
            vocab = vocab + file_vocab
    return unique(vocab)


def save_vocabulary(vocabulary: list, dst_path: str) -> None:
    with io.open(file=dst_path, mode="w", encoding='utf-8') as destination_file:
        for word in vocabulary:
            destination_file.write(word + "\n")


def copy_path(src_path, dst_path, append=True):
    with io.open(src_path, "r", encoding="utf-8") as src:
        with io.open(dst_path, "a" if append else "w", encoding="utf-8") as dst:
            dst.writelines([l for l in src.readlines()])


def save_dict_to_file(dictionary: dict, dst_path: str) -> None:
    with io.open(file=dst_path, mode="w", encoding="utf-8") as dst:
        for k, v in dictionary.items():
            dst.write(f"{k} {v}\n")
        dst.close()


def load_constraints(constraints_path: str) -> list:
    # Create a set with all the pairs contained in the file specified by the constraint path
    constraints = list()
    with open(file=constraints_path, mode="r", encoding="utf-8") as constraints_file:
        for line in constraints_file:
            w0, w1 = line.replace("\n", "").strip().split(" ")
            constraints.append((w0, w1))
            constraints.append((w1, w0))
    constraints_file.close()
    return unique(constraints)


def load_pairs(path: str) -> list:
    pairs = list()
    with open(file=path, mode="r", encoding="utf-8") as pairs_file:
        for line in pairs_file:
            w0, w1 = line.replace("\n", "").strip().split(" ")
            pairs.append((w0, w1))
        pairs_file.close()
    return unique(pairs)


def load_multiple_constraints(path: list) -> list:
    constraints = list()
    for constraint_path in path:
        current_constraints = load_constraints(constraint_path)
        constraints = constraints + current_constraints
    return unique(constraints)


def compute_set_difference(generated_constraints_path: str, augmented_constraints_path: str) -> list:
    og_pairs = list()
    aug_pairs = list()
    with io.open(file=generated_constraints_path, mode="r", encoding="utf-8") as og_pairs_file:
        for line in og_pairs_file.readlines():
            w0, w1 = line.split(" ")
            og_pairs.append(tuple(sorted((w0, w1))))

    with io.open(file=augmented_constraints_path, mode="r", encoding="utf-8") as aug_pairs_file:
        for line in aug_pairs_file.readlines():
            w0, w1 = line.split(" ")
            aug_pairs.append(tuple(sorted((w0, w1))))

    print(len(aug_pairs))
    print(len(og_pairs))
    # TODO: compute list difference
    return []
    # return unique(aug_pairs.difference(og_pairs))


def _extract_sentences(root_path: str, constraint: Optional[str]) -> list:
    sentences = list()

    for root, _, files in os.walk(root_path, topdown=False):
        for file in files:
            if constraint and constraint not in file:
                continue
            path = os.path.join(root, file)
            with io.open(path, "r", encoding="utf-8") as input_file:
                # Load the JSON content
                content = json.load(input_file)

                # Obtain the wrapped object
                data = content["rasa_nlu_data"]

                # Obtain the list of sentences
                common_examples = data["common_examples"]

                for example in common_examples:
                    sentences.append(example["text"])
    return unique(sentences)


def extract_all_sentences(root_path: str) -> list:
    return _extract_sentences(root_path, None)


def extract_train_sentences(root_path: str) -> list:
    return _extract_sentences(root_path, "train")


def extract_test_sentences(root_path: str) -> list:
    return _extract_sentences(root_path, "test")


def get_cross_synset_pairs(src_synset: Synset, dst_synset: Synset) -> list:
    # Remove phrasal expressions from the literals
    src_literals = remove_phrases(src_synset.literals)
    dst_literals = remove_phrases(dst_synset.literals)

    # Generates a list of unique pairs representing the cartesian product of the list of literals of the two synsets
    return unique([tuple(sorted((w1, w2), key=itemgetter(0))) for w1 in src_literals for w2 in dst_literals])


def get_synset_pairs(synset: Synset) -> list:
    # Remove phrasal expressions from the literals
    literals = remove_phrases(synset.literals)

    # Generate a list of unique pairs representing the cartesian product of the list of literals of the single synset
    pairs = unique([tuple(sorted((w1, w2), key=itemgetter(0))) for w1 in literals for w2 in literals if not w1 == w2])
    return pairs


def append_constraints_to_file(constraints: list, src_path: str):
    with io.open(file=src_path, mode="a", encoding="utf-8") as src_file:
        for constraint in constraints:
            if type(constraint) is not tuple:
                raise TypeError('List of tuples expected.')
            line = f"{constraint[0]} {constraint[1]}\n"
            src_file.write(line)
        src_file.close()
