import io
import json
import os
import re
import string
from datetime import datetime
from math import sqrt
from os.path import isfile, isdir
from typing import Optional

import gensim.models as gs
import numpy as np


def load_vectors(source_path: str, vocab: set) -> (Optional[str], dict):
    print(f"Started loading vectors from {source_path} @ {datetime.now()}")
    print(f"No. of words in vocabulary: {len(vocab)}")
    words = dict()
    try:
        with open(file=source_path, mode="r", encoding="utf-8") as source_file:
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
    except OSError:
        print("Unable to read word vectors, aborting.")
        return {}
    print(f"Finished loading a total of {len(words)} vectors @ {datetime.now()}")
    return dimensions, normalise(words)


def load_vectors_novocab(source_path: str) -> (Optional[str], dict):
    print(f"Started loading vectors from {source_path} @ {datetime.now()}")
    words = dict()
    try:
        with open(file=source_path, mode="r", encoding="utf-8") as source_file:
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


def store_vectors(dimens: str, destination_path: str, vectors: dict) -> None:
    print(f"Storing a total of {len(vectors)} counter-fitted vectors in {destination_path} @ {datetime.now()}")
    with open(file=destination_path, mode="w", encoding="utf-8") as destination_file:
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


def partial_gradient(u, v, normalised=True):
    # Partial gradient, optimized as per counterfitting implementation
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


def parse_vocabulary_from_file(file_path: str) -> set:
    with io.open(file=file_path, mode="r", encoding='utf-8') as file:
        content = json.load(file)

        # Obtain the wrapped object
        data = content["rasa_nlu_data"]

        # Obtain the list of sentences
        common_examples = data["common_examples"]

        vocab = set()

        # For each example, parse its text and return the list containing all the words in the sentence
        for example in common_examples:
            sentence = example['text']
            punctuation = string.punctuation.replace("-", "")
            new_sentence = re.sub('[' + punctuation + ']', '', sentence)
            vocab = vocab | set(new_sentence.split())

        file.close()
    return vocab


def compute_vocabulary(base_path: str) -> set:
    vocab = set()
    scenario_folders = [os.path.join(base_path, f) for f in os.listdir(base_path) if isdir(os.path.join(base_path, f))]
    for scenario_folder in scenario_folders:
        # Compute the path for the scenario folder
        files = [os.path.join(scenario_folder, f) for f in os.listdir(scenario_folder) if
                 isfile(os.path.join(scenario_folder, f))]
        for file in files:
            file_vocab = parse_vocabulary_from_file(file)
            vocab = vocab | file_vocab
    return vocab


def save_vocabulary(vocabulary: set, destination_path: str) -> None:
    with io.open(file=destination_path, mode="w", encoding='utf-8') as destination_file:
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


def load_constraints(constraints_path: str) -> set:
    # Create a set with all the pairs contained in the file specified by the constraint path
    constraints = set()
    with open(file=constraints_path, mode="r", encoding="utf-8") as constraints_file:
        for line in constraints_file:
            w0, w1 = line.replace("\n", "").split(" ")
            constraints.add((w0, w1))
            constraints.add((w1, w0))
    constraints_file.close()
    return constraints


def load_multiple_constraints(constraint_paths: list) -> set:
    constraints = set()
    for constraint_path in constraint_paths:
        current_constraints = load_constraints(constraint_path)
        constraints = constraints | current_constraints
    return constraints


def compute_set_difference(generated_constraints_path: str, augmented_constraints_path: str) -> set:
    og_pairs = set()
    aug_pairs = set()
    with io.open(file=generated_constraints_path, mode="r", encoding="utf-8") as og_pairs_file:
        for line in og_pairs_file.readlines():
            w0, w1 = line.split(" ")
            og_pairs.add(tuple(sorted((w0, w1))))

    with io.open(file=augmented_constraints_path, mode="r", encoding="utf-8") as aug_pairs_file:
        for line in aug_pairs_file.readlines():
            w0, w1 = line.split(" ")
            aug_pairs.add(tuple(sorted((w0, w1))))

    print(len(aug_pairs))
    print(len(og_pairs))
    return aug_pairs.difference(og_pairs)



def extract_sentences(datasets_root_path: str) -> set:
    sentences = set()

    for root, _, files in os.walk(datasets_root_path, topdown=False):
        for file in files:
            path = os.path.join(root, file)
            with io.open(path, "r", encoding="utf-8") as input_file:
                # Load the JSON content
                content = json.load(input_file)

                # Obtain the wrapped object
                data = content["rasa_nlu_data"]

                # Obtain the list of sentences
                common_examples = data["common_examples"]

                for example in common_examples:
                    sentences.add(example["text"])
    return sentences


if __name__ == "__main__":
    diff = compute_set_difference("../lang/constraints/verb/antonyms.txt", "../lang/constraints/verb/antonyms_aug.txt")
    print(diff)
