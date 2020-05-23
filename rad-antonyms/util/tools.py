import io
import json
import os
import pathlib
import re
import string
import sys
from datetime import datetime
from math import sqrt
from operator import itemgetter
from os.path import isfile, isdir
from typing import Optional, Union

import numpy as np
from rowordnet import Synset


def compute_vocabulary(root_path: str) -> list:
    """
    Extension over parse_vocabulary_from_file to a folder containing multiple (sub-folders and) such files.
    :param root_path: Path to the root of scenario folders.
    :return: All the words contained in all the sentences in the folders, presented in RASA NLU JSON format.
    """
    vocab = list()
    scenario_folders = [os.path.join(root_path, f) for f in os.listdir(root_path) if isdir(os.path.join(root_path, f))]
    for scenario_folder in scenario_folders:
        # Compute the path for the scenario folder
        files = [os.path.join(scenario_folder, f) for f in os.listdir(scenario_folder) if
                 isfile(os.path.join(scenario_folder, f))]
        for file in files:
            file_vocab = parse_vocabulary_from_file(file)
            vocab = vocab + file_vocab
    return unique(vocab)


def copy_path(src_path: str, dst_path: str, append=True) -> None:
    """
    Feels like re-inventing the wheel.
    :param src_path: Path to the source file.
    :param dst_path: Path to destination file.
    :param append: If false, the file is overwritten, else it is appended to.
    :return:
    """
    with io.open(src_path, "r", encoding="utf-8") as src:
        with io.open(dst_path, "a" if append else "w", encoding="utf-8") as dst:
            for line in src:
                dst.write(line)


def cos_sim(v1: Union[np.ndarray, np.iterable, int, float], v2: Union[np.ndarray, np.iterable, int, float]) -> float:
    """
    Returns the cosine similarity of two vectors, defined as the division between their dot product and the product.
    of their L2 Norms.
    :param v1: The first vector.
    :param v2: The second vector.
    :return: The cosine similarity, as a float.
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def distance(v1: Union[np.ndarray, np.iterable, int, float], v2: Union[np.ndarray, np.iterable, int, float],
             normalised=True) -> float:
    """
    Complement of cosine similarity, the distance between two vectors, optimized as per counterfitting implementation
    https://github.com/nmrksic/counter-fitting/blob/master/counterfitting.py#L204
    :param v1: The first vector.
    :param v2: The second vector.
    :param normalised: If vectors are normalized, skip the division with their norm.
    :return:
    """
    return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) if not normalised else 1 - np.dot(v1, v2)


def extract_sentences_from_dir(root_path: str, constraint: Optional[str]) -> list:
    """
    Extension over extract_sentences_from_file to a folder containing multiple (sub-folders and) such files
    :param root_path: Path to the root directory
    :param constraint: Optionally, a string that has to be contained in the file name to be considered valid
    :return:
    """
    sentences = list()
    for root, _, files in os.walk(root_path, topdown=False):
        for file in files:
            if constraint and constraint not in file:
                continue
            path = os.path.join(root, file)
            sentences.extend(extract_sentences_from_file(path))
    return unique(sentences)


def extract_sentences_from_file(path: str) -> list:
    """
    Extract sentences from a RASA NLU JSON file.
    :param path: Path of the file
    :return: The list of all sentences
    """
    sentences = list()
    with io.open(file=path, mode="r", encoding="utf-8") as input_file:
        content = json.load(input_file)

        data = content["rasa_nlu_data"]

        # Obtain the list of sentences
        common_examples = data["common_examples"]

        for example in common_examples:
            sentences.append(example["text"])

    return unique(sentences)


def get_cross_synset_pairs(src_synset: Synset, dst_synset: Synset) -> list:
    """
    Computes the carthesian product between the elements of two Synsets, generating pairs with a semantic property,
    specified by the outbound relationship that links the two synsets.
    :param src_synset: The first synset.
    :param dst_synset: The second synset.
    :return: The carthesian product, as a list of tuples.
    """
    # Remove phrasal expressions from the literals
    src_literals = remove_phrases(src_synset.literals)
    dst_literals = remove_phrases(dst_synset.literals)

    return unique([tuple(sorted((w1, w2), key=itemgetter(0))) for w1 in src_literals for w2 in dst_literals])


def get_synset_pairs(synset: Synset) -> list:
    """
    Computes the carthesian product between the elements of a single Synset, generating pairs of synonyms.
    :param synset: The first synset.
    :return: The carthesian product, as a list of tuples.
    """
    # Remove phrasal expressions from the literals
    literals = remove_phrases(synset.literals)

    # Generate a list of unique pairs representing the cartesian product of the list of literals of the single synset
    pairs = unique([tuple(sorted((w1, w2), key=itemgetter(0))) for w1 in literals for w2 in literals if not w1 == w2])
    return pairs


def get_time() -> str:
    """
    Computes the current time in a HH:MM format.
    :return:
    """
    return datetime.now().strftime('%H:%M')


def is_venv():
    """
    :return: True if the script is ran inside a VENV, Flase otherwise.
    """
    return hasattr(sys, 'real_prefix') or sys.base_prefix != sys.prefix


def load_constraints(path: str) -> list:
    """
    Loads the constraints specified in a file in the format of "word1 word2" separated by new line.
    :param path: Path to the file containing the constraints.
    :return: The list of unique tuples contained in the file, as well as their inverse pairs.
    """
    # Create a set with all the pairs contained in the file specified by the constraint path
    constraints = list()
    with open(file=path, mode="r", encoding="utf-8") as constraints_file:
        for line in constraints_file:
            w0, w1 = line.replace("\n", "").strip().split(" ")
            constraints.append((w0, w1))
            constraints.append((w1, w0))
    constraints_file.close()
    return unique(constraints)


def load_multiple_constraints(paths: list) -> list:
    """
    Extension over load_constraints to a list of constraint files.
    :param paths: List of paths representing files to be extracted from.
    :return: The list of unique tuples contanined in the file.
    """
    constraints = list()
    for constraint_path in paths:
        current_constraints = load_constraints(constraint_path)
        constraints = constraints + current_constraints
    return unique(constraints)


def load_pairs(path: str) -> list:
    """
    Loads the pairs specified in a file in the format of "word1 word2" separated by new line.
    :param path: Path to the file containing the pairs.
    :return: The list of unique tuples contained in the file, but not their inverse counterpart as opposed to
    load_constraints.
    """
    pairs = list()
    with open(file=path, mode="r", encoding="utf-8") as pairs_file:
        for line in pairs_file:
            w0, w1 = line.replace("\n", "").strip().split(" ")
            pairs.append((w0, w1))
        pairs_file.close()
    return unique(pairs)


def load_vectors(path: str, vocabulary: set) -> (Optional[str], dict):
    """
    Compute the dictionary of word vectors for a language, filtered by its vocabulary.
    :param path: The path to the file containing the word vectors.
    :param vocabulary: The set containing the vocabulary of the language.
    :return: A dictionary where keys are the words that appear in the vocabulary and the values are their embeddings,
    as numpy arrays.
    """
    print(f"Started loading vectors from {path} @ {datetime.now()}")
    print(f"No. of words in vocabulary: {len(vocabulary)}")
    words = dict()
    try:
        with open(file=path, mode="r", encoding="utf-8") as source_file:
            # Get the first line. Check if there's only 2 space-separated strings (hints a dimension)
            dimensions = str(next(source_file))
            if len(dimensions.split(" ")) == 2:
                # We have a dimensions line. Keep it in the variable, continue with the next lines
                pass
            else:
                # We do not have a dimensions line
                line = dimensions.split(' ', 1)
                key = line[0]
                if key in vocabulary:
                    words[key] = np.fromstring(line[1], dtype="float32", sep=' ')
                dimensions = None
            for line in source_file:
                line = line.split(' ', 1)
                key = line[0]
                if key in vocabulary:
                    words[key] = np.fromstring(line[1], dtype="float32", sep=' ')
    except:
        print("Unable to read word vectors, aborting.")
        return None
    print(f"Finished loading a total of {len(words)} vectors @ {datetime.now()}")
    return dimensions, normalise(words)


def load_vectors_novocab(path: str) -> (Optional[str], dict):
    """
    Compute the dictionary of word vectors of a language, without being constrained by a vocabulary.
    :param path: The path to the file containing the word vectors.
    :return: A dictionary where keys are the words and the values are their embeddings,
    as numpy arrays.
    """
    print(f"Started loading vectors from {path} @ {datetime.now()}")
    words = dict()
    try:
        with open(file=path, mode="r", encoding="utf-8") as source_file:
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


def load_vsp_pairs(path: str) -> dict:
    """
    Parses a specific formatted VSP pairs file and returns the dictionary corresponding to the VSP pairs.
    :param path: The path of the file where the dictionary is stored.
    :return: The dictionary.
    """
    vsp_pairs = dict()
    with io.open(file=path, mode="r", encoding="utf-8") as src_file:
        for line in src_file:
            content = line.split(':', 1)
            vsp_pairs[tuple(content[0].split(','))] = content[1]
    return vsp_pairs


def normalise(words: dict) -> dict:
    """
    :param words: A dictionary containing the words in a dictionary mapped to their embeddings
    :return: Normalized (safely, for null vectors) correspondents of each vector.
    """
    for word in words:
        words[word] /= sqrt((words[word] ** 2).sum() + 1e-6)
    return words


def parse_vocabulary_from_file(path: str) -> list:
    """
    :param path: Parth of the vocabulary containing RASA NLU JSONS.
    :return:
    """
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


def partial_gradient(u: Union[np.ndarray, np.iterable, int, float], v: Union[np.ndarray, np.iterable, int, float],
                     normalised=True) -> Union[np.ndarray, np.iterable, int, float]:
    """
    Computes the partial derivative of the cosine distance between the two vectors with respect to the first.
    :param u: The first vector.
    :param v: The second vector.
    :param normalised: True if vectors are normalized. Gradient formula can be simplified if the vectors are normalized.
    :return:
    """
    if normalised:
        return u * np.dot(u, v) - v
    else:
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        return (u * np.dot(u, v) - v * np.power(norm_u, 2)) / (norm_v * np.power(norm_u, 3))


def process_pair(words: tuple) -> Optional[tuple]:
    """
    Processes a pair of words to remove reflexive forms, etc.
    :param words: Pair of words
    :return: The two cleaned forms of the words if the conditions are met, otherwise None.
    """

    # Replace all reflexive forms
    to_replace = ["[se]", "|se|", "[-și]", "[o]", "|-și|", "|și|", "[-i]", "[i]", "[și]", "a "]
    raw_line = "".join(words)
    for sub in to_replace:
        raw_line = raw_line.replace(sub, "")

    # Replace multiple spaces, strip beginning / ending spaces
    processed_line = re.sub('\s{2,}', ' ', raw_line).strip()

    words = processed_line.split(' ')

    # Return the new tuple
    # Or the empty string if the words are the same or contain each other, or ar capital nouns
    if len(words) != 2:
        return None
    if words[1] in words[0] or words[0] in words[1]:
        return None
    if words[1][0].isupper() or words[0][0].isupper():
        return None
    return tuple(words)


def reactivate_venv(root_path: str) -> None:
    """
    Re-activates the virtual environment.
    :param root_path: Root path of the project.
    :return: None
    """
    activate_abs_path = os.path.join(pathlib.Path(root_path).absolute(), "venv", "Scripts")
    os.chdir(activate_abs_path)
    os.subprocess.Popen(["activate.bat"])


def remove_phrases(words: list) -> list:
    """
    Removes phrases from Synsets. Phrases are expressions with multiple words linked with _.
    :param words: The list of words in a synset
    :return: All the words that are not phrases or parts of phrases.
    """
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


def save_dict_to_file(dictionary: dict, dst_path: str) -> None:
    """
    Feels like re-inventing the weel. Good for formatting, though.
    :param dictionary: Dictionary we want to save to a file.
    :param dst_path: The path to the file to be saved.
    :return:
    """
    with io.open(file=dst_path, mode="w", encoding="utf-8") as dst:
        for k, v in dictionary.items():
            dst.write(f"{k} {v}\n")
        dst.close()


def save_pairs_to_file(pairs: list, dst_path: str, append=False) -> None:
    """
    Saves a list of pairs to a file.
    :param pairs: List of pairs to be saved to a file.
    :param dst_path: Path to the destination file.
    :param append: If true, appends to the current file content, otherwise overwrites it.
    :return: No return.
    """
    with io.open(file=dst_path, mode="a" if append else "w", encoding="utf-8") as dst_file:
        for pair in pairs:
            dst_file.write(f"{pair[0]} {pair[1]}\n")
        dst_file.close()


def save_list_to_file(content: list, dst_path: str, append=False) -> None:
    """
    Saves a list (of strings) to file.
    :param content: The list to be saved.
    :param dst_path: The destination path
    :param append: If true, appends to the current file content, otherwise overwrites it.
    :return:
    """
    with io.open(file=dst_path, mode="a" if append else "w", encoding='utf-8') as destination_file:
        for element in content:
            destination_file.write(element + "\n")


def store_vectors(dimens: str, dst_path: str, vectors: dict) -> None:
    print(f"Storing a total of {len(vectors)} vectors in {dst_path} at {datetime.now()}")
    with open(file=dst_path, mode="w", encoding="utf-8") as destination_file:
        if dimens:
            destination_file.write(dimens)
        keys = vectors.keys()
        for key in keys:
            destination_file.write(key + " " + " ".join(map(str, np.round(vectors[key], decimals=4))) + "\n")
    print(f"Finished storing vectors @ {datetime.now()}")


def unique(l: list) -> list:
    return list(dict.fromkeys(l))
