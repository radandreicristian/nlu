import configparser
import gc
import os.path
import random
import sys
from copy import deepcopy
from datetime import datetime

import numpy as np

import util.tools as to
from util.comparator import Comparator


class SettingConfig:
    """
    Class that encapsulates all the parameters and variables required for a counterfitting run
    """

    def __init__(self, config_path):

        # Read the config file
        self.config = configparser.RawConfigParser()
        try:
            self.config.read(config_path)
        except OSError:
            print("Unable to read config, aborting.")
            return

        # Read the word vectors path from the config
        self.input_vectors_path = self.config.get("paths", "VEC_PATH")

        # Read the vocabulary mode (all words or just dataset words) from the config
        self.vocab_mode = self.config.get("settings", "VOCABULARY")
        if self.vocab_mode == 'all':
            self.diacritics = 'True'
            vocab_path = self.config.get("paths", "VOCAB_PATH")
        elif self.vocab_mode == 'small':
            self.diacritics = self.config.get("settings", "DIACRITICS")
            if self.diacritics == 'True':
                vocab_path = self.config.get("paths", "VOCAB_PATH_DATASET_DIAC")
            else:
                vocab_path = self.config.get("paths", "VOCAB_PATH_DATASET_NODIAC")
        else:
            print('Wrong value for parameter VOCABULARY in config. Exiting')
            return

        synonym_paths = list()
        antonym_paths = list()

        # Read the root path of the constraints files
        constraints_root_path = self.config.get("paths", "CONSTRAINTS_ROOT_PATH")

        # Read the PoS variable
        self.parts_of_speech = self.config.get("settings", "POS").replace("[", "").replace("]", "").replace(" ",
                                                                                                            "").split(
            ",")

        # Append antonyms and synonyms of each selected PoS from their respective folder
        for part_of_speech in self.parts_of_speech:
            antonym_paths.append(os.path.join(constraints_root_path, part_of_speech, "antonyms.txt"))
            synonym_paths.append(os.path.join(constraints_root_path, part_of_speech, "synonyms.txt"))

        self.synonyms = to.load_multiple_constraints(synonym_paths)
        self.antonyms = to.load_multiple_constraints(antonym_paths)

        # Read and parse the mode (whether to include synonyms, antonyms or VSP pairs in the current run)
        mode = self.config.get("settings", "MODE").replace("[", "").replace("]", "").replace(" ", "").split(",")

        vocab = list()
        with open(file=vocab_path, mode="r", encoding="utf-8") as vocab_file:
            for line in vocab_file:
                vocab.append(line.strip())

        # Add augmented words from synonyms list to the vocab
        for pair in self.synonyms:
            if pair[0] in vocab or pair[1] in vocab:
                vocab.append(pair[0])
                vocab.append(pair[1])

        # Add augmented words from antonym lists to the vocab
        for pair in self.antonyms:
            if pair[0] in vocab or pair[1] in vocab:
                vocab.append(pair[0])
                vocab.append(pair[1])

        vocab = to.unique(vocab)

        # Load the word vectors
        dimensions, self.vectors = to.load_vectors(self.input_vectors_path, set(vocab))

        # Return if vectors were not successfully loaded
        if not self.vectors:
            print("Unable to load initial vectors")
            return

        self.output_vectors_path = self.config.get("paths", "CF_VEC_PATH").split(".")[
                                       0] + f"_{str(datetime.timestamp(datetime.now())).split('.')[0]}.vec"

        # The vocabulary contains the keys of vectors successfully loaded by the initial vocabulary: Words in the
        # initial vocabulary with no corresponding vector are skipped
        self.vocabulary = to.unique(self.vectors.keys())

        self.dimensions = f"{len(self.vocabulary)} {dimensions.split(' ')[1]}"

        # Load synonym and antonym pairs from the paths specified
        self.mode = mode
        self.vsp_path = self.config.get("paths", "VSP_PAIRS_VERB_PATH")

        # Read the hyperparameters of our run
        self.hyper_k1 = self.config.getfloat("hyperparameters", "hyper_k1")
        self.hyper_k2 = self.config.getfloat("hyperparameters", "hyper_k2")
        self.hyper_k3 = self.config.getfloat("hyperparameters", "hyper_k3")
        self.sgd_iters = self.config.getint("hyperparameters", "sgd_iters")

        self.delta = self.config.getfloat("hyperparameters", "delta")
        self.gamma = self.config.getfloat("hyperparameters", "gamma")
        self.rho = self.config.getfloat("hyperparameters", "rho")
        print(
            f"Initialized counterfitting settings. Vocab path: {vocab_path}, PoS paths: {self.parts_of_speech},"
            f" Mode: {self.mode}, diacritics: {self.diacritics}."
            f" Hyperpameters: {self.hyperparams_tostring()}")

    def init_comparator(self, config_path: str, counterfit_vectors: dict) -> None:
        self.comparator = Comparator(config_path, original_vectors=self.vectors, counterfit_vectors=counterfit_vectors,
                                     mode=self.mode, pos=self.parts_of_speech,
                                     hyperparameters=self.hyperparams_tostring(), diacritics=self.diacritics,
                                     vocabulary=self.vocab_mode, original_vectors_path=self.input_vectors_path,
                                     counterfit_vectors_path=self.output_vectors_path)

    def hyperparams_tostring(self) -> str:
        return (f"k1={self.hyper_k1}, k2={self.hyper_k2}, k3={self.hyper_k3}"
                f" delta={self.delta}, gamma={self.gamma}, rho={self.rho}, sgd_iters={self.sgd_iters}")


def compute_vsp_pairs(vectors: dict, vocab: list, config: SettingConfig) -> dict:
    print(f"Computing VSP pairs @ {datetime.now()}")
    vsp_pairs = dict()
    if not config.rho:
        rho = 0.2
    else:
        rho = config.rho
    th = 1 - rho
    vocab = list(vocab)
    words_count = len(vocab)

    step_size = 1000
    vec_size = random.choice(list(vectors.values())).shape[0]

    # List of ranges of step size
    ranges = list()

    left_range_limit = 0
    while left_range_limit < words_count:
        # Create tuple of left range -> right range (min between nr of words (maximum) or left limit + step)
        current_range = (left_range_limit, min(words_count, left_range_limit + step_size))
        ranges.append(current_range)
        left_range_limit += step_size

    range_count = len(ranges)
    for left_range in range(range_count):
        for right_range in range(left_range, range_count):
            print(f"LR: {left_range}/{range_count}. RR: {right_range}/{range_count}")
            left_translation = ranges[left_range][0]
            right_translation = ranges[right_range][0]

            vecs_left = np.zeros((step_size, vec_size), dtype="float32")
            vecs_right = np.zeros((step_size, vec_size), dtype="float32")

            full_left_range = range(ranges[left_range][0], ranges[left_range][1])
            full_right_range = range(ranges[right_range][0], ranges[right_range][1])

            for index in full_left_range:
                vecs_left[index - left_translation, :] = vectors[vocab[index]]

            for index in full_right_range:
                vecs_right[index - right_translation, :] = vectors[vocab[index]]

            dot_product = vecs_left.dot(vecs_right.T)
            indices = np.where(dot_product >= th)

            pairs_count = indices[0].shape[0]
            left_indices = indices[0]
            right_indices = indices[1]

            for index in range(0, pairs_count):
                left_word = vocab[left_translation + left_indices[index]]
                right_word = vocab[right_translation + right_indices[index]]

                if left_word != right_word:
                    score = 1 - dot_product[left_indices[index], right_indices[index]]
                    vsp_pairs[(left_word, right_word)] = score
                    vsp_pairs[(right_word, left_word)] = score
        # Perform a garbage collection operation at the end of a left range iteration.
        # If performed inside the inner for -> Significant impact on time, if outside impact on memory
        # TODO: Figure which is the more appropriate one
        gc.collect()
    print(f"Computed VSP pairs @ {datetime.now()}")
    to.save_dict_to_file(vsp_pairs, config.vsp_path)
    return vsp_pairs


def _sgd_step_ant(antonym_pairs: list, enriched_vectors: dict, config: SettingConfig, gradient_updates: dict,
                  update_count: dict) -> (dict, dict):
    # For each antonym pair
    vocab = set(config.vocabulary)
    for (w0, w1) in antonym_pairs:

        # Extra check for reduced vocabulary:
        if w0 not in vocab or w1 not in vocab:
            break

        # Compute distance in new vector space
        dist = to.distance(enriched_vectors[w0], enriched_vectors[w1])
        if dist < config.delta:

            # Compute the partial gradient
            gradient = to.partial_gradient(enriched_vectors[w0], enriched_vectors[w1])

            # Weight it by K1
            gradient *= config.hyper_k1
            if w0 in gradient_updates:
                gradient_updates[w0] += gradient
                update_count[w0] += 1
            else:
                gradient_updates[w0] = gradient
                update_count[w0] = 1
    return gradient_updates, update_count


def _sgd_step_syn(synonym_pairs: list, enriched_vectors: dict, config: SettingConfig, gradient_updates: dict,
                  update_count: dict) -> (dict, dict):
    vocab = set(config.vocabulary)
    for (w0, w1) in synonym_pairs:

        # Extra check for reduced vocabulary:
        if w0 not in vocab or w1 not in vocab:
            break

        dist = to.distance(enriched_vectors[w0], enriched_vectors[w1])
        if dist > config.gamma:
            gradient = to.partial_gradient(enriched_vectors[w0], enriched_vectors[w1])
            gradient *= config.hyper_k2
            if w1 in gradient_updates:
                gradient_updates[w1] += gradient
                update_count[w1] += 1
            else:
                gradient_updates[w1] = gradient
                update_count[w1] = 1
    return gradient_updates, update_count


def _sgd_step_vsp(vsp_pairs: dict, enriched_vectors: dict, config: SettingConfig, gradient_updates: dict,
                  update_count: dict) -> (dict, dict):
    vocab = set(config.vocabulary)
    for (w0, w1) in vsp_pairs:
        # Extra check for reduced vocabulary:
        if w0 not in vocab or w1 not in vocab:
            break

        original_distance = vsp_pairs[(w0, w1)]
        new_distance = to.distance(enriched_vectors[w0], enriched_vectors[w1])

        if original_distance <= new_distance:
            gradient = to.partial_gradient(enriched_vectors[w0], enriched_vectors[w1])
            gradient *= config.hyper_k3

            if w0 in gradient_updates:
                gradient_updates[w0] -= gradient
                update_count[w0] += 1
            else:
                gradient_updates[w0] = -gradient
                update_count[w0] = 1
    return gradient_updates, update_count


def sgd_step(vectors: dict, synonym_pairs: list, antonym_pairs: list, vsp_pairs: dict, config: SettingConfig):
    enriched_vectors = deepcopy(vectors)
    gradient_updates = dict()
    update_count = dict()

    # AR / AF Term (Antonyms):
    if 'ant' in config.mode:
        gradient_updates, update_count = _sgd_step_ant(antonym_pairs, enriched_vectors, config, gradient_updates,
                                                       update_count)

    # SA / SC Term (Synonyms):
    if 'syn' in config.mode:
        gradient_updates, update_count = _sgd_step_syn(synonym_pairs, enriched_vectors, config, gradient_updates,
                                                       update_count)

    # VSP / KN Term (Regularization to the original values):
    if 'vsp' in config.mode:
        gradient_updates, update_count = _sgd_step_vsp(vsp_pairs, enriched_vectors, config, gradient_updates,
                                                       update_count)

    for word in gradient_updates:
        update_term = gradient_updates[word] / (update_count[word])
        enriched_vectors[word] += update_term

    return to.normalise(enriched_vectors)


def counterfit(config: SettingConfig) -> dict:
    word_vectors = config.vectors
    vocabulary = config.vocabulary
    antonyms = config.antonyms
    synonyms = config.synonyms

    current_iteration = 0
    vsp_pairs = {}

    # Only compute the VSP Pairs step if listed in the config mode and the weight of the vsp pairs is different than 0
    if 'vsp' in config.mode and config.hyper_k3 > 0.0:  # if we need to compute the VSP terms.
        # TODO: Load existing VSP pairs if exist to avoid computationally heavy operation in the future
        vsp_pairs = compute_vsp_pairs(word_vectors, vocabulary, config)

    # Post-processing: remove synonym pairs which are deemed to be both synonyms and antonyms:
    for antonym_pair in antonyms:
        if antonym_pair in synonyms:
            synonyms.remove(antonym_pair)
        if antonym_pair in vsp_pairs:
            del vsp_pairs[antonym_pair]

    print("Running the optimisation procedure for ", config.sgd_iters, " SGD steps...")

    sgd_steps = config.sgd_iters

    while current_iteration < sgd_steps:
        current_iteration += 1
        print(f"\tRunning SGD Step {current_iteration}")
        word_vectors = sgd_step(word_vectors, synonyms, antonyms, vsp_pairs, config)
        print(f"\tFinished SGD Step {current_iteration}")
    return word_vectors


def run_experiment(config_path):
    print(f"Started counterfitting run @ {datetime.now()}")
    config = SettingConfig(config_path)
    if not config.vectors:
        print("Unable to load vectors. Aborting.")
        return
    enriched_vectors = counterfit(config)

    # Store all the counterfitting vectors
    to.store_vectors(dimens=config.dimensions, dst_path=config.output_vectors_path, vectors=enriched_vectors)

    config.init_comparator(config_path, enriched_vectors)

    # Perform comparative analysis with original vectors
    config.comparator.compare()


def main():
    try:
        config_filepath = sys.argv[1]
    except IndexError:
        print("\nUsing the default config file: parameters.cfg")
        config_filepath = "parameters.cfg"

    run_experiment(config_filepath)


if __name__ == "__main__":
    main()
