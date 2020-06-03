import argparse
import configparser
import os.path
from copy import deepcopy
from datetime import datetime

import util.tools as to
from util.comparator import Comparator


class SettingConfig:
    """
    Class that encapsulates all the parameters and variables required for a counterfitting run
    """

    def __init__(self, config_path, language_model_name):

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

        print("Loading constraints...")
        # Append antonyms and synonyms of each selected PoS from their respective folder

        # Read and parse the mode (whether to include synonyms, antonyms or VSP pairs in the current run)
        mode = self.config.get("settings", "MODE").replace("[", "").replace("]", "").replace(" ", "").split(",")

        self.antonyms = []
        self.synonyms = []
        self.vsp_pairs = {}

        if 'ant' in mode:
            for part_of_speech in self.parts_of_speech:
                antonym_paths.append(os.path.join(constraints_root_path, part_of_speech, "antonyms.txt"))
            self.antonyms = to.load_multiple_constraints(antonym_paths)

        if 'syn' in mode:
            for part_of_speech in self.parts_of_speech:
                synonym_paths.append(os.path.join(constraints_root_path, part_of_speech, "synonyms.txt"))
            self.synonyms = to.load_multiple_constraints(synonym_paths)

        if 'vsp' in mode:
            vsp_path = self.config.get("paths", "VSP_PAIRS_VERB_PATH")
            self.vsp_pairs = to.load_vsp_pairs(vsp_path)

        print("Loaded constraints.")

        vocab = list()

        print("Loading vocabulary...")
        with open(file=vocab_path, mode="r", encoding="utf-8") as vocab_file:
            for line in vocab_file:
                vocab.append(line.strip())

        # Add augmented words from synonyms list to the vocab
        # Small optimization trick for O(1) lookup:
        vocab_set = set(vocab)

        if 'syn' in mode:
            for pair in self.synonyms:
                if pair[0] in vocab_set or pair[1] in vocab_set:
                    vocab.append(pair[0])
                    vocab.append(pair[1])

        if 'ant' in mode:
            # Add augmented words from antonym lists to the vocab
            for pair in self.antonyms:
                if pair[0] in vocab_set or pair[1] in vocab_set:
                    vocab.append(pair[0])
                    vocab.append(pair[1])

        vocab = to.unique(vocab)
        print("Loaded vocabulary.")

        # Load the word vectors
        print("Loading word vectors...")
        dimensions, self.vectors = to.load_vectors(self.input_vectors_path, set(vocab))

        # Return if vectors were not successfully loaded
        if not self.vectors:
            print("Unable to load initial vectors")
            return

        print("Loaded word vectors ")
        if language_model_name:
            self.output_vectors_path = f"{self.config.get('paths', 'VEC_ROOT_PATH')}/{language_model_name}.vec"
        else:
            self.output_vectors_path = self.config.get("paths", "CF_VEC_PATH").split(".")[
                                           0] + f"_{str(datetime.timestamp(datetime.now())).split('.')[0]}.vec"

        # The vocabulary contains the keys of vectors successfully loaded by the initial vocabulary: Words in the
        # initial vocabulary with no corresponding vector are skipped
        self.vocabulary = to.unique(self.vectors.keys())

        self.dimensions = f"{len(self.vocabulary)} {dimensions.split(' ')[1]}"

        # Load synonym and antonym pairs from the paths specified
        self.mode = mode

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


def _sgd_step_ant(antonym_pairs: list, enriched_vectors: dict, config: SettingConfig, gradient_updates: dict,
                  update_count: dict) -> (dict, dict):
    """
    Performs a single stochastic gradient descent step for antonym pairs.
    :param antonym_pairs: List of tuples containing antonym pairs
    :param enriched_vectors: The enriched vectors currently under computation
    :param config: Configuration of the experiment run
    :param gradient_updates: Dictionary of gradient updates. Keys are words and values are the so far
    accumulated partial gradients of other vectors with respect the key.
    :param update_count: Dictionary of counts of gradient updates. Keys are words and values are the number
    of times another vector's gradient with respect to the key was calculated and accumulated in gradient_updates.
    :return: The gradient updates and update counts but updated.
    """

    # Optimization trick for searching since in sets it's O(1)
    vocab = set(config.vocabulary)
    for (w0, w1) in antonym_pairs:

        # Extra check for reduced vocabulary:
        if w0 not in vocab or w1 not in vocab:
            break

        # Compute distance in new vector space
        dist = to.distance(enriched_vectors[w0], enriched_vectors[w1])
        if dist < config.delta:

            # Compute the partial gradient of the distance with respect to w0
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
    """
    Performs a single stochastic gradient descent step for synonym pairs.
    :param synonym_pairs: List of tuples containing synonym pairs
    :param enriched_vectors: The enriched vectors currently under computation
    :param config: Configuration of the experiment run
    :param gradient_updates: Dictionary of gradient updates. Keys are words and values are the so far
    accumulated partial gradients of other vectors with respect the key.
    :param update_count: Dictionary of counts of gradient updates. Keys are words and values are the number
    of times another vector's gradient with respect to the key was calculated and accumulated in gradient_updates.
    :return: The gradient updates and update counts but updated.
    """

    # Optimization trick for searching since in sets it's O(1)
    vocab = set(config.vocabulary)
    for (w0, w1) in synonym_pairs:

        # Extra check for reduced vocabulary:
        if w0 not in vocab or w1 not in vocab:
            break

        dist = to.distance(enriched_vectors[w0], enriched_vectors[w1])
        if dist > config.gamma:

            # Compute the partial gradient of the distance with respect to w0
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
    """
    Performs a single stochastic gradient descent step for synonym pairs.
    :param vsp_pairs: Dictionary of {(w1, w2): distance} containing pre-computed VSP pairs.
    :param enriched_vectors: The enriched vectors currently under computation
    :param config: Configuration of the experiment run
    :param gradient_updates: Dictionary of gradient updates. Keys are words and values are the so far
    accumulated partial gradients of other vectors with respect the key.
    :param update_count: Dictionary of counts of gradient updates. Keys are words and values are the number
    of times another vector's gradient with respect to the key was calculated and accumulated in gradient_updates.
    :return: The gradient updates and update counts but updated.
    """

    # Optimization for searching since in sets it's O(1)
    vocab = set(config.vocabulary)
    for (w0, w1) in vsp_pairs:
        # Extra check for reduced vocabulary:
        if w0 not in vocab or w1 not in vocab:
            break

        original_distance = vsp_pairs[(w0, w1)]
        new_distance = to.distance(enriched_vectors[w0], enriched_vectors[w1])

        if original_distance <= new_distance:

            # Compute the partial gradient of the distance with respect to w0
            gradient = to.partial_gradient(enriched_vectors[w0], enriched_vectors[w1])
            gradient *= config.hyper_k3

            if w0 in gradient_updates:
                gradient_updates[w0] -= gradient
                update_count[w0] += 1
            else:
                gradient_updates[w0] = -gradient
                update_count[w0] = 1
    return gradient_updates, update_count


def sgd_step(vectors: dict, synonym_pairs: list, antonym_pairs: list, vsp_pairs: dict, config: SettingConfig) -> dict:
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
    """
    Driver function for the counterfitting algorithm.
    :param config: Configuration of the experiment run.
    :return: The enhanced word vectors.
    """
    word_vectors = config.vectors
    antonyms = config.antonyms
    synonyms = config.synonyms
    vsp_pairs = config.vsp_pairs

    current_iteration = 0
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
    return word_vectors


def run_experiment(config_path: str, language_model_name: str) -> None:
    """
    Wrapper for the counterfitting algorithm including vector storing and result comparison.
    :param config_path: Path to the configuration file
    :param language_model_name: Unique idenfitier for the language
    :return:
    """
    print(f"Started counterfitting run at {to.get_time()}")
    config = SettingConfig(config_path, language_model_name)
    if not config.vectors:
        print("Unable to load vectors. Aborting.")
        return
    enriched_vectors = counterfit(config)

    # Store all the counterfitting vectors
    to.store_vectors(dimens=config.dimensions, dst_path=config.output_vectors_path, vectors=enriched_vectors)

    config.init_comparator(config_path, enriched_vectors)

    # Perform comparative analysis with original vectors
    config.comparator.compare()

    # For the sake of our RAM
    to.clear_memory()


def init_argument_parser() -> argparse.ArgumentParser:
    """
    Initailizes the argument parser for command line usage.
    :return: An ArgumentParser objects that knows how to parse specific parameters.
    """
    parser = argparse.ArgumentParser(description='Counterfitting to linguistic constraints')
    parser.add_argument('-c', '--config_filepath',
                        action='store', nargs='?', type=str)
    parser.add_argument('-l', '--language_model_name',
                        action='store', nargs='?', type=str)

    return parser


def main():
    arg_parser = init_argument_parser()
    arguments = arg_parser.parse_args()

    config_filepath = arguments.config_filepath
    if not config_filepath:
        config_filepath = "parameters.cfg"

    language_model_name = arguments.language_model_name
    run_experiment(config_filepath, language_model_name)


if __name__ == "__main__":
    main()
