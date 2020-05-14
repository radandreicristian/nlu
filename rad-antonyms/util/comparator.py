import configparser
import io
import os
import pathlib
from typing import Optional, TextIO

import numpy as np

from util import tools


def compare_embeddings(path1, path2):
    file1 = io.open(file=path1, mode="r", encoding="utf-8")
    file2 = io.open(file=path2, mode="r", encoding="utf-8")

    # Load first vecs:
    print("Loading first vectors...")
    words1 = dict()
    dimensions = str(next(file1))
    if len(dimensions.split(" ")) == 2:
        # We have a dimensions line. Keep it in the variable, continue with the next lines
        pass
    else:
        # We do not have a dimensions line
        line = dimensions.split(' ', 1)
        key = line[0]
        words1[key] = np.fromstring(line[1], dtype="float32", sep=' ')
        dimensions = None
    for line in file1:
        line = line.split(' ', 1)
        key = line[0]
        words1[key] = np.fromstring(line[1], dtype="float32", sep=' ')
    print("Loaded first vectors.")

    print("Loading second vectors...")
    words2 = dict()
    dimensions = str(next(file2))
    if len(dimensions.split(" ")) == 2:
        # We have a dimensions line. Keep it in the variable, continue with the next lines
        pass
    else:
        # We do not have a dimensions line
        line = dimensions.split(' ', 1)
        key = line[0]
        words2[key] = np.fromstring(line[1], dtype="float32", sep=' ')
        dimensions = None
    for line in file2:
        line = line.split(' ', 1)
        key = line[0]
        words2[key] = np.fromstring(line[1], dtype="float32", sep=' ')
    print("Loaded second vectors.")

    count = 0
    for key in words1.keys():
        if not np.array_equal(words1[key], words2[key]):
            print(f"different embeddings for {key}")
            count += 1

    print(f"Different vectors : {count} / {len(words1)}")


class Comparator:

    def __init__(self, config_path: str, original_vectors: dict, counterfit_vectors: dict, **kwargs):

        self.config = configparser.RawConfigParser()
        try:
            self.config.read(config_path)
        except OSError:
            print("Unable to read config, aborting.")
            return

        self.counterfit_vectors = counterfit_vectors
        self.original_vectors = original_vectors

        # Save the keywordargs
        self.args = kwargs

        # Set counterfit and original vectors path
        self.counterfit_vectors_path = self.args['counterfit_vectors_path']

        self.original_vectors_path = os.path.join(pathlib.Path(__file__).parent.parent.absolute(),
                                                  self.config.get("paths",
                                                                  "VEC_PATH"))

        # Compute the path to antonyms and load them in a set
        dataset_antonyms_path = os.path.join(pathlib.Path(__file__).parent.parent.absolute(),
                                             self.config.get("paths", "DATASET_ANTONYMS_PATH"))

        dataset_synonyms_path = os.path.join(pathlib.Path(__file__).parent.parent.absolute(),
                                             self.config.get("paths", "DATASET_SYNONYMS_PATH"))

        self.dataset_antonyms = tools.load_pairs(dataset_antonyms_path)
        self.dataset_synonyms = tools.load_pairs(dataset_synonyms_path)

        # Set epsilon, the minimum distance difference between similarities of an origina/counterfit pair
        # to be considered valid
        self.epsilon = self.config.get("hyperparameters", "epsilon")
        self.output_path = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "lang",
                                        "counterfitting_reports",
                                        str("".join(self.counterfit_vectors_path).split("/")[-1].rsplit(".", 1)[
                                                0]) + ".txt")

    def compare(self):
        # Load original and counterfit vectors
        if not self.original_vectors:
            print("Loaded original vectors from file")
            _, og_vecs = tools.load_vectors_novocab(self.original_vectors_path)
        else:
            print("Loaded original vectors from parameter")
            og_vecs = self.original_vectors
        if not self.counterfit_vectors:
            print("Loaded counterfit vectors from file")
            _, cf_vecs = tools.load_vectors_novocab(self.counterfit_vectors_path)
        else:
            print("Loaded counterfit vectors from parameter")
            cf_vecs = self.counterfit_vectors

        with io.open(file=self.output_path, mode="w", encoding="utf-8") as output_file:
            # Write the kwargs as the counterfit run oiptions
            output_file.write("Counterfitting Run Options\n")
            for k, v in self.args.items():
                key = k.replace("'", '')
                value = str(v).replace("'", '').replace("]", "").replace("[", "")
                output_file.write(f"{key} : {value}\n")
            output_file.write("\n\n I. Antonyms Results \n\n")

            # Write the report regarding the antonym pairs in the dataset
            self.compare_counterfit_pairs(og_vecs, cf_vecs, output_file, self.dataset_antonyms)

            output_file.write("\n\n II. Syonyms Results \n\n")
            # In the same file,w rite the report regarding the synonym pairs in the dataset
            self.compare_counterfit_pairs(og_vecs, cf_vecs, output_file, self.dataset_synonyms)
            output_file.close()

    def compare_counterfit_pairs(self, original_vectors: dict, counterfit_vectors: dict, output_file: TextIO,
                                 pairs: set) -> None:
        valid_pairs = []
        for (w1, w2) in pairs:
            valid = self.report_pair(w1, w2, original_vectors, counterfit_vectors, output_file)
            if valid:
                valid_pairs.append(valid)
        output_file.write(f"\n Counterfit pairs with significant increase ( distance diff. > {self.epsilon})"
                          f"\n Count: {len(valid_pairs)} / {len(self.dataset_antonyms)} "
                          f" ({len(valid_pairs) / len(self.dataset_antonyms) * 100}%)")
        for pair in valid_pairs:
            output_file.write(f"\t{pair[0], pair[1]}".replace("'", "").replace(']', '').replace('[', ''))

    def report_pair(self, w1: str, w2: str, og_vecs: dict, cf_vecs: dict, output_file: TextIO) -> Optional[tuple]:
        # Compute and write the pair stats to file
        stats = self.compute_pair_stats(w1, w2, og_vecs, cf_vecs)
        output_file.write(stats)

        # Return the pair if the abs of the difference of cos similarities between original and counterfit
        # vectors is greater than epsilon
        return (w1, w2) if abs(
            float(tools.cos_sim(og_vecs[w1], og_vecs[w2]) - tools.cos_sim(cf_vecs[w1], cf_vecs[w2]))) > float(
            self.epsilon) else None

    @staticmethod
    def compute_pair_stats(w1: str, w2: str, og_vecs: dict, cf_vecs: dict) -> str:
        # Simply return a formatted string containing the full report of a pair of words
        return (f"Similarity report for {w1}, {w2}:\n"
                f"\tCos between original/counterfit {w1}: {tools.cos_sim(og_vecs[w1], cf_vecs[w1])}\n"
                f"\tCos between original/counterfit {w2}: {tools.cos_sim(og_vecs[w2], cf_vecs[w2])}\n"
                f"\tOriginal cos for {w1}/{w2}: {tools.cos_sim(og_vecs[w1], og_vecs[w2])}\n"
                f"\tCounterfit cos for {w1}/{w2}: {tools.cos_sim(cf_vecs[w1], cf_vecs[w2])}\n")


if __name__ == '__main__':
    compare_embeddings('../lang/vectors/ro_ft_300_allpos_rwn_ant_syn_augpairs_1588855064.vec',
                       '../lang/vectors/ro_ft_300_allpos_rwn_ant_syn_augpairs_1588856432.vec')
