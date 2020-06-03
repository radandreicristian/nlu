import configparser
import os
import pathlib

import pandas as pd

from util import tools


class Comparator:
    """
    Class that performs the comparation between two sets of vectors: the original and the counterfit version.
    """

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
        self.syn_output_path = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "lang",
                                            "counterfitting_reports",
                                            str("".join(self.counterfit_vectors_path).split("/")[-1].rsplit(".", 1)[
                                                    0]) + "_syn.csv")
        self.ant_output_path = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "lang",
                                            "counterfitting_reports",
                                            str("".join(self.counterfit_vectors_path).split("/")[-1].rsplit(".", 1)[
                                                    0]) + "_ant.csv")

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

        syn_cos_similarities_first_term = list()
        syn_cos_similarities_second_term = list()
        syn_original_cos_similarities = list()
        syn_counterfit_cosine_similarities = list()

        ant_cos_similarities_first_term = list()
        ant_cos_similarities_second_term = list()
        ant_original_cos_similarities = list()
        ant_counterfit_cosine_similarities = list()

        for syn_pair in self.dataset_synonyms:
            w0 = syn_pair[0]
            w1 = syn_pair[1]
            syn_cos_similarities_first_term.append(tools.cos_sim(og_vecs[w0], cf_vecs[w0]))
            syn_cos_similarities_second_term.append(tools.cos_sim(og_vecs[w1], cf_vecs[w1]))
            syn_original_cos_similarities.append(tools.cos_sim(og_vecs[w0], og_vecs[w1]))
            syn_counterfit_cosine_similarities.append(tools.cos_sim(cf_vecs[w0], cf_vecs[w1]))

        for ant_pair in self.dataset_antonyms:
            w0 = ant_pair[0]
            w1 = ant_pair[1]
            ant_cos_similarities_first_term.append(tools.cos_sim(og_vecs[w0], cf_vecs[w0]))
            ant_cos_similarities_second_term.append(tools.cos_sim(og_vecs[w1], cf_vecs[w1]))
            ant_original_cos_similarities.append(tools.cos_sim(og_vecs[w0], og_vecs[w1]))
            ant_counterfit_cosine_similarities.append(tools.cos_sim(cf_vecs[w0], cf_vecs[w1]))

        pretty_dataset_synonyms = list(map(lambda x: " ".join(x).replace(',', "").replace(")", "").replace("(", ""),
                                           self.dataset_synonyms))
        pretty_dataset_antonyms = list(map(lambda x: " ".join(x).replace(',', "").replace(")", "").replace("(", ""),
                                           self.dataset_antonyms))

        syn_data = {'Synonym Pairs': pretty_dataset_synonyms,
                    'W1 OG / CF sim.': syn_cos_similarities_first_term,
                    'W2 OG / CF sim.': syn_cos_similarities_second_term,
                    'Original pair sim.': syn_original_cos_similarities,
                    'Counterfit pair sim.': syn_counterfit_cosine_similarities}

        ant_data = {'Synonym Pairs': pretty_dataset_antonyms,
                    'W1 OG / CF sim.': ant_cos_similarities_first_term,
                    'W2 OG / CF sim.': ant_cos_similarities_second_term,
                    'Original pair sim.': ant_original_cos_similarities,
                    'Counterfit pair sim.': ant_counterfit_cosine_similarities}

        syn_df = pd.DataFrame(syn_data)
        ant_df = pd.DataFrame(ant_data)

        syn_df.to_csv(self.syn_output_path, encoding="utf-8")
        ant_df.to_csv(self.ant_output_path, encoding="utf-8")
