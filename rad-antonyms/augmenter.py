import configparser
import io
import os

import spacy
import yaml
from spacy.symbols import VERB

from util import tools


class Augmenter:

    def __init__(self, config_path, rasa_config_path):
        """
        :param config_path: path to the configuration file that contains the paramteres
        :param rasa_config_path: path to RASA's config.yml
        """

        # Read the config file
        self.config = configparser.RawConfigParser()
        try:
            self.config.read(config_path)
        except OSError:
            print("Unable to read config, aborting.")
            return

        with io.open(rasa_config_path, "r") as rasa_config_file:
            rasa_config = yaml.load(rasa_config_file, Loader=yaml.FullLoader)

            # Store the language from the RASA config
            language_name = rasa_config['language']

            # Load the SpaCy Language Model
            self.lang_model = spacy.load(language_name)

        # Load synonyms and antonyms
        constraints_root_path = self.config.get("paths", "CONSTRAINTS_ROOT_PATH")

        self.synonyms_path = os.path.join(constraints_root_path, "verbs", "synonyms.txt")
        self.antonyms_path = os.path.join(constraints_root_path, "verbs", "antonyms.txt")

        self.synonym_pairs = tools.load_constraints(self.synonyms_path)
        self.antonym_pairs = tools.load_constraints(self.antonyms_path)

    def augment(self, sentence: str) -> None:

        """
        :param sentence: The training sentence we want to augment our pairs based on
        """

        # Load the sentence into the spacy language model
        doc = self.lang_model(sentence)
        for token in doc:
            if token.pos == VERB:
                self.update_antonyms(token)
                self.update_synonyms(token)

    def update_synonyms(self, verb_lemma: str) -> None:
        pass

    def update_antonyms(self, verb_lemma: str) -> None:
        pass
