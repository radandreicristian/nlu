import configparser
import io
import os
import sys

import mlconjug
import spacy
import yaml

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

        self.synonyms_path = os.path.join(constraints_root_path, "verb", "synonyms.txt")
        self.antonyms_path = os.path.join(constraints_root_path, "verb", "antonyms.txt")

        self.augmented_synonyms_path = os.path.join(constraints_root_path, "verb", "synonyms_aug.txt")
        self.augmented_antonyms_path = os.path.join(constraints_root_path, "verb", "antonyms_aug.txt")

        self.synonym_pairs = tools.load_constraints(self.synonyms_path)
        self.antonym_pairs = tools.load_constraints(self.antonyms_path)

        self.conjugator = mlconjug.Conjugator(language='ro')

        # https://universaldependencies.org/tagset-conversion/ro-multext-uposf.html
        # TODO: not used at the moment, might use it in the future; Good to have
        self.valid_verb_forms = ['Vmii1', 'Vmii1p', 'Vmii1s', 'Vmii2s', 'Vmii2p', 'Vmii3p',
                                 'Vmii3s', 'Vmil1', 'Vmil3p', 'Vmil3s', 'Vmip1p', 'Vmip1s', 'Vmip2p', 'Vmip2s', 'Vmip3',
                                 'Vmip3p', 'Vmip3s', 'Vmis1p', 'Vmis1s', 'Vmis3p', 'Vmis3s', 'Vmm-2p', 'Vmm-2s', 'Vmnp',
                                 'Vmsp1p', 'Vmsp1s', 'Vmsp2p', 'Vmsp2s', 'Vmsp3', 'Vmsp3s']

    def process_sentence(self, sentence: str) -> tuple:

        """
        :param sentence: The training sentence we want to augment our pairs based on
        """

        synonym_output_pairs = set()
        antonym_output_pairs = set()

        # Load the sentence into the spacy language model
        doc = self.lang_model(sentence)

        # Iterate the tokens in the document
        for token in doc:

            # If the PoS Treebank Tag is in the valid verb forms, do the processing
            # TODO: Workaround to only consider certain verb forms - explore the romanian tags
            if token.tag_[0] == 'V':

                conjugated_antonym_pairs = self.update_pairs(token.text, token.lemma_, self.antonym_pairs)
                conjugated_synonym_pairs = self.update_pairs(token.text, token.lemma_, self.synonym_pairs)

                synonym_output_pairs = synonym_output_pairs | conjugated_synonym_pairs
                antonym_output_pairs = antonym_output_pairs | conjugated_antonym_pairs

        return synonym_output_pairs, antonym_output_pairs

    def update_pairs(self, target_word: str, target_lemma: str, constraint_pairs: set) -> set:

        # Create a set that will contain pairs of lemma words from the constraints
        lemma_pairs = set()

        # For each constraint pair that contains the lemma of the target word:
        for pair in (_pair for _pair in constraint_pairs if target_lemma in _pair):

            # Take the other element of the pair
            other = pair[1 - pair.index(target_lemma)]

            # Create a spacy doc from the other
            doc_other = self.lang_model(other)
            for token in doc_other:
                # Make sure it is a verb
                if token.tag_[0] == 'V':
                    # Add to the lemma pairs set the pair of the target word's lemma and its pair's lemma
                    lemma_pairs.add(tuple(sorted((target_lemma, token.lemma_), key=lambda x: x[0])))

        # Create a list of candidates - times/moods/persons of the conjugated lemma equal to the target word
        candidate_conjugations = list()

        # Figure the possible conjugations of the lemmas which are equal to the word
        try:
            target_conjugations = self.conjugator.conjugate(target_lemma)
            # For each conjugation of the target word
            for conjugation in target_conjugations.iterate():

                # Unpack the time, mood, person and value from the tuple
                time, mood, person, value = conjugation

                if value == target_word:
                    # Found a possible candidate, append the time/mood/person to the list
                    candidate_conjugations.append((time, mood, person))

        except ValueError:
            print(f"Unable to conjugate, possibly mistagged verb {target_lemma}")

        # Create a set that will contain the final conjugated pairs
        conjugated_pairs = set()

        for pair in lemma_pairs:
            other = pair[1 - pair.index(target_lemma)]

            for conj in candidate_conjugations:
                # Conjugate the lemma to the current iterated conjugation
                try:
                    conjugated_other = self.conjugator.conjugate(other).conjug_info[conj[0]][conj[1]][conj[2]]

                    # Add the pair composed of the target word and the conjugated form of the other to the final set
                    conjugated_pairs.add((target_word, conjugated_other))
                    print(f"\t successfully added pair: ({target_word}, {conjugated_other})")
                except ValueError as e:
                    print(f"Unable to add pair due to value: {other} {e.args}")
                except KeyError as e:
                    print(f"Unable to add pair due to key {other} {e.args}")

        return conjugated_pairs

    def process_dataset(self, sentences: set) -> None:
        synonyms_file = io.open(file=self.augmented_synonyms_path, mode="w", encoding="utf-8")
        antonyms_file = io.open(file=self.augmented_antonyms_path, mode="w", encoding="utf-8")

        size = len(sentences)
        for index, sentence in enumerate(sentences):
            print(f"Processing Sentence {index}/{size}")
            syn_pairs, ant_pairs = self.process_sentence(sentence)
            self.synonym_pairs |= syn_pairs
            self.antonym_pairs |= ant_pairs

        synonyms_file.write('\n'.join('{} {}'.format(x[0], x[1]) for x in self.synonym_pairs))
        antonyms_file.write('\n'.join('{} {}'.format(x[0], x[1]) for x in self.antonym_pairs))


def main():
    try:
        config_filepath = sys.argv[1]
        rasa_filepath = sys.argv[2]
    except IndexError:
        print("\nUsing the default config file: parameters.cfg")
        print("\nUsing the default rasa config file: config.yml")
        config_filepath = "parameters.cfg"
        rasa_filepath = "config.yml"
    augmenter = Augmenter(config_filepath, rasa_filepath)
    sentences = tools.extract_sentences("datasets/cu_diacritice")
    augmenter.process_dataset(sentences)


if __name__ == "__main__":
    main()
