import configparser
import copy
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from shutil import copyfile
from shutil import rmtree
from statistics import stdev, mean
from zipfile import ZipFile

import yaml
from sklearn.utils import resample

from util.result.cloud_result_writer import CloudResultWriter
from util.result.local_result_writer import LocalResultWriter
from util.semantic_augmenter import extract_verb_lemmas, augment_synonym_verbs, augment_antonym_verbs
from util.tools import copy_path, extract_sentences_from_file, save_pairs_to_file, get_time, get_timestamp

SPREADSHEET_START_VERTICAL_OFFSET = 3

import warnings

warnings.filterwarnings("ignore")


class SettingConfig:
    """
    Class encapsulating the parameters of a complete RASA NLU pipeline run.
    """

    def __init__(self, config_path):

        # Read the config file
        self.config = configparser.RawConfigParser()
        try:
            self.config.read(config_path)
        except OSError:
            print("Unable to read config, aborting.")
            return

        self.project_root = pathlib.Path(__file__).parent.absolute()

        # Load Hyperparameters
        self.splits = self.config.getint("hyperparameters", "SPLITS")
        self.sample_percent = self.config.getfloat("hyperparameters", "SAMPLE_PERCENT")

        # Load datasets path
        self.diacritics = self.config.get("settings", "DIACRITICS")
        if self.diacritics == "True":
            self.datasets_path = self.config.get("paths", "DATASET_DIAC_PATH")
        else:
            self.datasets_path = self.config.get("paths", "DATASET_NODIAC_PATH")

        # Load script and config paths
        self.rasa_script_path = self.config.get("paths", "SCRIPT_PATH")
        self.rasa_config_relative_path = self.config.get("paths", "RASA_CONFIG_PATH")
        self.rasa_config_absolute_path = os.path.join(pathlib.Path(__file__).parent.absolute(),
                                                      self.rasa_config_relative_path)

        # Load report paths
        self.intent_report_path = self.config.get("paths", "INTENT_REPORT_PATH")
        self.intent_errors_path = self.config.get("paths", "INTENT_ERRORS_PATH")
        self.slot_report_path = self.config.get("paths", "SLOT_REPORT_PATH")
        self.slot_errors_path = self.config.get("paths", "SLOT_ERRORS_PATH")
        self.conf_mat_path = self.config.get("paths", "CONFUSION_MATRIX_PATH")
        self.histogram_path = self.config.get("paths", "HISTOGRAM_PATH")

        # Load merged reports path
        self.merged_reports_root = self.config.get("paths", "MERGED_REPORTS_ROOT")
        self.merged_intent_report_path = self.config.get("paths", "MERGED_INTENT_REPORT_PATH")
        self.merged_intent_errors_path = self.config.get("paths", "MERGED_INTENT_ERRORS_PATH")
        self.merged_slot_report_path = self.config.get("paths", "MERGED_SLOT_REPORT_PATH")
        self.merged_slot_errors_path = self.config.get("paths", "MERGED_SLOT_ERRORS_PATH")
        self.merged_matrices_path = self.config.get("paths", "MERGED_MATRICES_PATH")
        self.merged_histograms_path = self.config.get("paths", "MERGED_HISTOGRAMS_PATH")

        self.archives_path = self.config.get("paths", "ARCHIEVES_PATH")

        with io.open(self.rasa_config_absolute_path, "r") as rasa_config_file:
            rasa_config = yaml.load(rasa_config_file, Loader=yaml.FullLoader)
            self.language = rasa_config['language']

        self.identifier = f"{self.language}_{'diac' if self.diacritics else 'nodiac'}"
        self.base_language_model = self.config.get("settings", "BASE_LANGAUGE")
        self.language_code = self.config.get("settings", "LANGUAGE_CODE")

        output_method = self.config.get("settings", "SAVE_TO")
        if output_method == 'Cloud':
            secret_path = self.config.get("settings", "SECRET_PATH")
            spreadsheet_name = self.config.get("settings", "SPREADSHEET_NAME")
            self.writer = CloudResultWriter(secret_path=secret_path, spreadsheet_name=spreadsheet_name)
        elif output_method == 'Local':
            output_path = self.config.get("paths", "QUANTITATIVE_RESULTS_PATH")
            self.writer = LocalResultWriter(splits=self.splits, output_path=output_path)
        else:
            raise ValueError("Incorrect value in config. Should be either 'Cloud' or 'Local'")

        self.should_augment = (self.config.get("components", "AUGMENTER") == 'True')


def _setup_folders(config: SettingConfig) -> None:
    pathlib.Path(config.merged_matrices_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(config.merged_histograms_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(config.archives_path).mkdir(parents=True, exist_ok=True)


def _update_spacy_language(config: SettingConfig, language: str) -> None:
    with io.open(file=config.rasa_config_absolute_path, mode="r", encoding="utf-8") as rasa_config_file:
        rasa_config = yaml.load(rasa_config_file, Loader=yaml.FullLoader)
        rasa_config_file.close()

    rasa_config['language'] = language

    with io.open(file=config.rasa_config_absolute_path, mode="w", encoding="utf-8") as rasa_config_file:
        yaml.dump(rasa_config, rasa_config_file)
        rasa_config_file.close()


def _create_single_split(base_path: str, split_identifier: int, sample_percent: float, train: bool) -> None:
    """
    Creates a single train / test split from a base train / test file in RASA NLU JSON format.
    :param base_path: Path of the original file.
    :param split_identifier: Identifier of the split.
    :param sample_percent: The percentage of examples to be sampled.
    :param train: Sample from / save as train file if True, as test file otherwise.
    :return: None.
    """

    split_type = "train" if train else "test"
    path = os.path.join(base_path, split_type + ".json")

    with io.open(path, "r", encoding="utf-8") as file:
        # Load the JSON content
        content = json.load(file)
        content_copy = copy.deepcopy(content)

        # Obtain the wrapped object
        data = content_copy["rasa_nlu_data"]

        # Obtain the list of sentences
        common_examples = data["common_examples"]

        sample_count = int(sample_percent * len(common_examples))
        resampled_examples = resample(common_examples, n_samples=sample_count)
        data["common_examples"] = resampled_examples

        # Create the new file
        write_path = os.path.join(base_path, split_type + "_" + str(split_identifier) + ".json")
        file = io.open(write_path, "w", encoding="utf-8")

        # Dump JSON content into the file
        json.dump(content_copy, file, ensure_ascii=False)
        file.close()


def _create_splits(config: SettingConfig) -> None:
    """
    Creates a number of splits according to a hyperparameter of the run.
    :param config: Configuration of the current run.
    :return: None.
    """
    # Iterate every scenario folder - Name has to conform "scenario_$nr"
    for file in os.listdir(config.datasets_path):

        # Compute the path of to the scenario folder
        scenario_folder_path = os.path.join(config.datasets_path, file)

        # Only take into account scenario directories
        if os.path.isdir(scenario_folder_path):

            # Ensure the folder contains the scenarios
            if len(os.listdir(scenario_folder_path)) == 0:
                print('Empty Scenario Directory. Exiting.')
                break

            # For each split, create the dataset split and save it
            for split_id in range(config.splits):
                # Create splits for train / test datasets
                _create_single_split(scenario_folder_path, split_id, train=True, sample_percent=config.sample_percent)
                _create_single_split(scenario_folder_path, split_id, train=False, sample_percent=config.sample_percent)
    print("Finished creating fresh dataset splits")


def _wipe_reports(config: SettingConfig) -> None:
    """
    Deletes the reports generated by previous runs.
    :param config: Configuration of the current run.
    :return: None.
    """
    # Wipe the merged intent and slot errors and reports
    merged_reports_paths = [config.merged_intent_report_path, config.merged_intent_errors_path,
                            config.merged_slot_report_path, config.merged_slot_errors_path]

    for file in merged_reports_paths:
        if os.path.exists(file):
            os.remove(file)

    # Wipe the current errors and reports
    reports_paths = [config.intent_report_path, config.intent_errors_path, config.slot_report_path,
                     config.slot_errors_path]

    for file in reports_paths:
        if os.path.exists(file):
            os.remove(file)

    # Delete all report directories
    for scenario_folder in os.listdir(config.datasets_path):
        scenario_path = os.path.join(config.datasets_path, scenario_folder)
        if os.path.isdir(scenario_path):
            for file in os.listdir(scenario_path):
                file_path = os.path.join(scenario_path, file)
                if os.path.isdir(file_path):
                    rmtree(file_path)


def _wipe_splits(config: SettingConfig) -> None:
    """
    Deletes splits generated by the previous run(s).
    :param config: Configuration of the current run.
    :return: None.
    """
    # Iterate every scenario folder - Name has to conform "scenario_$nr"
    for file in os.listdir(config.datasets_path):

        # Compute the path of to the scenario folder
        scenario_folder_path = os.path.join(config.datasets_path, file)

        # Only take into account scenario directories
        if os.path.isdir(scenario_folder_path):

            kept_files = ["train.json", "test.json"]

            for _file in os.listdir(scenario_folder_path):
                if _file not in kept_files:
                    _file_abs_path = os.path.join(scenario_folder_path, _file)
                    if os.path.isdir(_file_abs_path):
                        rmtree(_file_abs_path)
                    else:
                        os.remove(_file_abs_path)

    print("Finished wiping the splits")


def _process_intent_results(identifier: str, config: SettingConfig) -> float:
    #  Move intent report result to the merged report
    with io.open(config.intent_report_path, "r", encoding="utf-8") as report_file:
        # Open the merged intent report file for writing
        with io.open(config.merged_intent_report_path, "a+", encoding="utf-8") as output_file:
            # Load the content
            content = json.load(report_file)
            f1_score = content['weighted avg']['f1-score']

            # Write the weighted average for intent detection in the file
            output_file.write(
                "F1-Score - ID for "
                + identifier + f":{f1_score}\n")

            output_file.close()
        report_file.close()
    return f1_score


def _process_intent_errors(identifier: str, config: SettingConfig) -> None:
    # Move intent errors to the merged errors
    if not os.path.exists(config.intent_errors_path):
        return
    with io.open(config.intent_errors_path, "r", encoding="utf-8") as report_file:
        # Open the merged errors report file for writing
        with io.open(config.merged_intent_errors_path, "a+", encoding="utf-8") as output_file:
            # Load the content
            if os.path.getsize(config.intent_errors_path) != 0:
                content = json.load(report_file)
                if content:
                    output_file.write(f"\nIntent Errors report for {identifier}")
                    content = sorted(content, key=lambda k: (k['intent_prediction']['name'], k['intent']))
                    # Copy all the intent errors in a human-readable form to the merged errors report
                    for entry in content:
                        output_file.write(
                            f"\n\tPredicted: {entry['intent_prediction']['name']}. Actual: {entry['intent']}. "
                            f"Text: {entry['text']}. "
                            f"Conf: {entry['intent_prediction']['confidence']}".replace('\"', ""))
                    output_file.close()
            report_file.close()


def _process_intents(identifier: str, scenario_report_path: str, config: SettingConfig) -> float:
    """
    Processes the intent detection results, writing them in a human readable format in a merged report.
    :param identifier: Identifier of the scenario/split.
    :param scenario_report_path: Path to the report for the current scenario.
    :param config: Configuration of the current run.
    :return: The weighted average F1 score.
    """

    _process_intent_errors(identifier, config)
    f1_score = _process_intent_results(identifier, config)

    # Also move the whole reports for backup
    report_identifier_path = os.path.join(scenario_report_path, "intent_reports", "report_" +
                                          identifier.replace(" ", "").replace(",", "").replace("_",
                                                                                               "") + ".txt")
    errors_identifier_path = os.path.join(scenario_report_path, "intent_reports", "errors_" +
                                          identifier.replace(" ", "").replace(",", "").replace("_",
                                                                                               "") + ".txt")
    # Open (create) the file
    io.open(report_identifier_path, "w+", encoding="utf-8")
    io.open(errors_identifier_path, "w+", encoding="utf-8")

    # Save the content from the current report to the intent report
    copy_path(config.intent_report_path, report_identifier_path)
    copy_path(config.intent_errors_path, errors_identifier_path)

    return f1_score


def _process_slot_results(identifier: str, config: SettingConfig) -> float:
    # Move slot report result to the merged report
    with io.open(config.slot_report_path, "r", encoding="utf-8") as report_file:
        # Open the merged slot report file for appending
        with io.open(config.merged_slot_report_path, "a+", encoding="utf-8") as output_file:
            # Load the content
            content = json.load(report_file)

            # Write the weighted average for intent detection in the file
            f1_score = content['weighted avg']['f1-score']
            output_file.write(
                "F1-Score - SF for "
                + identifier + f":{f1_score}\n")

            output_file.close()
        report_file.close()
    return f1_score


def _process_slot_errors(identifier: str, config: SettingConfig) -> None:
    # Move slot errors to the merged errors
    if not os.path.exists(config.intent_errors_path):
        return
    with io.open(config.slot_errors_path, "r", encoding="utf-8") as report_file:
        # Open the merged errors report file for writing
        with io.open(config.merged_slot_errors_path, "a+", encoding="utf-8") as output_file:
            # Load the content
            if os.path.getsize(config.slot_errors_path) != 0:
                content = json.load(report_file)
                if content:
                    content = sorted(content, key=lambda k: k['entities'][0]['entity'] if k['entities'] else "")
                    # Copy all the slot errors in a human-readable form to the merged errors report
                    output_file.write(f"\nErrors report for {identifier}")
                    for entry in content:
                        output_file.write(
                            f"\n\tPredicted: {[(e['entity'], e['value']) for e in entry['predicted_entities']]}. "
                            f"Actual: {[(e['entity'], e['value']) for e in entry['entities']]}. "
                            f"Text: {entry['text']}".replace("[", "").replace(")]", "").replace("(", "").replace("',",
                                                                                                                 ": ").replace(
                                "),",
                                ",").replace(
                                '  ', ' ').replace('\'', "").replace("].", " - ")
                        )
            output_file.close()
        report_file.close()


def _process_slots(identifier: str, scenario_report_path: str, config: SettingConfig) -> float:
    """
    Processes the slot filling results, writing them in a human readable format in a merged report.
    :param identifier: Identifier of the scenario/split.
    :param scenario_report_path: Path to the report for the current scenario.
    :param config: Configuration of the current run.
    :return: The weighted average F1 score.
    """

    f1_score = _process_slot_results(identifier, config)
    _process_slot_errors(identifier, config)

    # Compute the path for the current intent report and errors
    report_identifier_path = os.path.join(scenario_report_path, "slot_reports", "report_" +
                                          identifier.replace(" ", "").replace(",", "").replace("_",
                                                                                               "") + ".txt")
    errors_identifier_path = os.path.join(scenario_report_path, "slot_reports", "errors_" +
                                          identifier.replace(" ", "").replace(",", "").replace("_",
                                                                                               "") + ".txt")

    # Open (create) the file
    io.open(report_identifier_path, "w+", encoding="utf-8")
    io.open(errors_identifier_path, "w+", encoding="utf-8")

    # Save the content from the current report to the intent report
    copy_path(config.slot_report_path, report_identifier_path)
    copy_path(config.slot_errors_path, errors_identifier_path)
    return f1_score


def _copy_confusion_matrix(identifier: str, config: SettingConfig) -> None:
    """
    Copy a confusion matrix to the merged folder.
    :param identifier: Identifier of the scenario / split.
    :param config: Configuration of the current run.
    :return: None.
    """
    copyfile(config.conf_mat_path,
             os.path.join(config.merged_matrices_path,
                          identifier.replace(" ", "").replace(",", "").replace("_", "") + ".png"))


def _copy_histogram(identifier: str, config: SettingConfig) -> None:
    """
    Copy a histogram to the merged folder.
    :param identifier: Identifier of the scenario / split.
    :param config: Configuration of the current run.
    :return: None.
    """
    copyfile(config.histogram_path,
             os.path.join(config.merged_histograms_path,
                          identifier.replace(" ", "").replace(",", "").replace("_", "") + ".png"))


def _process_split_augmented(file: str, folder_path: str, split_id: int, config: SettingConfig,
                             scenario_reports_path: str,
                             scenario_slot_results: list, scenario_intent_results: list) -> None:
    """
    Peforms a full pipeline run for a singple split. This includes extracting verbs, and modes, augmenting,
    counterfitting, spacy model pipeline,  training and evaluation.
    :param file: The identifier of the scenario (i.e. "scenario_0")
    :param folder_path: Path to the scenario root.
    :param split_id: Id of the split.
    :param config: Configuration of the current run.
    :param scenario_reports_path: Path of the reports for the current scenario.
    :param scenario_slot_results: Slot labeling results for the current scenario.
    :param scenario_intent_results: Intent classification results for the current scenario.
    :return: None
    """
    # Compute the identifier, get the train split and test split
    identifier = f" {file}, split {split_id}"
    train_split = os.path.join(folder_path, f"train_{split_id}.json")
    test_split = os.path.join(folder_path, f"test_{split_id}.json")

    # Extract verbs from train
    print(f"Started extracting sentences from the train data at {get_time()}..")
    train_sentences = extract_sentences_from_file(train_split)
    print(f"    Finished extracting sentences from the train data at {get_time()}..")

    # Augment -> save verb pairs separately
    print(f"Started extracting lemmas and verbal times from train sentences at {get_time()}.")
    verb_lemmas, verb_times = extract_verb_lemmas(config.base_language_model, train_sentences)
    print(f"    Finished extracting lemmas and verbal times from train sentences at {get_time()}.")

    print(f"Started augmenting antonym pairs at {get_time()}")
    train_verb_antonym_pairs = augment_antonym_verbs(verb_lemmas, verb_times)
    print(f"    Finished augmenting antonym pairs at {get_time()}")

    print(f"Started augmenting synonym pairs at {get_time()}")
    train_verb_syonym_pairs = augment_synonym_verbs(verb_lemmas, verb_times)
    print(f"    Finished augmenting synonym pairs at {get_time()}")

    # Backup antonyms and synonyms:
    root_path = pathlib.Path(__file__).parent.absolute()
    verb_constraints_path = os.path.join(root_path, "lang", "constraints", "verb")

    verb_antonyms_path = os.path.join(verb_constraints_path, "antonyms.txt")
    verb_synonyms_path = os.path.join(verb_constraints_path, "synonyms.txt")

    backup_verb_antonyms_path = os.path.join(verb_constraints_path, "antonyms_backup.txt")
    backup_verb_synonyms_path = os.path.join(verb_constraints_path, "synonyms_backup.txt")

    copy_path(verb_antonyms_path, backup_verb_antonyms_path, False)
    copy_path(verb_synonyms_path, backup_verb_synonyms_path, False)

    # Append new verbs to the currnet verb
    save_pairs_to_file(train_verb_antonym_pairs, verb_antonyms_path, True)
    save_pairs_to_file(train_verb_syonym_pairs, verb_synonyms_path, True)

    # Create unique identifier for vectors / spacy lang model
    timestamp = get_timestamp()
    model_identifier = f"ro_ft_300_train_temp_{timestamp}"

    # Run counterfitting
    print(f"Started updating word embeddings at {get_time()}.")
    counterfitting_command = ['python', 'embedding_enhancer.py', '-l', model_identifier]
    subprocess.Popen(counterfitting_command, shell=True).wait()
    print(f"    Finished updating word embeddings at {get_time()}.")

    return

    # Run spacy model creation
    os.chdir('util')

    model_path = f"models/{model_identifier}"
    model_out_path = f"models/out/{model_identifier}"
    vectors_path = f"lang/vectors/{model_identifier}.vec"
    language_model_name = model_identifier

    spacy_command = ['python', 'init_spacy_lang.py',
                     '-l', config.language_code,
                     '-m', model_path,
                     '-v', vectors_path,
                     '-o', model_out_path,
                     '-n', language_model_name,
                     '-f']
    print(f"Started creating SpaCy language model at {get_time()}.")
    subprocess.Popen(spacy_command, shell=True).wait()
    print(f"    Finished creating SpaCy language model at {get_time()}.")

    # Update rasa config path:
    os.chdir(root_path)

    # print(f"Updating RASA config language model at {get_time()}")
    _update_spacy_language(config, language_model_name)

    # Run the subprocess for RASA training and testing, and wait for its completion
    print(f"Started NLU training and testing via RASA NLU at {get_time()}.")
    command = [config.rasa_script_path, train_split, test_split]
    # print(command)
    subprocess.Popen(command, shell=True).wait()
    print(f"    Finished the NLU training and testing via RASA NLU at {get_time()}.")

    # Process the slot and intent errors & reports and save their return values
    print(f"Started computing and writing results at {get_time()}")
    intent_f1_score = _process_intents(identifier, scenario_reports_path, config)
    slot_f1_score = _process_slots(identifier, scenario_reports_path, config)

    print("     Copying confusion matrices and histograms.")
    # Move the confusion matrix to the results path
    _copy_confusion_matrix(identifier, config)
    _copy_histogram(identifier, config)

    print("     Creating quantitative analyses spreadsheets.")
    scenario_slot_results.append(float("{:0.4f}".format(slot_f1_score)))
    scenario_intent_results.append(float("{:0.4f}".format(intent_f1_score)))

    # Restore backup
    print(f"Restoring backup files and deleting language model to free space at {get_time()}.")
    copy_path(backup_verb_synonyms_path, verb_synonyms_path, False)
    copy_path(backup_verb_antonyms_path, verb_antonyms_path, False)

    # Put the default value back in the rasa config
    _update_spacy_language(config, config.base_language_model)

    os.remove(os.path.join(root_path, vectors_path))
    # Remove models and vectors, since they will never be used again, in order to save a lot of memory.
    # Sleep to avoid some unbeknown race condition :/
    time.sleep(2)
    shutil.rmtree(model_path)

    time.sleep(2)
    # shutil.rmtree(model_out_path)

    print(f"Finished restoring backup files and deleting language model at {get_time()}")
    print(f"Finished processing split {identifier}")


def _process_split_baseline(file: str, folder_path: str, split_id: int, config: SettingConfig,
                            scenario_reports_path: str,
                            scenario_slot_results: list, scenario_intent_results: list) -> None:
    """
        Peforms a full pipeline run for a singple split. This does not include extracting verbs and modes, augmenting,
        counterfitting, spacy model pipeline, just training and evaluation.
        :param file: The identifier of the scenario (i.e. "scenario_0")
        :param file_path: Path to the scenario root.
        :param split_id: Id of the split.
        :param config: Configuration of the current run.
        :param scenario_reports_path: Path of the reports for the current scenario.
        :param scenario_slot_results: Slot labeling results for the current scenario.
        :param scenario_intent_results: Intent classification results for the current scenario.
        :return: None
        """
    # Compute the identifier, get the train split and test split
    identifier = f" {file}, split {split_id}"
    train_split = os.path.join(folder_path, f"train_{split_id}.json")
    test_split = os.path.join(folder_path, f"test_{split_id}.json")

    # Run the subprocess for RASA training and testing, and wait for its completion
    print(f"Running RASA training and testing at {get_time()}")
    command = [config.rasa_script_path, train_split, test_split]
    print(command)
    subprocess.Popen(command, shell=True).wait()

    # Process the slot and intent errors & reports and save their return values
    print(f"Computing and writing results at {get_time()}")
    intent_f1 = _process_intents(identifier, scenario_reports_path, config)
    slot_f1 = _process_slots(identifier, scenario_reports_path, config)

    # Move the confusion matrix to the results path
    _copy_confusion_matrix(identifier, config)
    _copy_histogram(identifier, config)

    scenario_slot_results.append(float("{:0.4f}".format(slot_f1)))
    scenario_intent_results.append(float("{:0.4f}".format(intent_f1)))


def _process_datasets(config: SettingConfig) -> None:
    """
    Processes a dataset with multiple splits, generating reports and results.
    :param should_enhance: Specifies if the user also wants counterfitting to the pipeline.
    :param config: Configuration of the current run.
    :return: None
    """

    # Compute and write the title of the spreadsheet based on the loaded configurations
    spreadsheet_title = [config.identifier]
    config.writer.append_row(spreadsheet_title, 1)

    # For each scenario folder
    spreadsheet_row_index = SPREADSHEET_START_VERTICAL_OFFSET
    print(os.listdir(config.datasets_path))
    for file in os.listdir(config.datasets_path):

        # Compute the path for the scenario folder
        folder_path = os.path.join(config.datasets_path, file)
        if os.path.isdir(folder_path):

            # Compute the scenario file name

            # Compute the reports path and create the directory
            scenario_reports_path = os.path.join(folder_path, 'scenario_reports')
            if 'scenario_2' not in folder_path:
                print(f'folder {folder_path}. breaking')
                continue
            os.mkdir(scenario_reports_path)

            # Compute the intent and slot reports paths and create them
            intent_reports_path = os.path.join(scenario_reports_path, 'intent_reports')
            slot_reports_path = os.path.join(scenario_reports_path, 'slot_reports')

            os.mkdir(intent_reports_path)
            os.mkdir(slot_reports_path)

            scenario_slot_results = [f'Slot - {file}']
            scenario_intent_results = [f'Intent - {file}']

            for split_id in range(config.splits):
                # TODO: This should be a try-except + rollback
                if config.should_augment:
                    _process_split_augmented(file, folder_path, split_id, config, scenario_reports_path,
                                             scenario_slot_results,
                                             scenario_intent_results)
                else:
                    _process_split_baseline(file, folder_path, split_id, config, scenario_reports_path,
                                            scenario_slot_results,
                                            scenario_intent_results)

            # Append the mean value to each list for the scenario
            scenario_intent_results.append(float("{:0.4f}".format(mean(scenario_intent_results[1:]))))
            scenario_slot_results.append(float("{:0.4f}".format(mean(scenario_slot_results[1:]))))

            # Append the standard deviation to each list for the scenario if we have more than a split.
            if config.splits > 1:
                scenario_intent_results.append(
                    float("{:0.4f}".format(stdev(scenario_intent_results[1:len(scenario_intent_results) - 1]))))
                scenario_slot_results.append(
                    float("{:0.4f}".format(stdev(scenario_slot_results[1:len(scenario_slot_results) - 1]))))

            # print("Results yielded:", scenario_intent_results, scenario_slot_results)

            config.writer.append_row(scenario_slot_results, spreadsheet_row_index)
            config.writer.append_row(scenario_intent_results, spreadsheet_row_index)
            spreadsheet_row_index += 3

    config.writer.save()


def _create_analysis_archive(config: SettingConfig) -> None:
    """
    Creates a ZIP archieve from the merged reports of the current run.
    :param config: Configuration of the current run.
    :return: None
    """
    zip_path = os.path.join(config.archives_path, f"{config.identifier}_{time.time()}.zip")
    with ZipFile(file=zip_path, mode='w') as zipObj:
        # Iterate over all the files in directory
        for folderName, subfolders, filenames in os.walk(config.merged_reports_root):
            for filename in filenames:
                # create complete filepath of file in directory
                filepath = os.path.join(folderName, filename)
                # Add file to zip
                zipObj.write(filepath)
    zipObj.close()


def rasa_pipeline(config_path: str) -> None:
    """
    Driver function for RASA NLU base pipeline.
    :param config_path: Path to configuration file.
    :return: None
    """
    config = SettingConfig(config_path)
    _setup_folders(config)
    # Keep the same splits.
    # _wipe_splits(config)
    _wipe_reports(config)
    # _create_splits(config)
    _process_datasets(config)
    _create_analysis_archive(config)


def main():
    print("Starting the evaluation pipeline (pre-processing included)...")
    try:
        config_filepath = sys.argv[1]
    except IndexError:
        print("\nNo argument for the configuration file provided.Using the default configuration file.")
        config_filepath = "parameters.cfg"
    rasa_pipeline(config_filepath)


if __name__ == "__main__":
    main()
