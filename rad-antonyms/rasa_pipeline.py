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
from typing import Optional, Any
from zipfile import ZipFile

import gspread
import yaml
from gspread import Worksheet
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.utils import resample

from util.semantic_augmenter import extract_verb_lemmas, augment_synonym_verbs, augment_antonym_verbs
from util.tools import copy_path, extract_sentences_from_file, save_pairs_to_file, get_time

SPREADSHEET_START_VERTICAL_OFFSET = 3


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

        # Load Hyperparameters
        self.splits = self.config.getint("hyperparameters", "splits")
        self.sample_percent = self.config.getfloat("hyperparameters", "sample_percent")

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

        # Load merged reports path
        self.merged_reports_root = self.config.get("paths", "MERGED_REPORTS_ROOT")
        self.merged_intent_report_path = self.config.get("paths", "MERGED_INTENT_REPORT_PATH")
        self.merged_intent_errors_path = self.config.get("paths", "MERGED_INTENT_ERRORS_PATH")
        self.merged_slot_report_path = self.config.get("paths", "MERGED_SLOT_REPORT_PATH")
        self.merged_slot_errors_path = self.config.get("paths", "MERGED_SLOT_ERRORS_PATH")
        self.merged_matrices_path = self.config.get("paths", "MERGED_MATRICES_PATH")

        self.archives_path = self.config.get("paths", "ARCHIEVES_PATH")

        with io.open(self.rasa_config_absolute_path, "r") as rasa_config_file:
            rasa_config = yaml.load(rasa_config_file, Loader=yaml.FullLoader)
            self.language = rasa_config['language']

        self.identifier = f"{self.language}_{'diac' if self.diacritics else 'nodiac'}"
        self.base_language_model = self.config.get("settings", "BASE_LANGAUGE")
        self.language_code = self.config.get("settings", "LANGUAGE_CODE")
        self.spreadsheet_name = self.config.get("settings", "GOOGLE_SPREADSHEET_NAME")


def update_spacy_language(config: SettingConfig, language: str):
    with io.open(file=config.rasa_config_absolute_path, mode="r", encoding="utf-8") as rasa_config_file:
        rasa_config = yaml.load(rasa_config_file, Loader=yaml.FullLoader)
        rasa_config_file.close()

    rasa_config['language'] = language

    with io.open(file=config.rasa_config_absolute_path, mode="w", encoding="utf-8") as rasa_config_file:
        yaml.dump(rasa_config, rasa_config_file)
        rasa_config_file.close()


def create_single_split(base_path: str, split_identifier: int, sample_percent: float, train: bool) -> None:
    """
    Creates a single train / test split from a base train / test file in RASA NLU JSON format.
    :param base_path: Path of the original file.
    :param split_identifier: Identifier of the split.
    :param sample_percent: The percentage of examples to be sampled.
    :param train: Sample from / save as train file if True, as test file otherwise.
    :return: None.
    """
    path = base_path + ("_train.json" if train else "_test.json")
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
        write_path = base_path + ("_train_" if train else "_test_") + f"{split_identifier}" + ".json"
        file = io.open(write_path, "w", encoding="utf-8")

        # Dump JSON content into the file
        json.dump(content_copy, file, ensure_ascii=False)
        file.close()


def create_splits(config: SettingConfig) -> None:
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

            # Compute the file path - File path has to start with "scenario_$nr"
            file_path = os.path.join(scenario_folder_path, file)

            # For each split, create the dataset split and save it
            for split_id in range(config.splits):
                # Create splits for train / test datasets
                create_single_split(file_path, split_id, train=True, sample_percent=config.sample_percent)
                create_single_split(file_path, split_id, train=False, sample_percent=config.sample_percent)
    print("Finished creating fresh dataset splits")


def wipe_reports(config: SettingConfig) -> None:
    """
    Deletes the reports generated by previous runs.
    :param config: Configuration of the current run.
    :return: None.
    """
    # Wipe the merged intent and slot errors and reports
    merged_reports_paths = [config.merged_intent_report_path, config.merged_intent_errors_path,
                            config.merged_slot_report_path, config.merged_slot_errors_path]
    for file in merged_reports_paths:
        io.open(file, "w", encoding="utf-8").close()

    # Wipe the current errors and reports
    reports_paths = [config.intent_report_path, config.intent_errors_path, config.slot_report_path,
                     config.slot_errors_path]
    for file in reports_paths:
        io.open(file, "w", encoding="utf-8").close()

    # Delete all report directories
    for scenario_folder in os.listdir(config.datasets_path):
        scenario_path = os.path.join(config.datasets_path, scenario_folder)
        if os.path.isdir(scenario_path):
            for file in os.listdir(scenario_path):
                file_path = os.path.join(scenario_path, file)
                if os.path.isdir(file_path):
                    rmtree(file_path)


def wipe_splits(config: SettingConfig) -> None:
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

            # Do not proceed if the scenario does only contain the train and test file
            if len(os.listdir(scenario_folder_path)) <= 2:
                print('Directory contains at most train and test splits. Breaking.')
                return

            # Compute the file path - File path has to start with "scenario_$nr"
            file_path = os.path.join(scenario_folder_path, file)

            # Compute the file name for the splits, remove them
            for split_id in range(config.splits):
                train_split = file_path + "_train_" + f"{split_id}" + ".json"
                test_split = file_path + "_test_" + f"{split_id}" + ".json"
                os.remove(train_split)
                os.remove(test_split)
    print("Finished wiping the splits")


def process_intent_result(identifier: str, scenario_report_path: str, config: SettingConfig) -> float:
    """
    Processes the intent detection results, writing them in a human readable format in a merged report.
    :param identifier: Identifier of the scenario/split.
    :param scenario_report_path: Path to the report for the current scenario.
    :param config: Configuration of the current run.
    :return: The weighted average F1 score.
    """
    #  Move intent report result to the merged report
    with io.open(config.intent_report_path, "r", encoding="utf-8") as report_file:

        # Open the merged intent report file for writing
        with io.open(config.merged_intent_report_path, "a", encoding="utf-8") as output_file:
            # Load the content
            content = json.load(report_file)
            weighted_average_f1 = content['weighted avg']['f1-score']

            # Write the weighted average for intent detection in the file
            output_file.write(
                "F1-Score - ID for "
                + identifier + f":{weighted_average_f1}\n")

            output_file.close()
        report_file.close()

    # Move intent errors to the merged errors
    with io.open(config.intent_errors_path, "r", encoding="utf-8") as report_file:
        # Open the merged errors report file for writing
        with io.open(config.merged_intent_errors_path, "a", encoding="utf-8") as output_file:
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

    return weighted_average_f1


def process_slot_result(identifier: str, scenario_report_path: str, config: SettingConfig) -> float:
    """
    Processes the slot filling results, writing them in a human readable format in a merged report.
    :param identifier: Identifier of the scenario/split.
    :param scenario_report_path: Path to the report for the current scenario.
    :param config: Configuration of the current run.
    :return: The weighted average F1 score.
    """
    # Move slot report result to the merged report
    with io.open(config.slot_report_path, "r", encoding="utf-8") as report_file:
        # Open the merged slot report file for appending
        with io.open(config.merged_slot_report_path, "a", encoding="utf-8") as output_file:
            # Load the content
            content = json.load(report_file)

            # Write the weighted average for intent detection in the file
            weighted_average_f1 = content['weighted avg']['f1-score']
            output_file.write(
                "F1-Score - SF for "
                + identifier + f":{weighted_average_f1}\n")

            output_file.close()
        report_file.close()

    # Move slot errors to the merged errors
    with io.open(config.slot_errors_path, "r", encoding="utf-8") as report_file:
        # Open the merged errors report file for writing
        with io.open(config.merged_slot_errors_path, "a", encoding="utf-8") as output_file:
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
    return weighted_average_f1


def copy_confusion_matrix(identifier: str, config: SettingConfig) -> None:
    """
    Copy a confusion matrix to the merged folder.
    :param identifier: Identifier of the scenario / split.
    :param config: Configuration of the current run.
    :return: None.
    """
    copyfile(config.conf_mat_path,
             os.path.join(config.merged_matrices_path,
                          identifier.replace(" ", "").replace(",", "").replace("_", "") + ".png"))


def process_split(file: str, file_path: str, split_id: int, config: SettingConfig, scenario_reports_path: str,
                  scenario_slot_results: list, scenario_intent_results: list) -> None:
    """
    Peforms a full pipeline run for a singple split. This includes extracting verbs, and modes, augmenting,
    counterfitting, spacy model pipeline,  training and evaluation.
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
    train_split = f"{file_path}_train_{split_id}.json"
    test_split = f"{file_path}_test_{split_id}.json"

    # Extract verbs from train
    print(f"Extracting train sentences from train data at {get_time()}")
    train_sentences = extract_sentences_from_file(train_split)

    # Augment -> save verb pairs separately
    print(f"Exacting lemmas and times from sentences at {get_time()}")
    verb_lemmas, verb_times = extract_verb_lemmas(config.base_language_model, train_sentences)

    print(f"Augmenting antonym pairs at {get_time()}")
    train_verb_antonym_pairs = augment_antonym_verbs(verb_lemmas, verb_times)

    print(f"Augmenting synonym pairs at {get_time()}")
    train_verb_syonym_pairs = augment_synonym_verbs(verb_lemmas, verb_times)

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
    timestamp = str(time.time()).split('.')[0]
    model_identifier = f"ro_ft_300_train_temp_{timestamp}"

    # Run counterfitting
    print(f"Started counterfitting at {get_time()}")
    counterfitting_command = ['python', 'counterfitting.py', '-l', model_identifier]
    subprocess.Popen(counterfitting_command, shell=True).wait()

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
    print(f"Creating spacy model at {get_time()}")
    subprocess.Popen(spacy_command, shell=True).wait()

    # Update rasa config path:
    os.chdir(root_path)

    print(f"Updating RASA config language model at {get_time()}")
    update_spacy_language(config, language_model_name)

    # Run the subprocess for RASA training and testing, and wait for its completion
    print(f"Running RASA training and testing at {get_time()}")
    command = [config.rasa_script_path, train_split, test_split]
    print(command)
    subprocess.Popen(command, shell=True).wait()

    # Process the slot and intent errors & reports and save their return values
    print(f"Computing and writing results at {get_time()}")
    intent_f1 = process_intent_result(identifier, scenario_reports_path, config)
    slot_f1 = process_slot_result(identifier, scenario_reports_path, config)

    # Move the confusion matrix to the results path
    copy_confusion_matrix(identifier, config)

    scenario_slot_results.append(float("{:0.4f}".format(slot_f1)))
    scenario_intent_results.append(float("{:0.4f}".format(intent_f1)))

    # Restore backup
    print(f"Restoring backup files and deleting language model at {get_time()}")
    copy_path(backup_verb_synonyms_path, verb_synonyms_path, False)
    copy_path(backup_verb_antonyms_path, verb_antonyms_path, False)

    # Put the default value back in the rasa config
    update_spacy_language(config, config.base_language_model)

    # Remove models and vectors, since they will never be used again, in order to save a lot of memory.
    # Avoiding some weird race condition :/
    time.sleep(2)
    shutil.rmtree(model_path)

    time.sleep(2)
    shutil.rmtree(model_out_path)
    os.remove(os.path.join(root_path, vectors_path))

    print(f"Finished restoring backup files and deleting language model at {get_time()}")
    print(f"Finished processing split {identifier}")


def process_datasets(config: SettingConfig) -> None:
    """
    Processes a dataset with multiple splits, generating reports and results.
    :param config: Configuration of the current run.
    :param sheet: The object associated to the Google Sheet where the results will be written.
    :return: None
    """
    print("Started running RASA")

    # Compute and write the title of the spreadsheet based on the loaded configurations
    spreadsheet_title = [config.identifier]
    insert_row_authorized(config, spreadsheet_title, 1)

    # For each scenario folder
    spreadsheet_row_index = SPREADSHEET_START_VERTICAL_OFFSET
    for file in os.listdir(config.datasets_path):

        # Compute the path for the scenario folder
        folder_path = os.path.join(config.datasets_path, file)
        if os.path.isdir(folder_path):

            # Break if the directory does not contain the splits
            if len(os.listdir(folder_path)) <= 2:
                print("Directory only contains train and test files, but no splits. Breaking.")
                break

            # Compute the scenario file name
            file_path = os.path.join(folder_path, file)

            # Compute the reports path and create the directory
            scenario_reports_path = os.path.join(folder_path, 'scenario_reports')
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
                process_split(file, file_path, split_id, config, scenario_reports_path, scenario_slot_results,
                              scenario_intent_results)

            # Append the mean value to each list for the scenario
            scenario_intent_results.append(float("{:0.4f}".format(mean(scenario_intent_results[1:]))))
            scenario_slot_results.append(float("{:0.4f}".format(mean(scenario_slot_results[1:]))))

            # Append the standard deviation to each list for the scenario
            scenario_intent_results.append(
                float("{:0.3f}".format(stdev(scenario_intent_results[1:len(scenario_intent_results) - 2]))))
            scenario_slot_results.append(
                float("{:0.3f}".format(stdev(scenario_slot_results[1:len(scenario_slot_results) - 2]))))

            # Append the line in the google doc:
            # sheet.insert_row(scenario_slot_results, spreadsheet_row_index)
            insert_row_authorized(config, scenario_slot_results, spreadsheet_row_index)
            # sheet.insert_row(scenario_intent_results, spreadsheet_row_index)
            insert_row_authorized(config, scenario_intent_results, spreadsheet_row_index)
            spreadsheet_row_index += 3


def insert_row_authorized(config: SettingConfig, values: Any, index: int = 1):
    worksheet = _get_worksheet(config)
    worksheet.insert_row(values, index)


def _get_worksheet(config: SettingConfig) -> Optional[Worksheet]:
    """
    Creates the object associated to the first sheet in the provided Google Sheets spreadsheet.
    NOTE: This function always uses and overwrites the first sheet.
    TODO: Figure how to avoid this.
    :param config: The configuration of the current run.
    :return: Optionally, the Worksheet object associated, or None if it fails.
    """
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_name('thesis_secret.json', scope)
    client = gspread.authorize(credentials)
    return client.open(config.spreadsheet_name).sheet1


def create_analysis_archive(config: SettingConfig) -> None:
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
    Driver function for the RASA NLU Pipeline. Performs all the steps from cleaning files from previous runs to
    creating splits, processing them individually and reporting results.
    :param config_path:
    :return: None
    """
    config = SettingConfig(config_path)
    wipe_reports(config)
    wipe_splits(config)
    create_splits(config)
    process_datasets(config)
    create_analysis_archive(config)


def main():
    try:
        config_filepath = sys.argv[1]
    except IndexError:
        print("\nUsing the default config files")
        config_filepath = "parameters.cfg"
    rasa_pipeline(config_filepath)


if __name__ == "__main__":
    main()
