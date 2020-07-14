import argparse
import os
import pathlib
import subprocess
import sys


def is_venv():
    """
    :return: True if the script is ran inside a VENV, Flase otherwise.
    """
    return hasattr(sys, 'real_prefix') or sys.base_prefix != sys.prefix


def reactivate_venv(root_path: str) -> None:
    """
    Re-activates the virtual environment.
    :param root_path: Root path of the project.
    :return: None
    """
    activate_abs_path = os.path.join(pathlib.Path(root_path).absolute(), "venv", "Scripts")
    os.chdir(activate_abs_path)
    subprocess.Popen(["activate.bat"])


def init_argument_parser() -> argparse.ArgumentParser:
    """
    Initailizes the argument parser for command line usage.
    :return: An ArgumentParser objects that knows how to parse specific parameters.
    """
    parser = argparse.ArgumentParser(description='Automated SpaCy language model creation')

    # Language - ISO code of the language
    parser.add_argument(
        '-l', '--language',
        action='store', nargs='?', type=str)

    # Model location - relative path from the root to the location of the model
    parser.add_argument(
        '-m', '--model_loc',
        action='store', nargs='?', type=str
    )

    # Vectors location - relative path from the root to the location of vectors:
    parser.add_argument(
        '-v', '--vectors_loc',
        action='store', nargs='?', type=str
    )

    # Model output location - relative path from the root to the location of the model output
    parser.add_argument(
        '-o', '--output_loc',
        action='store', nargs='?', type=str
    )

    # SpaCy langauge name
    parser.add_argument(
        '-n', '--name',
        action='store', nargs='?', type=str
    )

    parser.add_argument(
        '-f', '--overwrite',
        action='store_true'
    )

    return parser


def init_language_model(executable_path: str, language_code: str, model_path: str, vectors_path: str,
                        force_overwrite: bool) -> None:
    """
    Performs the spacy init step.
    :param executable_path: Path to the python executable.
    :param language_code: ISO code of the language.
    :param model_path: Path to where the model should be created. Personal recomandation: {project_root}/models
    :param vectors_path: Path to where the vectors are stored. Supposedly {project_root}/lang/vectors/*.vec
    :param force_overwrite: True if the user is okay with overwriting existing models.
    :return: None
    """
    model_initialization_command = [executable_path, '-m', 'spacy', 'init-model', language_code, model_path, '-v',
                                    vectors_path]

    if not os.path.isdir(model_path):
        print("Model directory does not exist. Creating.")
        pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)

    if len(os.listdir(model_path)) > 0 and not force_overwrite:
        answer = input("Model path not empty. Contents may be overwritten. Proceed? (y/n)\n")
        if answer in ['Yes', 'yes', 'Y', 'y']:
            print("Overwriting content...")
        elif answer in ['No', 'no', 'N', 'n']:
            print("Overwriting declined. Aborting.")
            return
        else:
            print("Invalid response (expected y/n, you donkey). Aborting.")
            return

    try:
        subprocess.Popen(model_initialization_command, shell=True).wait()
    except OSError or IOError:
        print("Error during the previous subprocess call")
        raise


def package_language_model(executable_path: str, model_path: str, model_output_path: str,
                           force_overwrite: bool) -> None:
    """
    Performs spacy package step.
    :param executable_path: Path to the python executable.
    :param model_path: Path to where the model should be created. Personal recommendation: {project_root}/models
    :param model_output_path: Path to where the packaged model shoudl be saved. Persona recommendation
        {project_root}/models/out
    :param force_overwrite: True if the user is okay with overwriting existing models.
    :return:
    """
    model_packaging_command = [executable_path, '-m', 'spacy', 'package', model_path, model_output_path]

    if force_overwrite:
        model_packaging_command.append("--force")

    if not os.path.isdir(model_output_path):
        print("Model output directory does not exist. Creating.")
        pathlib.Path(model_output_path).mkdir(parents=True, exist_ok=True)

    if len(os.listdir(model_output_path)) > 0 and not force_overwrite:
        answer = input("Model output path not empty. Contents may be overwritten. Proceed? (y/n)\n")
        if answer in ['Yes', 'yes', 'Y', 'y']:
            print("Overwriting content...")
        elif answer in ['No', 'no', 'N', 'n']:
            print("Overwriting declined. Aborting.")
            return
        else:
            print("Invalid response (expected y/n, you donkey). Aborting.")
            return

    try:
        subprocess.Popen(model_packaging_command, shell=True).wait()
    except OSError or IOError:
        print("Error during the previous subprocess call")
        raise


def setup_language_model(executable_path: str, setup_script_path: str) -> None:
    """
    Performs spacy setup setp.
    :param executable_path: Path to the python executable.
    :param setup_script_path: Path to the setup script.
    :return: None
    """
    os.chdir(setup_script_path)
    model_setup_command = [executable_path, 'setup.py', 'sdist']
    try:
        subprocess.Popen(model_setup_command).wait()
    except OSError or IOError:
        print("Error during the previous subprocess call")
        raise


def install_language_model(setup_script_path: str, language_model_identifier: str) -> None:
    """
    Pefroms spacy language installation via PIP.
    :param setup_script_path: Path to the setup script.
    :param language_model_identifier: Identifier for the langauge model
        (usually [isocode]-[version.subversion.subsubverison].
    :return: None
    """
    distribution_path = os.path.join(setup_script_path, 'dist')
    os.chdir(distribution_path)
    installation_command = ['pip', 'install', f"{language_model_identifier}.tar.gz"]
    try:
        subprocess.Popen(installation_command).wait()
    except OSError or IOError:
        print("Error during the previous subprocess call")
        raise


def link_language_model(executable_path: str, root_path: str, generated_model_path, language_name) -> None:
    """
    Peforms spacy language linking.
    IMPORTANT: THE EXECUTING TERMINAL / CONSOLE / SHELL SHOULD HAVE ADMIN PERMISSION FOR OS LINKING
    :param executable_path: Path of the python executable.
    :param root_path: Root path of the project.
    :param generated_model_path: Path to the generated model.
    :param language_name: Language unique identifier.
    :return:
    """
    linking_command = [executable_path, '-m', 'spacy', 'link', generated_model_path, language_name]
    try:
        # Re-activate the venv?
        reactivate_venv(root_path)
        subprocess.Popen(linking_command).wait()
    except OSError or IOError:
        print("Error during the previous subprocess call")
        raise


def create_language_model(root_path: str, language_code: str, model_path: str, vectors_path: str,
                          model_output_path: str, language_name: str, force_overwrite: bool) -> None:
    """
    :param language_code: ISO code of the language (i.e. ro, en, es)
    :param model_path: Path where the language model will be stored (directory)
    :param vectors_path: Path wehere the vectors are stored (file, usually .vec or .bin)
    :param model_output_path: Path where the language model output will be generated
    :param language_name: The unique identifier for the language
    :param force_overwrite: Skips overwriting questions/warnings if set

    The following function executes and waits for the following shell commands, resulting in a spacy language model
    (venv) python -m spacy init-model [language_code] [model_path] --vectors_loc [vectors_loc]
    (venv) python -m spacy package [model_path] [output_path]
    (venv) cd [output_path]\{[language_code]_model-0.0.0}
    (venv) python setup.py sdist
    (venv) cd .\dist\
    (venv) pip install .\{[language_code]_model-0.0.0.tar.gz}

    (venv) [Admin] python -m spacy link [output_path]\{[language_code]_model-0.0.0}\{[language_code]_model}
    """

    executable = sys.executable

    # Perform the model initialization:
    print("Initializing language model...")
    init_language_model(executable, language_code, model_path, vectors_path, force_overwrite)
    print("Successfully initialized the language model.")

    # Perform packaging step
    print("Packaging language model...")
    package_language_model(executable, model_path, model_output_path, force_overwrite)
    print("Successfully packaged the language model.")

    # Perform the setup step:
    print("Performing setup step...")
    language_model_identifier = f"{language_code}_model-0.0.0"
    setup_script_path = os.path.join(model_output_path, language_model_identifier)
    setup_language_model(executable, setup_script_path)
    print("Finished setup step.")

    # Performing installation step:
    print("Performing model installation step...")
    install_language_model(setup_script_path, language_model_identifier)
    print("Finished installation step.")

    # Performing linking step:
    print("Performing linking step...")
    generated_model_path = os.path.join(setup_script_path, f"{language_code}_model")
    link_language_model(executable, root_path, generated_model_path, language_name)
    print("Succesfully linked. Model is now usable via spacy")


def main():
    arg_parser = init_argument_parser()

    # Set absolute path to the root of this project
    file_path = pathlib.Path(__file__).parent.absolute()
    root_path = pathlib.Path(file_path).parent.absolute()

    arguments = arg_parser.parse_args()

    lang_code = arguments.language
    if not lang_code:
        print("Language code (-l) not supplied.")

    model = os.path.join(root_path, arguments.model_loc)
    if not model:
        print("Path to where the model should be initialized (-m) not supplied.")

    vectors = os.path.join(root_path, arguments.vectors_loc)
    if not vectors:
        print("Path to the location of the word vectors (-v) not supplied.")

    model_out = os.path.join(root_path, arguments.output_loc)

    if not model_out:
        print("Path to the location of the output path (-o) not supplied.")

    lang_name = arguments.name
    if not lang_name:
        print("Name of the spaCy language model (-n) not supplied.")

    if not (lang_code and model_out and model and vectors and lang_name):
        print("Usage: python init_spacy_lang.py -l LANG -m MODELS_LOC -v VECTORS_LOC -o OUTPUT_LOC -n LANG_MODEL_NAME")

    overwrite = arguments.overwrite or False
    # Extra check: Make sure we are inside a virtual environment (for safety measures, but can be removed)
    if not is_venv():
        print('Not inside a VENV. Aborting.')
        return

    create_language_model(root_path, lang_code, model, vectors, model_out, lang_name, overwrite)


if __name__ == '__main__':
    main()
