import io


def generate_antonym_confusions_report(dump_path: str) -> None:
    """
    Prints error analaytics for antonyms from an intent errors dump generated from the RASA pipeline.
    :param dump_path: Path to the intent dumps.
    :return: None, just prints the errors to stdout.
    """
    # List of dicts. Index in list = scenario. Dict key :error (Pred/Act), dict value: nr of encounters in scenario
    antonym_errors = [
        # Scenario 0
        {"aprindeLumina/stingeLumina": 0,
         "stingeLumina/aprindeLumina": 0,
         "cresteIntensitateLumina/scadeIntensitateLumina": 0,
         "scadeIntensitateLumina/cresteIntensitateLumina": 0,
         "cresteTemperatura/scadeTemperatura": 0,
         "scadeTemperatura/cresteTemperatura": 0,
         "pornesteTV/opresteTV": 0,
         "opresteTV/pornesteTV": 0,
         "cresteIntensitateMuzica/scadeIntensitateMuzica": 0,
         "scadeIntensitateMuzica/cresteIntensitateMuzica": 0},

        # Scenario 1
        {"aprindeLumina/stingeLumina": 0,
         "stingeLumina/aprindeLumina": 0,
         "cresteIntensitateLumina/scadeIntensitateLumina": 0,
         "scadeIntensitateLumina/cresteIntensitateLumina": 0,
         "cresteTemperatura/scadeTemperatura": 0,
         "scadeTemperatura/cresteTemperatura": 0,
         "pornesteTV/opresteTV": 0,
         "opresteTV/pornesteTV": 0,
         "cresteIntensitateMuzica/scadeIntensitateMuzica": 0,
         "scadeIntensitateMuzica/cresteIntensitateMuzica": 0},

        # Scenario 2
        {"aprindeLumina/stingeLumina": 0,
         "stingeLumina/aprindeLumina": 0,
         "cresteIntensitateLumina/scadeIntensitateLumina": 0,
         "scadeIntensitateLumina/cresteIntensitateLumina": 0,
         "cresteTemperatura/scadeTemperatura": 0,
         "scadeTemperatura/cresteTemperatura": 0,
         "pornesteTV/opresteTV": 0,
         "opresteTV/pornesteTV": 0,
         "cresteIntensitateMuzica/scadeIntensitateMuzica": 0,
         "scadeIntensitateMuzica/cresteIntensitateMuzica": 0},

        # Scenario 3.1
        {"aprindeLumina/stingeLumina": 0,
         "stingeLumina/aprindeLumina": 0,
         "cresteIntensitateLumina/scadeIntensitateLumina": 0,
         "scadeIntensitateLumina/cresteIntensitateLumina": 0,
         "cresteTemperatura/scadeTemperatura": 0,
         "scadeTemperatura/cresteTemperatura": 0,
         "pornesteTV/opresteTV": 0,
         "opresteTV/pornesteTV": 0,
         "cresteIntensitateMuzica/scadeIntensitateMuzica": 0,
         "scadeIntensitateMuzica/cresteIntensitateMuzica": 0},

        # Scenario 3.2
        {"aprindeLumina/stingeLumina": 0,
         "stingeLumina/aprindeLumina": 0,
         "cresteIntensitateLumina/scadeIntensitateLumina": 0,
         "scadeIntensitateLumina/cresteIntensitateLumina": 0,
         "cresteTemperatura/scadeTemperatura": 0,
         "scadeTemperatura/cresteTemperatura": 0,
         "pornesteTV/opresteTV": 0,
         "opresteTV/pornesteTV": 0,
         "cresteIntensitateMuzica/scadeIntensitateMuzica": 0,
         "scadeIntensitateMuzica/cresteIntensitateMuzica": 0},

        # Scenario 3.3
        {"aprindeLumina/stingeLumina": 0,
         "stingeLumina/aprindeLumina": 0,
         "cresteIntensitateLumina/scadeIntensitateLumina": 0,
         "scadeIntensitateLumina/cresteIntensitateLumina": 0,
         "cresteTemperatura/scadeTemperatura": 0,
         "scadeTemperatura/cresteTemperatura": 0,
         "pornesteTV/opresteTV": 0,
         "opresteTV/pornesteTV": 0,
         "cresteIntensitateMuzica/scadeIntensitateMuzica": 0,
         "scadeIntensitateMuzica/cresteIntensitateMuzica": 0},
    ]

    keys = antonym_errors[0].keys()
    # Parse a human readable format rasa result. Don't shoot me for not using the json
    with io.open(file=dump_path, mode="r", encoding="utf-8") as input_file:
        current_scenario_index = 0
        current_scenario = "scenario_0"
        for line in input_file:

            # If scenario header line
            if 'Intent Errors report' in line:
                scenario = line[line.index('scenario'):line.index(',')]
                if scenario > current_scenario:
                    current_scenario_index += 1
                    current_scenario = scenario

            # If regular error line
            if 'Predicted' in line:
                split_line = line.strip("\t").strip().split(" ")
                first_intent = split_line[1].replace(".", "")
                second_intet = split_line[3].replace(".", "")
                error = f"{first_intent}/{second_intet}"
                if error in keys:
                    antonym_errors[current_scenario_index][error] += 1
        input_file.close()
    for scenario in antonym_errors:
        for k, v in scenario.items():
            scenario[k] = int(v / 5)

    for index, _scenario in enumerate(antonym_errors):
        print(f"Scenario {index}")
        for k, v in _scenario.items():
            print(f"\t{k} {v}")


if __name__ == '__main__':
    generate_antonym_confusions_report(
        "C:\\Uni\\Thesis\\radandreicristian\\nlu\\rad-antonyms\\results\\analysis_archives\\same_splits\\augmentedmaxdist_samesplits\\merged_reports\\intent_errors_merged.txt")
