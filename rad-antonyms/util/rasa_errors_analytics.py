import io


def analyze_antonyms(dump_path: str) -> None:
    # List of dicts. Index in list = scenario. Dict key :error, dict value: nr of encounters in scenario
    antonym_errors = [{"aprinde/stinge_lumina": 0,
                       "creste/scade_lumina": 0,
                       "creste/scade_temp": 0,
                       "porneste/opreste_tv": 0},
                      {"aprinde/stinge_lumina": 0,
                       "creste/scade_lumina": 0,
                       "creste/scade_temp": 0,
                       "porneste/opreste_tv": 0},
                      {"aprinde/stinge_lumina": 0,
                       "creste/scade_lumina": 0,
                       "creste/scade_temp": 0,
                       "porneste/opreste_tv": 0},
                      {"aprinde/stinge_lumina": 0,
                       "creste/scade_lumina": 0,
                       "creste/scade_temp": 0,
                       "porneste/opreste_tv": 0},
                      {"aprinde/stinge_lumina": 0,
                       "creste/scade_lumina": 0,
                       "creste/scade_temp": 0,
                       "porneste/opreste_tv": 0},
                      {"aprinde/stinge_lumina": 0,
                       "creste/scade_lumina": 0,
                       "creste/scade_temp": 0,
                       "porneste/opreste_tv": 0},
                      ]
    light_intents = ['aprindeLumina', 'stingeLumina']

    light_intensity_intents = ['cresteIntensitateLumina', 'scadeIntensitateLumina']

    temperature_intents = ['cresteTemperatura', 'scadeTemperatura']

    tv_intents = ['opresteTV', 'pornesteTV']

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
                split_line = line.split(" ")
                first_intent = split_line[4].replace(".", "")
                second_intet = split_line[6].replace(".", "")
                if first_intent in light_intents and second_intet in light_intents:
                    antonym_errors[current_scenario_index]['aprinde/stinge_lumina'] += 1
                if first_intent in light_intensity_intents and second_intet in light_intensity_intents:
                    antonym_errors[current_scenario_index]['creste/scade_lumina'] += 1
                if first_intent in temperature_intents and second_intet in temperature_intents:
                    antonym_errors[current_scenario_index]['creste/scade_temp'] += 1
                if first_intent in tv_intents and second_intet in tv_intents:
                    antonym_errors[current_scenario_index]['porneste/opreste_tv'] += 1

        input_file.close()
    for scenario in antonym_errors:
        for k, v in scenario.items():
            scenario[k] = int(v / 5)

    print(antonym_errors)


if __name__ == '__main__':
    analyze_antonyms("C:/Uni/Thesis/intent_errors_merged.txt")
