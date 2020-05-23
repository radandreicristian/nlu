# SEMANTIC AUGMENTER, RASA NLU PIPELINE

## Overall Project Strucutre and Configuration
The project contains Python scripts used in the pipeline, tools, linguistinc data and task-specific training and testing data.
 
 The 'datasets' folder contains structured RASA NLU training and testing JSON files. 

The 'lang' folder contains linguistic data and specifications, such as synonym/antonym thesauri, vocabularies and **SHOULD ALSO INCLUDE WORD VECTORS** (which are too large for a VCS), in lang/vectors. (See VEC_PATH in parameters.cfg or Additional Files section)

The 'utils' folder contains helper scripts 

The root folder contains RASA NLU data and the main scripts used in the pipeline.

## Pipeline Description

The pipeline incorporates 5 main steps.

* Semantic Relations Generator.
* Verb Exctraction and Augmenting.
* Counterfitting.
* SpaCy Model Creation.
* RASA Training and Inference.

### Relations Generator 
The /relations_generator.py script generates synonym and antonym pairs from RoWordNet and stores them, according to their part-of-speech. The subfolders corresponding to each of the four parts of speech, verb, adverb, noun, adjective, will be generated under the path specified by CONSTRAINTS_ROOT_PATH in parameters.cfg.
Once generated, the pairs can be used in the next components of the pipeline.

### Verb Extraction and Augmenting
In util/semantic_augmenter.py are the functions responsible for augmenting the antonym verb pairs.
* Verb lemma & conjugation times extraction.
* Synset construction.
* Conjugation of pairs and appending to the initial pairs.

### Counterfitting - Parameters
Counterfitting runs can be parametrized from the ./parameters.cfg file. 
Before running counterfitting, please add the original vectors file in lang/vectors or change its path in ./parameters.cfg. As a command line parameter, it takes the name of the language (an identifier) for the newly generated verbs. Running counterfitting will modify the word vectors to be according to the linguistic constraints provided by the previous steps of the pipeline. Counterfitting generates a comparative analysis of certain pairs and stores the modified vectors.

### SpaCy Mode Createion
A language model based on the vectors created by the previous step is setup by util/init_spacy_lang.py script. If used individually, parameters for output paths, language name and identifier and vectors locations are to be provided. These parameters are otherwise automatically provided by the context and previous steps of the pipeline. After this step, you should be able to load the model in a RASA run, using pretrained_embeddings_spacy and the generated language.

### RASA Training and Inference
The final steps of the pipeline is to train and test a RASA NLU model, based on the language model provided and training data. /rasa_pipeline.py handles the last steps, as well as calls to the previous components.


### Pipeline Configuration
There is a single common configuration file for the whole pipeline, which includes paths used throughout the project and settings parameters/hyperparameters for some components.
The paths section includes paths relative to the project root where the input / output files of the run are stored.


### Additional Files
Some files are not in this repository, but will be needed.
* Google spreadsheet secret JSON: (Follow the tutorial on https://bit.ly/3c101g6) Save the JSON secret in the folder, then change its name / configure it in rasa_pipeline.py.
* Romanian base SpaCy model with PoS Tagger / Parser (available here: https://bit.ly/36qsomO). Store this preferably in models/out/ and then perform spacy linking step:
 
```spacy link **ABSOLUTE_PATH_TO_THE_EXTRACTED_FOLDER/ro_model-0.0.0/ro_model** ro_ft_300``` 
* Romanian base word vectors (fastText - available here https://bit.ly/3cYdcQi) unarchieve and add to lang/vectors.

### Running:
``` pip install requirements.txt  ```

``` python ./rasa_pipeline.py```
