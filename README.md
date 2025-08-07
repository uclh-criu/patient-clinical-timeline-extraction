# Patient Clinical Timeline Extraction

This repository provides a set of tools for constructing patient clinical timelines from free text in Electronic Healthcare Records (EHRs).

You can choose between six different methods for relation extraction between clinical entities and dates, which are used to construct the timelines. 

These are:
1. Naive heuristic model - see `naive_extractor.py`.
2. CNN + BiLSTM - see `custom_extractor.py`.
3. Finetuned BioBERT - see `bert_extractor.py`.
4. Llama-3.2-3B - see `llama_extractor.py`.
5. OpenAI - see `openai_extractor.py`.
6. RelCAT - see `relcat_extractor.py`.

You are able to customise the behaviour of each extraction method, and you can also retrain methods 2 and 3 using your own data.

The repository provides all the functionality you need to run experiments, calculate metrics, produce plots and log results on your own datasets.

## Data

At present the system only works with data in CSV format. 

We assume that you have already done Named Entity Recognition (e.g. using [MedCAT](https://github.com/CogStack/cogstack-nlp)) and absolute date detection (e.g. using [AnonCAT](https://github.com/antsh3k/deidentify)) which are prerequisites for the relation extraction task.

Create a csv file with your data and place it in the data folder in the root of the repository. The file should have the following columns and formats:
- `patient`: Patient ID
- `note_id`: Unique note identifier  
- `note`: Clinical note text
- `document_timestamp`: Document date (YYYY-MM-DD)
- `extracted_entities`: JSON array of entities with positions
- `formatted_dates`: JSON array of dates (original and YYYY-MM-DD parsed format) with positions
- `relationship_gold`: JSON array of correct entity-date relationships for evaluation

The exact name of the columns you are using can be configured in `config.py`.

A sample file `data/synthetic.csv` is provided as an example of the format required.

## Usage

### Get Started

1. Clone the repo
2. Put your csv file inside the data folder
3. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Inference

1. Change `config.py` to point to right file paths and parameters - this is where you define the experiment:
*   `DATA_SOURCE`: Choose the data source (`'synthetic'`, `'synthetic_multi'`, `'imaging'`, `'nph'`).
*   `ENTITY_MODE`: Specify the entity types to be considered (`'diagnosis_only'`, `'multi_entity'`).
*   `EXTRACTION_METHOD`: Choose the method to use (`'naive'`, `'custom'`, `'bert'`, `'relcat'`, `'openai'`, `'llama'`).


2. Run the inference pipeline 
```bash
python run_inference_evals.py
```

When you run the inference pipeline, the script will:
1. Generate predictions and save these to a new column in your original data CSV file. 
2. Calculate and display evaluation metrics (Precision, Recall, F1-score) by comparing the predictions to the `relationship_gold` column.
3. Save the detailed metrics, confusion matrices, and other experiment outputs to the `experiment_outputs/` directory.
4. Log a summary of the run, including the configuration and final metrics, to `inference_log.csv`.

Note: the above should work 'out of the box' for the naive method but the other methods require some setup first (see below).

### Extractor Customisation & Configuration

#### Custom Extractor

If you want to use or retrain the `'custom'` method, edit `custom_model_training/training_config_custom.py` to set the training configuration and the hyperparameters to test as part of a grid search. 

You can also adapt the model architecture within `custom_model_training/DiagnosisDateRelationModel.py`.

You will also need to specify the path to a vocabulary file to use when training. One option is to build a custom vocabulary from your own dataset:
```bash
python custom_model_training/build_vocab.py
```

Then run training:
```bash
python custom_model_training/train_custom_model.py
```

#### BioBERT

To use or retrain the Finetuned BioBERT model, first edit the configuration in `bert_model_training/training_config_bert.py`.

Then, run the training script:
```bash
python bert_model_training/train_bert_model.py
```
This will save the finetuned model to the path specified in the config, ready for inference.

#### Llama-3.2-3B

For Llama-3.2 support, ensure you have the model downloaded in the specified path (see `LLAMA_MODEL_PATH` in `config.py`) and that you have access to the model via HuggingFace. The model can be downloaded from: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct.

#### OpenAI

For OpenAI support, create a .env file in the root of the repository and set an `OPENAI_API_KEY` variable. Note that OpenAI models should only be used with synthetic data.
```bash
OPENAI_API_KEY = 'your-api-key-here'  # Set OPENAI_API_KEY here
```

You can specify which OpenAI model to use by setting the `OPENAI_MODEL` variable in `config.py`. The default is `'gpt-4o'`, but you can change this to any other compatible model, such as `'gpt-3.5-turbo'`.

#### RelCAT

The RelCAT extractor uses a pre-trained model. There is no training script for it in this repository. To use it, ensure the paths to the model (`MEDCAT_MODEL_PATH`) and the concept database (`MEDCAT_CDB_PATH`) are correctly specified in `config.py`.
