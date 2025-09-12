# Patient Clinical Timeline Extraction

This repository provides a set of tools for constructing patient clinical timelines from free text in Electronic Healthcare Records (EHRs).

You can choose between four different methods for relation extraction between clinical entities and dates, which are used to construct the timelines. 

These are:
1. Naive heuristic model
2. Finetuned BERT
3. Llama-3.2-3B
4. RelCAT

The repository provides utility functions for these extraction methods, and notebooks showing how to run experiments.

## Data Pipelines

### Training Pipeline (MedCAT Trainer)
1. Upload a CSV file with columns `name` and `text` to [MedCAT Trainer](https://github.com/CogStack/MedCATtrainer)
2. Annotate entities, dates, and relationships using the MedCAT Trainer interface
3. Download the JSON export and run through `create_dataset.ipynb`
4. This creates `medcat_trainer_dataset.csv` with all required columns including ground truth relationships
5. Use this dataset for training and evaluation of all extractors

### Inference Pipeline
1. Use the [MedCAT library](https://github.com/CogStack/cogstack-nlp) to extract entities
2. Use [AnonCAT](https://github.com/antsh3k/deidentify) to extract absolute dates
3. Use the built-in relative date extractor to find relative date phrases
4. No manual labeling required - suitable for processing new documents at scale

**Note**: Pipeline 1 is preferred for training and development as it provides high-quality labeled data through MedCAT's annotation interface. Pipeline 2 is designed for production inference where manual labeling is not feasible.

## Data Format

The system works with CSV files containing the following columns:

- `doc_id`: Document identifier
- `note_text`: Clinical note text
- `entities_json`: JSON array of entities with positions and values
- `dates_json`: JSON array of absolute dates with positions and values
- `relative_dates_json`: JSON array of relative date phrases (e.g., "last week", "3 days ago") - automatically extracted if missing
- `links_json`: JSON array of correct entity-date relationships for evaluation (training only)

**Optional columns:**
- `patient`: Patient ID
- `document_timestamp`: Document date (YYYY-MM-DD)

A sample file `data/synthetic.csv` is provided as an example of the required format.

All data files should be stored within the `data` folder.

## Usage

### Get Started

1. Clone the repo
2. Put your CSV file inside the data folder
3. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Extractor Customisation & Configuration

#### Llama-3.2-3B

For Llama-3.2 support, ensure you have the model downloaded in the root folder and that you have access to the model via HuggingFace. The model can be downloaded from: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct.