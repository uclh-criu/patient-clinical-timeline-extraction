# Patient Clinical Timeline Extraction

This repository provides a set of tools for constructing patient clinical timelines from free text in Electronic Healthcare Records (EHRs).

You can choose between four different methods for relation extraction between clinical entities and dates, which are used to construct the timelines. 

These are:
1. Naive heuristic model
2. Finetuned BERT
3. Llama-3.2-3B
4. RelCAT

The repository provides utility functions for these extraction methods, and notebooks showing how to run experiments.

## Data

At present the system only works with data in CSV format. 

We assume that you have already done Named Entity Recognition (e.g. using [MedCAT](https://github.com/CogStack/cogstack-nlp)) and absolute date detection (e.g. using [AnonCAT](https://github.com/antsh3k/deidentify)) which are prerequisites for the relation extraction task.

Create a csv file with your data and place it in the data folder in the root of the repository. The file should have the following columns and formats:
- `doc_id`: Document identifier
- `note_text`: Clinical note text
- `entities_json`: JSON array of entities with positions and values
- `dates_json`: JSON array of dates with positions and values
- `links_json`: JSON array of correct entity-date relationships for evaluation

TBC:
- `patient`: Patient ID
- `document_timestamp`: Document date (YYYY-MM-DD)

A sample file `data/synthetic.csv` is provided as an example of the format required.

## Usage

### Get Started

1. Clone the repo
2. Put your csv file inside the data folder
3. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Extractor Customisation & Configuration

#### Llama-3.2-3B

For Llama-3.2 support, ensure you have the model downloaded in the root folder and that you have access to the model via HuggingFace. The model can be downloaded from: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct.