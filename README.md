# Patient Clinical Timeline Extraction

This repository provides a set of tools for constructing patient clinical timelines from free text in Electronic Healthcare Records (EHRs).

You can choose between four different methods for relation extraction between clinical entities and dates, which are used to construct the timelines. 

These are:
1. **Naive model** which finds the nearest date within a maximum distance of characters for each clinical entity
2. **Finetuned BERT model** for binary relation classification between dates and clinical entities with span pooling to focus on entity representations 
3. **LLM** approach with few-shot prompting, either for binary relation classification or multi-relation extraction
4. **[RelCAT](https://arxiv.org/abs/2501.16077)**, a module of the [MedCAT](https://arxiv.org/abs/2010.01165) framework, specifically designed for relation extraction between entities in clinical text

The repository provides utility functions for these extraction methods, as well as customisable notebooks for training, evaluation and inference.

## Usage

### Get Started

1. Clone the repo
3. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Data Format

The system works with CSV files containing the following columns:

- `patient_id`: Patient identifier
- `doc_id`: Document identifier
- `document_timestamp`: Document date (YYYY-MM-DD)
- `note_text`: Clinical note text

A sample file `data/data.csv` is provided as an example of the required format.

### Data Pipelines

There are three core pipelines that can be run, one for training/evaluation, one for inference, and one for post-processing to construct the patient timelines.

#### 1. Training Pipeline via MedCAT Trainer

This pipeline uses the [MedCAT Trainer](https://github.com/CogStack/MedCATtrainer) tool to label entities, dates and relations in the clinical text, and uses this data to train, finetune and evaluate the various relation extraction methods.

Steps:
1. Upload the raw CSV file to MedCAT Trainer. Note the column names will need to be changed, see the [MedCAT Trainer Documentation](https://medcattrainer.readthedocs.io/en/latest/) for further details
2. Annotate entities, dates, and relationships using the MedCAT Trainer interface
3. Download the JSON export and run this through the `create_training_dataset.ipynb` notebook. An example json file is provided in `data/MedCAT_Export.json` if you wish to try the training process without having to do the manual labelling
4. This creates `training_dataset.csv` with all required columns including ground truth relationships
5. Use this dataset for training and evaluating any extraction methods in the `notebooks_training` folder

#### 2. Inference Pipeline

This pipeline is designed for processing new documents at scale and doesn't require manual labelling. It uses the [MedCAT library](https://github.com/CogStack/cogstack-nlp/blob/main/medcat-v2/README.md) to extract entities, various utility functions to extract dates and the models from the training pipeline to do the relation extraction and make predictions.

Steps:
1. Run a new raw CSV file through the `create_inference_dataset.ipynb` notebook (note: this should be in the exact format described above in the Data Format section)
2. This creates `inference_dataset.csv` with all required columns
3. Use any of the extractors in the `notebooks_inference` folder to do inference on the data and generate predictions
4. This will create files named `{extractor_type}_predictions.json` in the `outputs` folder

#### 3. Post-Processing Pipeline

This pipeline aggregates notes by individual patient and uses the predicted relations to construct the final patient timelines.

Steps:
1. Run a `{extractor_type}_predictions.json` file created by any of the inference pipeline extractors through the `create_patient_timelines.ipynb` notebook
2. This will create:
   - Interactive timeline visualizations (saved as HTML files)
   - JSON timeline summaries for each patient
   - Both can be found in the `outputs` folder