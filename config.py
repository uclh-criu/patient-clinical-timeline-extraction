import torch

# --- Data Source Settings --- #
# Specifies the source of the data to be used for inference, evaluation, and timeline generation.
# Valid options: 'imaging', 'synthetic', 'synthetic_multi', 'nph'
DATA_SOURCE = 'synthetic_multi'

# --- Entity Mode Setting --- #
# Controls which entity types to use for the relation extraction task.
# Valid options: 'diagnosis_only', 'multi_entity'
ENTITY_MODE = "multi_entity"

# --- Execution Settings --- #
# Controls the extraction method to use for the relation extraction task.
# Valid options: 'custom', 'naive', 'relcat', 'openai', 'llama', 'bert'
EXTRACTION_METHOD = 'bert'
TRAINING_SET_RATIO = 0.8 # The ratio of data to be used for the training set. The rest will be used for the test/inference set.
DATA_SPLIT_RANDOM_SEED = 42 # A fixed random seed to ensure the train/test split is always the same.
INFERENCE_SAMPLES = None # Limits the number of samples used from the test set. Set to None to use all test samples.

# --- Custom Extractor Parameters --- #
MODEL_PATH = 'custom_model_training/models/nph_custom.pt'  
VOCAB_PATH = 'custom_model_training/vocabs/nph_vocab.pt'
PREDICTION_MAX_DISTANCE = 500 # Max char distance used by custom_extractor when finding candidate pairs
PREDICTION_MAX_CONTEXT_LEN = 512 # Max sequence length used by custom_extractor when creating input tensors
CUSTOM_CONFIDENCE_THRESHOLD = 0.5 # Confidence threshold for custom model predictions
USE_PRETRAINED_EMBEDDINGS = False # Whether to use pre-trained embeddings from a BERT model
BERT_MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT' # The BERT model to use for embeddings
PRETRAINED_EMBEDDINGS_PATH = 'custom_model_training/vocabs/clinicalbert_embeddings.pt' # Path to pre-trained embeddings

# --- Naive (proximity) Extractor Parameters --- #
PROXIMITY_MAX_DISTANCE = 200

# --- BERT Extractor Parameters --- #
BERT_MODEL_PATH = './bert_model_training/bert_model'
BERT_CONFIDENCE_THRESHOLD = 0.185  # Confidence threshold for BERT predictions

# --- Llama Extractor Parameters --- #
LLAMA_MODEL_PATH = './Llama-3.2-3B-Instruct'

# --- OpenAI Extractor Parameters --- #
OPENAI_MODEL = 'gpt-4o'

# --- RelCAT Extractor Parameters --- #
MEDCAT_MODEL_PATH = 'extractors/relcat/relcat_model.pt'  # Example Path
MEDCAT_CDB_PATH = 'extractors/relcat/cdb.dat'        # Example Path

# --- Relative Date Extraction LLM Settings --- #
ENABLE_RELATIVE_DATE_EXTRACTION = False      # Whether to extract relative dates
RELATIVE_DATE_LLM_MODEL = 'openai'           # Which LLM to use: 'openai' or 'llama'
RELATIVE_DATE_OPENAI_MODEL = 'gpt-3.5-turbo' # OpenAI model for relative date extraction (cheaper than gpt-4o)
RELATIVE_DATE_CONTEXT_WINDOW = 2500         # Maximum context window to send to LLM for date extraction

# --- Data File Paths --- #
SYNTHETIC_DATA_PATH = 'data/synthetic.csv'
IMAGING_DATA_PATH = 'data/imaging.csv'
SYNTHETIC_MULTI_DATA_PATH = 'data/synthetic_multi.csv'
NPH_DATA_PATH = 'data/nph.csv'

# --- Data Column Names --- #
PATIENT_ID_COLUMN = 'patient'     # Column containing patient identifiers
NOTE_ID_COLUMN = 'note_id'        # Column containing unique note identifiers
TEXT_COLUMN = 'note'              # Column containing clinical notes
TIMESTAMP_COLUMN = 'document_timestamp'   # Column containing document creation timestamp
DATES_COLUMN = 'formatted_dates'          # Column containing pre-extracted date entities
ENTITY_GOLD_COLUMN = 'entity_gold'                  # Gold standard for entities (NER task)
RELATIONSHIP_GOLD_COLUMN = 'relationship_gold'      # Gold standard for relationships (RE task)
DIAGNOSES_COLUMN = 'extracted_disorders'            # Used in diagnosis_only mode
SNOMED_COLUMN = 'extracted_snomed_entities'         # Used in multi_entity mode
UMLS_COLUMN = 'extracted_umls_entities'             # Used in multi_entity mode

# Output directory for timeline visualizations
TIMELINE_OUTPUT_DIR = 'experiment_outputs/timelines'

# --- Test Data Paths --- #
# These paths point to the datasets used for automated testing
TEST_DIAGNOSIS_ONLY_DATA_PATH = 'data/synthetic.csv'
TEST_MULTI_ENTITY_DATA_PATH = 'data/synthetic_multi.csv'

# --- Debug Settings --- #
DEBUG_MODE = False  # Set to True for verbose logging during API calls and data processing
MODEL_DEBUG_MODE = True  # Set to True to enable diagnostic prints in the model during training and inference

# --- Entity Category Mappings --- #
# Maps detailed entity categories to simplified categories for evaluation and display
CATEGORY_MAPPINGS = {
    # Diagnosis-related categories
    'disorder': 'diagnosis',
    'neoplastic process': 'diagnosis',
    'acquired abnormality': 'diagnosis',
    'anatomical abnormality': 'diagnosis',
    'mental or behavioral dysfunction': 'diagnosis',
    'pathologic function': 'diagnosis',
    
    # Symptom-related categories
    'sign or symptom': 'symptom',
    'finding': 'symptom',
    
    # Procedure-related categories
    'procedure': 'procedure',
    'diagnostic procedure': 'procedure',
    'therapeutic or preventive procedure': 'procedure',
    
    # Medication-related categories
    'clinical drug': 'medication',
    'pharmacologic substance': 'medication',
    'product': 'medication',
    'organism': 'medication'  # Assuming organisms are used in medications
}

# --- Hardware Settings --- #
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

