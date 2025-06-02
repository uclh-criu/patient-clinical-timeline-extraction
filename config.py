import torch

# --- Data Source Settings --- #
# Specifies the source of the data to be used.
# Valid options: 'imaging', 'notes', 'letters', 'sample', 'synthetic'
DATA_SOURCE = 'sample'

# --- LLM (OpenAI) Extractor Parameters --- #
OPENAI_MODEL = 'gpt-4o'

# --- Llama Extractor Parameters --- #
LLAMA_MODEL_PATH = './Llama-3.2-3B-Instruct'

# --- Execution Settings --- #
# Mode to run when main.py is executed.
# Valid options: 'evaluate', 'compare'
RUN_MODE = 'evaluate'

# Extraction method to use (relevant for 'evaluate' mode).
# Valid options: 'custom', 'naive', 'relcat', 'openai', 'llama'
EXTRACTION_METHOD = 'naive'

# Methods to include when running in 'compare' mode
COMPARISON_METHODS = ['naive']

# Whether to generate patient timeline visualizations
GENERATE_PATIENT_TIMELINES = True
TIMELINE_OUTPUT_DIR = 'experiment_outputs/timelines'

# --- Relative Date Extraction LLM Settings --- #
ENABLE_RELATIVE_DATE_EXTRACTION = True      # Whether to extract relative dates
RELATIVE_DATE_LLM_MODEL = 'openai'          # Which LLM to use: 'openai' or 'llama'
RELATIVE_DATE_OPENAI_MODEL = 'gpt-3.5-turbo' # OpenAI model for relative date extraction (cheaper than gpt-4o)
RELATIVE_DATE_CONTEXT_WINDOW = 1000         # Maximum context window to send to LLM for date extraction

# --- Data File Paths --- #
SAMPLE_DATA_PATH = 'data/sample.csv'
SYNTHETIC_DATA_PATH = 'data/synthetic.csv'
IMAGING_DATA_PATH = 'data/processed_notes_with_dates_and_disorders_imaging.csv'
NOTES_DATA_PATH = 'data/processed_notes_with_dates_and_disorders_notes.csv'
LETTERS_DATA_PATH = 'data/processed_notes_with_dates_and_disorders_letters.csv'

# --- Data Column Names --- #
REAL_DATA_TEXT_COLUMN = 'note'              # Column containing clinical notes
REAL_DATA_PATIENT_ID_COLUMN = 'patient'     # Column containing patient identifiers
REAL_DATA_TIMESTAMP_COLUMN = 'document_timestamp'   # Column containing document creation timestamp
REAL_DATA_DIAGNOSES_COLUMN = 'extracted_disorders'  # Column containing pre-extracted disorder entities
REAL_DATA_DATES_COLUMN = 'formatted_dates'          # Column containing pre-extracted date entities
REAL_DATA_GOLD_COLUMN = 'gold_standard'     # Column containing gold standard labels (if available)

# --- Debug Settings --- #
DEBUG_MODE = False  # Set to True for verbose logging during API calls and data processing

# --- Processing Parameters (for prediction) --- #
# Max char distance used by custom_extractor when finding candidate pairs
PREDICTION_MAX_DISTANCE = 500 
# Max sequence length used by custom_extractor when creating input tensors
PREDICTION_MAX_CONTEXT_LEN = 512

# --- Naive (proximity) Extractor Parameters --- #
PROXIMITY_MAX_DISTANCE = 200  

# --- File Paths (Custom Model) --- #
MODEL_PATH = 'model_training/best_model.pt'  
VOCAB_PATH = 'model_training/vocab.pt'      

# --- RelCAT Extractor Parameters --- #
# Ensure these paths are correct for your setup
MEDCAT_MODEL_PATH = 'extractors/relcat/relcat_model.pt'  # Example Path
MEDCAT_CDB_PATH = 'extractors/relcat/cdb.dat'        # Example Path

# --- Hardware Settings --- #
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

