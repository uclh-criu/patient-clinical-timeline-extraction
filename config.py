import torch

# --- Execution Settings --- #
# Mode to run when main.py is executed.
# Valid options: 'single', 'evaluate', 'compare'
RUN_MODE = 'evaluate'

# Extraction method to use (relevant for 'single' and 'evaluate' modes).
# Valid options: 'custom', 'naive', 'relcat', 'llm', 'llama'
EXTRACTION_METHOD = 'naive'

# Methods to include when running in 'compare' mode
COMPARISON_METHODS = ['naive']

# --- Data Source Settings --- #
# Specifies the source of the data to be used.
# Valid options: 'synthetic', 'imaging', 'notes', 'letters', 'sample'
DATA_SOURCE = 'sample'  # Example: 'imaging', 'notes', 'letters', 'synthetic', or 'sample'

# --- Debug Settings --- #
DEBUG_MODE = False  # Set to True for verbose logging during API calls and data processing

# --- Real Data File Paths (used if DATA_SOURCE is not 'synthetic') --- #
IMAGING_DATA_PATH = 'data/processed_notes_with_dates_and_disorders_imaging.csv'
NOTES_DATA_PATH = 'data/processed_notes_with_dates_and_disorders_notes.csv'
LETTERS_DATA_PATH = 'data/processed_notes_with_dates_and_disorders_letters.csv'
SAMPLE_DATA_PATH = 'data/sample.csv'

# --- Real Data Column Names (assuming consistent across imaging, notes, letters files) --- #
REAL_DATA_TEXT_COLUMN = 'note'              # Column containing clinical notes
REAL_DATA_GOLD_COLUMN = 'gold_standard'     # Column containing gold standard labels (if available)
REAL_DATA_DIAGNOSES_COLUMN = 'extracted_disorders'  # Column containing pre-extracted disorder entities
REAL_DATA_DATES_COLUMN = 'formatted_dates'          # Column containing pre-extracted date entities
REAL_DATA_TIMESTAMP_COLUMN = 'document_timestamp'   # Column containing document creation timestamp

# --- Processing Parameters (for prediction) --- #
# Max char distance used by custom_extractor when finding candidate pairs
PREDICTION_MAX_DISTANCE = 500 
# Max sequence length used by custom_extractor when creating input tensors
PREDICTION_MAX_CONTEXT_LEN = 512

# --- Naive (proximity) Extractor Parameters --- #
PROXIMITY_MAX_DISTANCE = 200  

# --- LLM (OpenAI) Extractor Parameters --- #
OPENAI_MODEL = 'gpt-4o'

# --- Llama Extractor Parameters --- #
LLAMA_MODEL_PATH = './Llama-3.2-3B-Instruct'

# --- Relative Date Extraction LLM Settings --- #
ENABLE_RELATIVE_DATE_EXTRACTION = True      # Whether to extract relative dates
RELATIVE_DATE_LLM_MODEL = 'openai'          # Which LLM to use: 'openai' or 'llama'
RELATIVE_DATE_OPENAI_MODEL = 'gpt-3.5-turbo' # OpenAI model for relative date extraction (cheaper than gpt-4o)
RELATIVE_DATE_CONTEXT_WINDOW = 1000         # Maximum context window to send to LLM for date extraction

# --- File Paths (Synthetic Data) --- #
SYNTHETIC_DATASET_PATH = 'data/synthetic_data.json' # Path to synthetic data JSON file
MODEL_PATH = 'model_training/best_model.pt'  
VOCAB_PATH = 'model_training/vocab.pt'      

# --- RelCAT Extractor Parameters --- #
# Ensure these paths are correct for your setup
MEDCAT_MODEL_PATH = 'extractors/relcat/relcat_model.pt'  # Example Path
MEDCAT_CDB_PATH = 'extractors/relcat/cdb.dat'        # Example Path

# --- Hardware Settings --- #
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

