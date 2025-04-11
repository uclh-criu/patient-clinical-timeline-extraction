import torch

# --- Execution Settings --- #
# Mode to run when main.py is executed.
# Valid options: 'single', 'evaluate', 'compare'
RUN_MODE = 'evaluate'

# Extraction method to use (relevant for 'single' and 'evaluate' modes).
# Valid options: 'custom', 'naive', 'relcat', 'llm'
EXTRACTION_METHOD = 'relcat'

# --- Processing Parameters (for prediction) --- #
# Max char distance used by custom_extractor when finding candidate pairs
PREDICTION_MAX_DISTANCE = 500 
# Max sequence length used by custom_extractor when creating input tensors
PREDICTION_MAX_CONTEXT_LEN = 512

# --- Naive (proximity) Extractor Parameters --- #
PROXIMITY_MAX_DISTANCE = 200  

# --- LLM (OpenAI) Extractor Parameters --- #
OPENAI_MODEL = 'gpt-4o'

# --- File Paths --- #
DATASET_PATH = 'data/synthetic_data.json'
MODEL_PATH = 'model_training/best_model.pt'  
VOCAB_PATH = 'model_training/vocab.pt'      

# --- RelCAT Extractor Parameters --- #
# Ensure these paths are correct for your setup
MEDCAT_MODEL_PATH = 'extractors/relcat/relcat_model.pt'  # Example Path
MEDCAT_CDB_PATH = 'extractors/relcat/cdb.dat'        # Example Path

# --- Hardware Settings --- #
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

