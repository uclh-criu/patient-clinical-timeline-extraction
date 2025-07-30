import torch

# --- Hardware Configuration --- #
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Device to use for training 

# --- BERT Model Configuration --- #
# Note: You can provide a single value for each parameter, or a list of values to trigger grid search
# Example: BERT_LEARNING_RATE = [2e-5, 3e-5] will train models with both learning rates

# Entity Mode
ENTITY_MODE = 'diagnosis_only'  # Entity mode: 'diagnosis_only' or 'multi_entity'

# BERT Training Data
BERT_TRAINING_DATA_PATH = 'data/synthetic.csv'  # Data to train BERT model on
BERT_PRETRAINED_MODEL = 'dmis-lab/biobert-base-cased-v1.1'  # Pre-trained model to use
BERT_MAX_SEQ_LENGTH = 512  # Maximum sequence length for BERT
BERT_BATCH_SIZE = 8  # Batch size for BERT training
BERT_LEARNING_RATE = 2e-5  # Learning rate for BERT fine-tuning
BERT_NUM_TRAIN_EPOCHS = 3  # Number of epochs for BERT fine-tuning
BERT_DROPOUT = 0.1  # Dropout rate for BERT fine-tuning

# --- Grid Search Configuration --- #
# Set ENABLE_GRID_SEARCH to False to use only the first value in each list
# When True, all combinations of hyperparameters will be tested
ENABLE_GRID_SEARCH = True