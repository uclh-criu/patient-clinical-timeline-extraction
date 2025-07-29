import torch

# --- Hardware Configuration --- #
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Device to use for training 

# --- BERT Model Configuration --- #
# BERT Training Data
BERT_TRAINING_DATA_PATH = 'data/synthetic_updated.csv'  # Data to train BERT model on
BERT_PRETRAINED_MODEL = 'dmis-lab/biobert-base-cased-v1.1'  # Pre-trained model to use
BERT_MAX_SEQ_LENGTH = 512  # Maximum sequence length for BERT
BERT_BATCH_SIZE = 8  # Batch size for BERT training
BERT_LEARNING_RATE = 2e-5  # Learning rate for BERT fine-tuning
BERT_NUM_TRAIN_EPOCHS = 3  # Number of epochs for BERT fine-tuning