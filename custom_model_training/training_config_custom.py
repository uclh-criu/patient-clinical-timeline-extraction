import torch

# --- Hardware Configuration --- #
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Device to use for training 

# --- Custom Model Training Configuration --- #
ENTITY_MODE = 'diagnosis_only'  # Entity mode: 'diagnosis_only' or 'multi_entity'
TRAINING_DATA_PATH = 'data/synthetic.csv'  # Path to training data
BERT_MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'  # Name of the BERT model to use for tokenization
VOCAB_PATH = 'custom_model_training/vocabs/clinicalbert_vocab.pt'  # Path to vocabulary file for this dataset
PRETRAINED_EMBEDDINGS_PATH = 'custom_model_training/vocabs/clinicalbert_embeddings.pt'  # Path to pre-trained embeddings
USE_PRETRAINED_EMBEDDINGS = True  # Whether to use pre-trained embeddings
USE_WEIGHTED_LOSS = True # Whether to use weighted loss to address class imbalance
POS_WEIGHT = None  # Positive class weight for weighted BCE loss. None = auto-calculate, or set a specific value (e.g., 13.5). Formula: POS_WEIGHT = num_negative_samples / num_positive_samples

# --- Grid Search & Hyperparameters --- #
# Note: You can provide a single value for each parameter, or a list of values to trigger grid search
# Example: LEARNING_RATE = [0.001, 0.0005] will train models with both learning rates

ENABLE_GRID_SEARCH = True # Set ENABLE_GRID_SEARCH to False to use only the first value in each list
MAX_DISTANCE = 200 # Maximum distance between diagnosis and date to consider
MAX_CONTEXT_LEN = [128, 256, 512]  # Maximum context length for text window around entities
#VOCAB_SIZE = 8000  # Maximum vocabulary size
EMBEDDING_DIM = [64, 128]  # Dimension of word embeddings
ENTITY_CATEGORY_EMBEDDING_DIM = 8  # Dimension for entity category embeddings
HIDDEN_DIM = [128, 256]  # Dimension of hidden layer
OUTPUT_DIM = 1  # Output dimension
BATCH_SIZE = [32, 64, 128]  # Batch size for training
LEARNING_RATE = [0.001, 0.0001, 0.00005]  # Learning rate
NUM_EPOCHS = 50  # Number of training epochs
DROPOUT = [0.4, 0.6, 0.8]  # Dropout rate for regularization
USE_DISTANCE_FEATURE = True  # Whether to use distance between diagnosis and date as a feature
USE_POSITION_FEATURE = True  # Whether to use relative position (diagnosis before date) as a feature