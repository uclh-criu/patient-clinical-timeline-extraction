import torch

# --- Hardware Configuration --- #
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Device to use for training 

# --- Custom Model Training Configuration --- #
TRAINING_DATA_PATH = 'data/synthetic.csv'  # Path to training data
MAX_DISTANCE = 500  # Maximum distance between diagnosis and date to consider
MAX_CONTEXT_LEN = 150  # Maximum context length for text window around entities
VOCAB_SIZE = 10000  # Maximum vocabulary size
EMBEDDING_DIM = 100  # Dimension of word embeddings
HIDDEN_DIM = 128  # Dimension of hidden layer
OUTPUT_DIM = 1  # Binary classification
BATCH_SIZE = 32  # Batch size for training
LEARNING_RATE = 0.001  # Learning rate
NUM_EPOCHS = 10  # Number of training epochs (alias for EPOCHS)
DROPOUT = 0.3  # Dropout rate for regularization
USE_DISTANCE_FEATURE = True  # Whether to use distance between diagnosis and date as a feature
USE_POSITION_FEATURE = True  # Whether to use relative position (diagnosis before date) as a feature