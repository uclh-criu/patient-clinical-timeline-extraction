import torch

# --- Hardware Configuration --- #
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Device to use for training 

# --- Custom Model Training Configuration --- #
# Note: You can provide a single value for each parameter, or a list of values to trigger grid search
# Example: LEARNING_RATE = [0.001, 0.0005] will train models with both learning rates

ENTITY_MODE = 'multi_entity'  # Entity mode: 'diagnosis_only' or 'multi_entity'
TRAINING_DATA_PATH = 'data/synthetic_results_final.csv'  # Path to training data
MAX_DISTANCE = 200 # Maximum distance between diagnosis and date to consider
MAX_CONTEXT_LEN = [32, 64, 128, 256]  # Maximum context length for text window around entities
VOCAB_SIZE = 8000  # Maximum vocabulary size
EMBEDDING_DIM = 128  # Dimension of word embeddings
ENTITY_CATEGORY_EMBEDDING_DIM = 8  # Dimension for entity category embeddings
HIDDEN_DIM = 256  # Dimension of hidden layer
OUTPUT_DIM = 1  # Output dimension
BATCH_SIZE = 8  # Batch size for training
LEARNING_RATE = [0.001, 0.0005, 0.0001]  # Learning rate
NUM_EPOCHS = 8  # Number of training epochs
DROPOUT = 0.3  # Dropout rate for regularization
USE_DISTANCE_FEATURE = True  # Whether to use distance between diagnosis and date as a feature
USE_POSITION_FEATURE = True  # Whether to use relative position (diagnosis before date) as a feature

# --- Grid Search Configuration --- #
# Set ENABLE_GRID_SEARCH to False to use only the first value in each list
# When True, all combinations of hyperparameters will be tested
ENABLE_GRID_SEARCH = True