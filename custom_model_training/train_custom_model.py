import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Adjust relative paths for imports since train.py is in model_training/
# Get the parent directory (project root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import from our modules using adjusted paths
# Files within the same directory (model_training)
from DiagnosisDateRelationModel import DiagnosisDateRelationModel 
from custom_model_training.Vocabulary import Vocabulary
from custom_model_training.ClinicalNoteDataset import ClinicalNoteDataset
import custom_model_training.training_config_custom as training_config_custom 

# Files from other top-level directories
from custom_model_training.training_utils_custom import prepare_custom_training_data
from custom_model_training.training_utils_custom import train_model, plot_training_curves
from config import DEVICE, MODEL_PATH, VOCAB_PATH, TRAINING_SET_RATIO, DATA_SPLIT_RANDOM_SEED
from utils.inference_utils import load_and_prepare_data

def train():
    print(f"Using device: {DEVICE}")
    
    # Ensure the training directory exists for outputs
    model_full_path = os.path.join(project_root, MODEL_PATH)
    vocab_full_path = os.path.join(project_root, VOCAB_PATH)
    output_dir = os.path.dirname(model_full_path) 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check if vocabulary file exists - it should be built before training
    if not os.path.exists(vocab_full_path):
        print(f"Error: Vocabulary file not found at {vocab_full_path}")
        print("Please run build_vocab.py first to create the vocabulary.")
        sys.exit(1)
    
    # Load the pre-built vocabulary
    print(f"Loading vocabulary from: {vocab_full_path}")
    try:
        # Handle different PyTorch versions
        try:
            # Try the newer PyTorch approach
            vocab_instance = torch.load(vocab_full_path, weights_only=False)
        except TypeError:
            # For older PyTorch versions that don't have weights_only
            vocab_instance = torch.load(vocab_full_path)
            
        print(f"Vocabulary loaded successfully. Size: {vocab_instance.n_words} words")
    except Exception as e:
        print(f"Error loading vocabulary: {e}")
        sys.exit(1)
    
    # Step 1: Load training data from the path specified in training_config
    training_data_path = os.path.join(project_root, training_config_custom.TRAINING_DATA_PATH)
    
    if not os.path.exists(training_data_path):
        print(f"Error: Training data file not found at {training_data_path}")
        sys.exit(1)
    
    print(f"Loading training data from: {training_data_path}")
    
    # Use the canonical data preparation pipeline with the 'train' data split
    # This will directly load the CSV file and parse all entities and relationships
    print("Loading and preparing data...")
    
    # First, load the training split using load_and_prepare_data
    # This ensures we're using the same split logic as in inference
    import config as main_config
    if main_config.ENTITY_MODE == 'disorder_only':
        prepared_train_data, relationship_gold = load_and_prepare_data(
            training_data_path, None, main_config, data_split_mode='train'
        )
    else:
        prepared_train_data, entity_gold, relationship_gold, _ = load_and_prepare_data(
            training_data_path, None, main_config, data_split_mode='train'
        )
    
    # Now prepare the training features using the training data
    features, labels, _ = prepare_custom_training_data(
        training_data_path, training_config_custom.MAX_DISTANCE, None, data_split_mode='train'
    )
    print(f"Loaded {len(features)} examples")
    
    # Check class balance
    if len(labels) > 0:
        positive = sum(labels)
        negative = len(labels) - positive
        print(f"Class distribution: {positive} positive examples ({positive/len(labels)*100:.1f}%), {negative} negative examples ({negative/len(labels)*100:.1f}%)")
        
        # Check if we have any positive examples
        if positive == 0:
            print("ERROR: No positive examples found. The model will not learn anything useful.")
            print("Please check your data format and ensure that relationship_gold contains valid relationships.")
            sys.exit(1)
    else:
        print("ERROR: No examples found in the dataset!")
        print("Please check your data format and ensure that extracted_disorders and formatted_dates are correctly parsed.")
        sys.exit(1)
    
    # Step 3: Create train / val / test datasets 
    train_features, val_features, train_labels, val_labels = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    print(f"Train: {len(train_features)}, Validation: {len(val_features)}")
    
    # Create datasets using training_config settings
    train_dataset = ClinicalNoteDataset(
        train_features, train_labels, vocab_instance, 
        training_config_custom.MAX_CONTEXT_LEN, training_config_custom.MAX_DISTANCE
    ) 
    val_dataset = ClinicalNoteDataset(
        val_features, val_labels, vocab_instance, 
        training_config_custom.MAX_CONTEXT_LEN, training_config_custom.MAX_DISTANCE
    )
    
    # Create data loaders using training_config batch size
    train_loader = DataLoader(train_dataset, batch_size=training_config_custom.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_config_custom.BATCH_SIZE)
    
    # Step 4: Initialize and train model using training_config settings
    model = DiagnosisDateRelationModel(
        vocab_size=vocab_instance.n_words, 
        embedding_dim=training_config_custom.EMBEDDING_DIM,
        hidden_dim=training_config_custom.HIDDEN_DIM
    ).to(DEVICE)
    
    # Loss function and optimizer using training_config learning rate
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=training_config_custom.LEARNING_RATE)
    
    # Train model using training_config epochs
    print("Training model...")
    train_losses, val_losses, val_accs = train_model(
        model, train_loader, val_loader, optimizer, criterion, 
        training_config_custom.NUM_EPOCHS, DEVICE, model_full_path
    )
    
    # Plot training curves
    plot_save_path = os.path.join(output_dir, 'training_curves.png')
    plot_training_curves(train_losses, val_losses, val_accs, plot_save_path)
    
    print("Done!")

if __name__ == "__main__":
    train() 