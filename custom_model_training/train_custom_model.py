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
from utils.inference_eval_utils import load_and_prepare_data
import config as main_config

def train():
    print(f"Using device: {DEVICE}")
    
    # Get entity mode from training config
    entity_mode = training_config_custom.ENTITY_MODE
    
    # Get dataset name from training data path
    dataset_path = training_config_custom.TRAINING_DATA_PATH
    dataset_name = os.path.basename(dataset_path)
    
    # Create model name based on dataset and entity mode
    model_name = f"custom_{os.path.splitext(dataset_name)[0]}_{entity_mode}.pt"
    
    # Ensure the training directory exists for outputs
    output_dir = os.path.join(project_root, os.path.dirname(MODEL_PATH))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set the full path for the model
    model_full_path = os.path.join(project_root, os.path.dirname(MODEL_PATH), model_name)
    vocab_full_path = os.path.join(project_root, VOCAB_PATH)
    
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
    
    # Check if we're in multi-entity mode
    print(f"Using entity mode: {entity_mode}")
    
    # Use the canonical data preparation pipeline with the 'train' data split
    # This will directly load the CSV file and parse all entities and relationships
    print("Loading and preparing data...")
    
    # Set the entity mode in the main config to match training_config_custom
    main_config.ENTITY_MODE = entity_mode
    
    try:
        if entity_mode == 'diagnosis_only':
            prepared_train_data, relationship_gold = load_and_prepare_data(
                training_data_path, None, main_config, data_split_mode='train'
            )
        else:
            prepared_train_data, entity_gold, relationship_gold, _ = load_and_prepare_data(
                training_data_path, None, main_config, data_split_mode='train'
            )
    except ZeroDivisionError:
        print("Error: ZeroDivisionError occurred during data loading. This is likely due to an empty dataset after splitting.")
        print("Trying again with data_split_mode='all' to use all available data...")
        
        if entity_mode == 'diagnosis_only':
            prepared_train_data, relationship_gold = load_and_prepare_data(
                training_data_path, None, main_config, data_split_mode='all'
            )
        else:
            prepared_train_data, entity_gold, relationship_gold, _ = load_and_prepare_data(
                training_data_path, None, main_config, data_split_mode='all'
            )
    
    # Now prepare the training features using the training data
    features, labels, _ = prepare_custom_training_data(
        training_data_path, training_config_custom.MAX_DISTANCE, None, data_split_mode='all'
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
    
    # Create entity category map for multi-entity mode
    entity_category_map = {
        'diagnosis': 0,
        'symptom': 1,
        'procedure': 2,
        'medication': 3
    }
    
    # Create datasets using training_config settings
    train_dataset = ClinicalNoteDataset(
        train_features, train_labels, vocab_instance, 
        training_config_custom.MAX_CONTEXT_LEN, training_config_custom.MAX_DISTANCE,
        entity_category_map
    ) 
    val_dataset = ClinicalNoteDataset(
        val_features, val_labels, vocab_instance, 
        training_config_custom.MAX_CONTEXT_LEN, training_config_custom.MAX_DISTANCE,
        entity_category_map
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
    metrics = train_model(
        model, train_loader, val_loader, optimizer, criterion, 
        training_config_custom.NUM_EPOCHS, DEVICE, model_full_path
    )
    
    # Plot training curves
    plot_save_path = os.path.join(output_dir, f"{os.path.splitext(model_name)[0]}_training_curves")
    plot_training_curves(metrics, plot_save_path)
    
    # Collect hyperparameters for logging
    hyperparams = {
        'ENTITY_MODE': entity_mode,
        'MAX_DISTANCE': training_config_custom.MAX_DISTANCE,
        'MAX_CONTEXT_LEN': training_config_custom.MAX_CONTEXT_LEN,
        'EMBEDDING_DIM': training_config_custom.EMBEDDING_DIM,
        'HIDDEN_DIM': training_config_custom.HIDDEN_DIM,
        'BATCH_SIZE': training_config_custom.BATCH_SIZE,
        'LEARNING_RATE': training_config_custom.LEARNING_RATE,
        'NUM_EPOCHS': training_config_custom.NUM_EPOCHS,
        'DROPOUT': training_config_custom.DROPOUT,
        'USE_DISTANCE_FEATURE': training_config_custom.USE_DISTANCE_FEATURE,
        'USE_POSITION_FEATURE': training_config_custom.USE_POSITION_FEATURE,
        'ENTITY_CATEGORY_EMBEDDING_DIM': training_config_custom.ENTITY_CATEGORY_EMBEDDING_DIM,
        'train_examples': len(train_features),
        'val_examples': len(val_features),
        'positive_examples_pct': positive/len(labels)*100
    }
    
    # Log the training run
    from custom_model_training.training_utils_custom import log_training_run
    log_training_run(model_full_path, hyperparams, metrics, dataset_name, entity_mode)
    
    print("Done!")

if __name__ == "__main__":
    train()