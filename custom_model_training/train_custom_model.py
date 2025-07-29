import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import itertools
import copy
import time

# Add parent directory to path to allow importing from models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEVICE, MODEL_PATH, VOCAB_PATH, TRAINING_SET_RATIO, DATA_SPLIT_RANDOM_SEED
from utils.inference_eval_utils import load_and_prepare_data
import config as main_config
import custom_model_training.training_config_custom as training_config_custom
from custom_model_training.DiagnosisDateRelationModel import DiagnosisDateRelationModel
from custom_model_training.ClinicalNoteDataset import ClinicalNoteDataset
from custom_model_training.training_utils_custom import prepare_custom_training_data, train_model, plot_training_curves, log_training_run

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def generate_hyperparameter_grid():
    """
    Generate a grid of hyperparameters for grid search.
    
    Returns:
        list: List of dictionaries, each containing a unique combination of hyperparameters
    """
    # If grid search is disabled, return a single configuration with the first value of each list
    if not getattr(training_config_custom, 'ENABLE_GRID_SEARCH', True):
        config = {}
        for key, value in vars(training_config_custom).items():
            if key.startswith('__'):
                continue
            if isinstance(value, list):
                config[key] = value[0]  # Take the first value
            else:
                config[key] = value
        return [config]
    
    # Identify hyperparameters that are lists
    param_grid = {}
    for key, value in vars(training_config_custom).items():
        if key.startswith('__'):
            continue
        if isinstance(value, list) and len(value) > 0:
            param_grid[key] = value
    
    # If no lists are found, return the original config
    if not param_grid:
        return [vars(training_config_custom)]
    
    # Generate all combinations of hyperparameters
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    # Create a list of dictionaries with all combinations
    grid = []
    for combo in combinations:
        config = {}
        # Copy all parameters from training_config_custom
        for key, value in vars(training_config_custom).items():
            if key.startswith('__'):
                continue
            config[key] = value
        
        # Override with the current combination
        for i, key in enumerate(keys):
            config[key] = combo[i]
        
        grid.append(config)
    
    return grid

def train_with_config(config):
    """
    Train a model with the given hyperparameter configuration.
    
    Args:
        config (dict): Dictionary of hyperparameters
        
    Returns:
        tuple: (best_val_acc, model_path, metrics)
    """
    print(f"\n{'='*80}")
    print(f"Training with configuration:")
    # Print only the hyperparameters that are being tuned (those that are lists in the original config)
    for key, value in config.items():
        if isinstance(getattr(training_config_custom, key, None), list):
            print(f"  {key}: {value}")
    print(f"{'='*80}\n")
    
    # Get entity mode from config
    entity_mode = config['ENTITY_MODE']
    
    # Get dataset name from training data path
    dataset_path = config['TRAINING_DATA_PATH']
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
        return 0, None, None
    
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
        return 0, None, None
    
    # Step 1: Load training data from the path specified in training_config
    training_data_path = os.path.join(project_root, config['TRAINING_DATA_PATH'])
    
    if not os.path.exists(training_data_path):
        print(f"Error: Training data file not found at {training_data_path}")
        return 0, None, None
    
    print(f"Loading training data from: {training_data_path}")
    
    # Check if we're in multi-entity mode
    print(f"Using entity mode: {entity_mode}")
    
    # Use the canonical data preparation pipeline with the 'train' data split
    print("Loading and preparing data...")
    
    # Set the entity mode in the main config to match the current config
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
        training_data_path, config['MAX_DISTANCE'], None, data_split_mode='all'
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
            return 0, None, None
    else:
        print("ERROR: No examples found in the dataset!")
        print("Please check your data format and ensure that extracted_disorders and formatted_dates are correctly parsed.")
        return 0, None, None
    
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
    
    # Create datasets using config settings
    train_dataset = ClinicalNoteDataset(
        train_features, train_labels, vocab_instance, 
        config['MAX_CONTEXT_LEN'], config['MAX_DISTANCE'],
        entity_category_map
    ) 
    val_dataset = ClinicalNoteDataset(
        val_features, val_labels, vocab_instance, 
        config['MAX_CONTEXT_LEN'], config['MAX_DISTANCE'],
        entity_category_map
    )
    
    # Create data loaders using config batch size
    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'])
    
    # Step 4: Initialize and train model using config settings
    model = DiagnosisDateRelationModel(
        vocab_size=vocab_instance.n_words, 
        embedding_dim=config['EMBEDDING_DIM'],
        hidden_dim=config['HIDDEN_DIM']
    ).to(DEVICE)
    
    # Loss function and optimizer using config learning rate
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])
    
    # Train model using config epochs
    print("Training model...")
    metrics = train_model(
        model, train_loader, val_loader, optimizer, criterion, 
        config['NUM_EPOCHS'], DEVICE, model_full_path
    )
    
    # Plot training curves
    plot_save_path = os.path.join(output_dir, f"{os.path.splitext(model_name)[0]}_training_curves")
    plot_training_curves(metrics, plot_save_path)
    
    # Collect hyperparameters for logging
    hyperparams = {
        'ENTITY_MODE': entity_mode,
        'MAX_DISTANCE': config['MAX_DISTANCE'],
        'MAX_CONTEXT_LEN': config['MAX_CONTEXT_LEN'],
        'EMBEDDING_DIM': config['EMBEDDING_DIM'],
        'HIDDEN_DIM': config['HIDDEN_DIM'],
        'BATCH_SIZE': config['BATCH_SIZE'],
        'LEARNING_RATE': config['LEARNING_RATE'],
        'NUM_EPOCHS': config['NUM_EPOCHS'],
        'DROPOUT': config['DROPOUT'],
        'USE_DISTANCE_FEATURE': config['USE_DISTANCE_FEATURE'],
        'USE_POSITION_FEATURE': config['USE_POSITION_FEATURE'],
        'ENTITY_CATEGORY_EMBEDDING_DIM': config['ENTITY_CATEGORY_EMBEDDING_DIM'],
        'train_examples': len(train_features),
        'val_examples': len(val_features),
        'positive_examples_pct': positive/len(labels)*100
    }
    
    # Log the training run
    log_training_run(model_full_path, hyperparams, metrics, dataset_name, entity_mode)
    
    return metrics['best_val_acc'], model_full_path, metrics

def train():
    print(f"Using device: {DEVICE}")
    
    # Generate hyperparameter grid
    print("Generating hyperparameter grid for training...")
    hyperparameter_grid = generate_hyperparameter_grid()
    print(f"Generated {len(hyperparameter_grid)} hyperparameter combinations for grid search.")
    
    # Track the best model across all runs
    best_val_acc = 0
    best_model_path = None
    best_config = None
    best_metrics = None
    
    # Train with each hyperparameter combination
    start_time = time.time()
    for i, config in enumerate(hyperparameter_grid):
        print(f"\nTraining run {i+1}/{len(hyperparameter_grid)}")
        val_acc, model_path, metrics = train_with_config(config)
        
        # Update best model if this run is better
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = model_path
            best_config = config
            best_metrics = metrics
            print(f"\n*** New best model found with validation accuracy: {best_val_acc:.2f}% ***")
    
    # Print summary of grid search
    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Grid search completed in {elapsed_time:.2f} seconds.")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {best_model_path}")
    print(f"Best hyperparameters:")
    for key, value in best_config.items():
        if isinstance(getattr(training_config_custom, key, None), list):
            print(f"  {key}: {value}")
    print(f"{'='*80}")
    
    print("Done!")

if __name__ == "__main__":
    train()