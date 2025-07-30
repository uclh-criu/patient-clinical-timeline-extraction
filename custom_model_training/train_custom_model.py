import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import time

# Add parent directory to path to allow importing from models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEVICE, MODEL_PATH, VOCAB_PATH, TRAINING_SET_RATIO, DATA_SPLIT_RANDOM_SEED
from utils.inference_eval_utils import load_and_prepare_data
from utils.training_utils import generate_hyperparameter_grid, log_training_run, plot_training_curves
import config as main_config
import custom_model_training.training_config_custom as training_config_custom
from custom_model_training.DiagnosisDateRelationModel import DiagnosisDateRelationModel
from custom_model_training.ClinicalNoteDataset import ClinicalNoteDataset
from custom_model_training.training_utils_custom import prepare_custom_training_data, train_model

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train_with_config(config, train_dataset, val_dataset, train_features, val_features, labels_info):
    """
    Train a model with the given hyperparameter configuration.
    
    Args:
        config (dict): Dictionary of hyperparameters
        train_dataset: Pre-prepared training dataset
        val_dataset: Pre-prepared validation dataset
        train_features: Training features for statistics
        val_features: Validation features for statistics
        labels_info: Dictionary with label statistics
        
    Returns:
        tuple: (best_val_f1, best_model_state, metrics, model_name, config)
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
    
    # Create data loaders using config batch size
    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'])
    
    # Step 4: Initialize and train model using config settings
    # Check if weighted loss should be used
    use_weighted_loss = config.get('USE_WEIGHTED_LOSS', False)
    if use_weighted_loss:
        # Calculate or use provided positive weight
        pos_weight = config.get('POS_WEIGHT', None)
        if pos_weight is None and 'positive' in labels_info and 'negative' in labels_info:
            # Auto-calculate from data
            pos_weight = labels_info['negative'] / max(1, labels_info['positive'])
            print(f"Auto-calculated positive weight: {pos_weight:.2f}")
        
        if pos_weight is not None:
            # Convert to tensor and use weighted BCE loss
            pos_weight_tensor = torch.tensor([pos_weight]).to(DEVICE)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            print(f"Using weighted BCE loss with positive weight: {pos_weight:.2f}")
        else:
            # Fall back to standard BCE loss if weight cannot be determined
            criterion = nn.BCELoss()
            print("Using standard BCE loss (could not determine positive weight)")
            use_weighted_loss = False
    else:
        # Use standard BCE loss
        criterion = nn.BCELoss()
        print("Using standard BCE loss")
    
    # Initialize model with apply_sigmoid=False when using BCEWithLogitsLoss
    model = DiagnosisDateRelationModel(
        vocab_size=train_dataset.vocab.n_words, 
        embedding_dim=config['EMBEDDING_DIM'],
        hidden_dim=config['HIDDEN_DIM'],
        apply_sigmoid=not use_weighted_loss  # False when using weighted loss (BCEWithLogitsLoss)
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])
    
    # Train model using config epochs
    print("Training model...")
    best_model_state, metrics = train_model(
        model, train_loader, val_loader, optimizer, criterion, 
        config['NUM_EPOCHS'], DEVICE
    )
    
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
        'USE_WEIGHTED_LOSS': config.get('USE_WEIGHTED_LOSS', False),
        'POS_WEIGHT': pos_weight if config.get('USE_WEIGHTED_LOSS', False) else None,
        'train_examples': len(train_features),
        'val_examples': len(val_features),
        'positive_examples_pct': labels_info['positive_pct']
    }
    
    # Log the training run to CSV (but don't save model or plots yet)
    model_full_path = os.path.join(project_root, os.path.dirname(MODEL_PATH), model_name)
    log_training_run(
        model_full_path, 
        hyperparams, 
        metrics, 
        dataset_name, 
        entity_mode,
        os.path.join(project_root, 'custom_model_training'),
        'custom_model_training_log.csv'
    )
    
    # Return metrics, model state, and config for later use
    return metrics['best_val_f1'], best_model_state, metrics, model_name, config

def train():
    print(f"Using device: {DEVICE}")
    
    # Generate hyperparameter grid
    print("Generating hyperparameter grid for training...")
    hyperparameter_grid = generate_hyperparameter_grid(training_config_custom)
    print(f"Generated {len(hyperparameter_grid)} hyperparameter combinations for grid search.")
    
    # Print summary of hyperparameter combinations
    print("\nHyperparameter combinations to be tested:")
    for param_name, values in training_config_custom.__dict__.items():
        if isinstance(values, list) and not param_name.startswith('__'):
            print(f"  {param_name}: {values}")
    
    # Get entity mode and dataset path from first config (they should be the same for all configs)
    entity_mode = hyperparameter_grid[0]['ENTITY_MODE']
    training_data_path = os.path.join(project_root, hyperparameter_grid[0]['TRAINING_DATA_PATH'])
    dataset_name = os.path.basename(training_data_path)
    
    # Set the entity mode in the main config
    main_config.ENTITY_MODE = entity_mode
    
    print(f"\n{'='*80}")
    print(f"PREPARING DATA (ONLY ONCE)")
    print(f"Using entity mode: {entity_mode}")
    print(f"Loading training data from: {training_data_path}")
    print(f"{'='*80}\n")
    
    # Load vocabulary once
    vocab_full_path = os.path.join(project_root, VOCAB_PATH)
    
    # Check if vocabulary file exists - it should be built before training
    if not os.path.exists(vocab_full_path):
        print(f"Error: Vocabulary file not found at {vocab_full_path}")
        print("Please run build_vocab.py first to create the vocabulary.")
        return
    
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
        return
    
    # Load and prepare data once
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
    
    # Prepare features and labels once
    max_distance = hyperparameter_grid[0]['MAX_DISTANCE']  # Use the first config's MAX_DISTANCE
    features, labels, _ = prepare_custom_training_data(
        training_data_path, max_distance, None, data_split_mode='all',
        prepared_data=prepared_train_data, relationship_gold=relationship_gold
    )
    print(f"Loaded {len(features)} examples")
    
    # Check class balance
    labels_info = {}
    if len(labels) > 0:
        positive = sum(labels)
        negative = len(labels) - positive
        positive_pct = positive/len(labels)*100
        labels_info['positive'] = positive
        labels_info['negative'] = negative
        labels_info['positive_pct'] = positive_pct
        print(f"Class distribution: {positive} positive examples ({positive_pct:.1f}%), {negative} negative examples ({(100-positive_pct):.1f}%)")
        
        # Check if we have any positive examples
        if positive == 0:
            print("ERROR: No positive examples found. The model will not learn anything useful.")
            print("Please check your data format and ensure that relationship_gold contains valid relationships.")
            return
    else:
        print("ERROR: No examples found in the dataset!")
        print("Please check your data format and ensure that extracted_disorders and formatted_dates are correctly parsed.")
        return
    
    # Create train / val / test datasets once
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
    
    # Create datasets once using config settings from first config
    max_context_len = hyperparameter_grid[0]['MAX_CONTEXT_LEN']
    train_dataset = ClinicalNoteDataset(
        train_features, train_labels, vocab_instance, 
        max_context_len, max_distance,
        entity_category_map
    ) 
    val_dataset = ClinicalNoteDataset(
        val_features, val_labels, vocab_instance, 
        max_context_len, max_distance,
        entity_category_map
    )
    
    # Track the best model across all runs
    best_val_f1 = 0
    best_model_state = None
    best_config = None
    best_metrics = None
    best_model_name = None
    
    # Train with each hyperparameter combination
    start_time = time.time()
    for i, config in enumerate(hyperparameter_grid):
        print(f"\nTraining run {i+1}/{len(hyperparameter_grid)}")
        val_f1, model_state, metrics, model_name, run_config = train_with_config(
            config, train_dataset, val_dataset, train_features, val_features, labels_info
        )
        
        # Update best model if this run is better
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model_state
            best_config = run_config
            best_metrics = metrics
            best_model_name = model_name
            print(f"\n*** New best model found with validation F1-score: {best_val_f1:.4f} ***")
            print(f"   Precision: {metrics['best_val_precision']:.4f}, Recall: {metrics['best_val_recall']:.4f}, Accuracy: {metrics['best_val_acc']:.2f}%, Threshold: {metrics['best_val_threshold']:.2f}")
    
    # Print summary of grid search
    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Grid search completed in {elapsed_time:.2f} seconds.")
    print(f"Best validation F1-score: {best_val_f1:.4f}")
    print(f"Best validation Precision: {best_metrics['best_val_precision']:.4f}")
    print(f"Best validation Recall: {best_metrics['best_val_recall']:.4f}")
    print(f"Best validation Accuracy: {best_metrics['best_val_acc']:.2f}%")
    print(f"Best F1 Threshold: {best_metrics['best_val_threshold']:.2f}")
    
    # Save the best model once at the end of the grid search
    if best_model_state is not None:
        model_full_path = os.path.join(project_root, os.path.dirname(MODEL_PATH), best_model_name)
        # Ensure directory exists before saving
        os.makedirs(os.path.dirname(model_full_path), exist_ok=True)
        torch.save(best_model_state, model_full_path)
        print(f"Best model saved to: {model_full_path}")
        
        # Plot training curves for the best model only
        plot_save_path = os.path.join(project_root, os.path.dirname(MODEL_PATH), 
                                     f"{os.path.splitext(best_model_name)[0]}_training_curves")
        
        # Add title suffix indicating if weighted loss was used
        title_suffix = ""
        if best_config.get('USE_WEIGHTED_LOSS', False):
            pos_weight = best_config.get('POS_WEIGHT', None)
            if pos_weight is None and 'positive' in labels_info and 'negative' in labels_info:
                pos_weight = labels_info['negative'] / max(1, labels_info['positive'])
            
            if pos_weight is not None:
                title_suffix = f" (Weighted Loss, w={pos_weight:.2f})"
        
        # Plot training curves for the best model
        plot_training_curves(best_metrics, plot_save_path, title_suffix=title_suffix)
    
    print(f"Best hyperparameters:")
    for key, value in best_config.items():
        if isinstance(getattr(training_config_custom, key, None), list):
            print(f"  {key}: {value}")
    print(f"{'='*80}")
    
    print("Done!")

if __name__ == "__main__":
    train()