import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import time
import random
import numpy as np
from transformers import AutoTokenizer

# Add parent directory to path to allow importing from models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEVICE, MODEL_PATH, TRAINING_SET_RATIO, DATA_SPLIT_RANDOM_SEED
from custom_model_training.training_config_custom import VOCAB_PATH
from utils.inference_eval_utils import load_and_prepare_data
from utils.training_utils import generate_hyperparameter_grid, log_training_run, plot_training_curves
import config as main_config
import custom_model_training.training_config_custom as training_config_custom
from custom_model_training.DiagnosisDateRelationModel import DiagnosisDateRelationModel
from custom_model_training.ClinicalNoteDataset import ClinicalNoteDataset
from custom_model_training.training_utils_custom import prepare_custom_training_data, train_model

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility across all libraries used in training.
    
    Args:
        seed (int): The random seed to use (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seeds set to {seed} for reproducibility")

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
    
    # Create model name based on dataset name only
    model_name = f"{os.path.splitext(dataset_name)[0]}_custom.pt"
    
    # Create data loaders using config batch size
    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'])
    
    # Step 4: Initialize and train model using config settings
    # Check if weighted loss should be used
    use_weighted_loss = config.get('USE_WEIGHTED_LOSS', False)
    pos_weight = None
    if use_weighted_loss:
        # Calculate or use provided positive weight
        pos_weight = config.get('POS_WEIGHT', None)
        if pos_weight is None and 'positive' in labels_info and 'negative' in labels_info:
            # Auto-calculate from data
            pos_weight = labels_info['negative'] / max(1, labels_info['positive'])
            print(f"Auto-calculated positive weight: {pos_weight:.2f}")
            # Update the config with the calculated value so it gets logged properly
            config['POS_WEIGHT'] = pos_weight
        
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
    
    # Check if we should use pre-trained embeddings
    pretrained_embeddings = None
    if config.get('USE_PRETRAINED_EMBEDDINGS', False):
        pretrained_embeddings_path = os.path.join(project_root, config.get('PRETRAINED_EMBEDDINGS_PATH', ''))
        if os.path.exists(pretrained_embeddings_path):
            print(f"Loading pre-trained embeddings from: {pretrained_embeddings_path}")
            try:
                # Load pre-trained embeddings
                pretrained_embeddings = torch.load(pretrained_embeddings_path)
                
                # Check if the embeddings match our vocabulary size and embedding dimension
                if pretrained_embeddings.shape[0] != train_dataset.vocab.n_words:
                    print(f"Warning: Pre-trained embeddings vocab size ({pretrained_embeddings.shape[0]}) " +
                          f"doesn't match current vocab size ({train_dataset.vocab.n_words}). " +
                          f"Will not use pre-trained embeddings.")
                    pretrained_embeddings = None
                elif pretrained_embeddings.shape[1] != config['EMBEDDING_DIM']:
                    print(f"Warning: Pre-trained embeddings dimension ({pretrained_embeddings.shape[1]}) " +
                          f"doesn't match specified EMBEDDING_DIM ({config['EMBEDDING_DIM']}). " +
                          f"Will not use pre-trained embeddings.")
                    pretrained_embeddings = None
                else:
                    print(f"Pre-trained embeddings loaded successfully with shape: {pretrained_embeddings.shape}")
            except Exception as e:
                print(f"Error loading pre-trained embeddings: {e}")
                pretrained_embeddings = None
        else:
            print(f"Pre-trained embeddings file not found at: {pretrained_embeddings_path}")
    
    # Initialize model with apply_sigmoid=False when using BCEWithLogitsLoss
    model = DiagnosisDateRelationModel(
        vocab_size=train_dataset.vocab.n_words, 
        embedding_dim=config['EMBEDDING_DIM'],
        hidden_dim=config['HIDDEN_DIM'],
        apply_sigmoid=not use_weighted_loss,  # False when using weighted loss (BCEWithLogitsLoss)
        pretrained_embeddings=pretrained_embeddings
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
        'USE_PRETRAINED_EMBEDDINGS': config.get('USE_PRETRAINED_EMBEDDINGS', False),
        'PRETRAINED_EMBEDDINGS_PATH': config.get('PRETRAINED_EMBEDDINGS_PATH', ''),
        'train_examples': len(train_features),
        'val_examples': len(val_features),
        'positive_examples_pct': labels_info['positive_pct']
    }
    
    # Don't log here, only log the best model at the end of grid search
    model_full_path = os.path.join(project_root, 'custom_model_training/models', model_name)
    # Disabled logging for individual runs
    if False:
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
    
    # Set random seeds for reproducibility
    set_random_seeds(DATA_SPLIT_RANDOM_SEED)
    
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
            loaded_data = torch.load(vocab_full_path, weights_only=False)
        except TypeError:
            # For older PyTorch versions that don't have weights_only
            loaded_data = torch.load(vocab_full_path)
        
        # Check the type of loaded data and handle accordingly
        from custom_model_training.Vocabulary import Vocabulary
        
        # If it's a dictionary (like from BERT), convert to our Vocabulary format
        if isinstance(loaded_data, dict):
            print("Loaded a dictionary-style vocabulary, converting to Vocabulary object...")
            vocab_instance = Vocabulary()
            
            # Reset the vocabulary to empty
            vocab_instance.word2idx = {}
            vocab_instance.idx2word = {}
            vocab_instance.n_words = 0
            
            # Add special tokens first to ensure they have the expected indices
            special_tokens = ['<pad>', '<unk>', '<cls>', '<sep>', '<mask>']
            for token in special_tokens:
                if token in loaded_data:
                    vocab_instance.add_word(token)
            
            # Add all other tokens
            for word, idx in loaded_data.items():
                if word not in vocab_instance.word2idx:
                    vocab_instance.add_word(word)
            
            print(f"Converted dictionary to Vocabulary object successfully.")
        elif hasattr(loaded_data, 'n_words'):
            # It's already a Vocabulary object
            vocab_instance = loaded_data
        else:
            raise TypeError("Loaded data is neither a dictionary nor a Vocabulary object")
            
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
    
    # Create train / val datasets using patient-based splitting to prevent data leakage
    
    # First, extract patient IDs from features
    patient_ids = []
    for feature in features:
        patient_id = feature.get('patient_id')
        patient_ids.append(patient_id)
    
    # Get unique patient IDs
    unique_patient_ids = list(set(patient_id for patient_id in patient_ids if patient_id is not None))
    
    if len(unique_patient_ids) > 1:
        print(f"Found {len(unique_patient_ids)} unique patients for patient-based splitting")
        
        # Create a stratification array based on whether patients have positive examples
        patients_with_positive = set()
        
        # Find patients with positive examples
        for i, (feature, label) in enumerate(zip(features, labels)):
            if label == 1 and feature.get('patient_id') is not None:
                patients_with_positive.add(feature.get('patient_id'))
        
        # Create stratification labels: 1 for patients with positive examples, 0 for others
        stratify_labels = [1 if p in patients_with_positive else 0 for p in unique_patient_ids]
        
        # Only stratify if we have both positive and negative examples
        if len(set(stratify_labels)) > 1:
            print("Using stratified patient-based splitting")
            train_patients, val_patients = train_test_split(
                unique_patient_ids,
                test_size=0.2,
                random_state=42,
                stratify=stratify_labels
            )
        else:
            print("Using non-stratified patient-based splitting (all patients have same label)")
            train_patients, val_patients = train_test_split(
                unique_patient_ids,
                test_size=0.2,
                random_state=42
            )
        
        # Now split features and labels based on patient IDs
        train_indices = [i for i, feature in enumerate(features) 
                        if feature.get('patient_id') in train_patients]
        val_indices = [i for i, feature in enumerate(features) 
                      if feature.get('patient_id') in val_patients]
        
        train_features = [features[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        val_features = [features[i] for i in val_indices]
        val_labels = [labels[i] for i in val_indices]
        
        print(f"Patient-based split: Train patients: {len(train_patients)}, Validation patients: {len(val_patients)}")
    else:
        # Fall back to traditional feature-based splitting if patient IDs are not available
        print("Warning: Patient IDs not available or only one patient found. Falling back to feature-based splitting.")
        train_features, val_features, train_labels, val_labels = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
    
    print(f"Train: {len(train_features)} examples, Validation: {len(val_features)} examples")
    
    # Create entity category map for multi-entity mode
    entity_category_map = {
        'diagnosis': 0,
        'symptom': 1,
        'procedure': 2,
        'medication': 3
    }
    
    # Create datasets once using config settings from first config
    max_context_len = hyperparameter_grid[0]['MAX_CONTEXT_LEN']
    
    # Check if we should use a BERT tokenizer for consistency with pre-trained embeddings
    tokenizer = None
    if hyperparameter_grid[0].get('USE_PRETRAINED_EMBEDDINGS', False):
        # Try to load the BERT tokenizer if specified
        bert_model_name = hyperparameter_grid[0].get('BERT_MODEL_NAME', 'emilyalsentzer/Bio_ClinicalBERT')
        try:
            print(f"Loading BERT tokenizer from {bert_model_name} for consistent tokenization...")
            tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            print("BERT tokenizer loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load BERT tokenizer: {e}")
            print("Falling back to simple whitespace tokenization.")
    
    train_dataset = ClinicalNoteDataset(
        train_features, train_labels, vocab_instance, 
        max_context_len, max_distance,
        entity_category_map,
        tokenizer=tokenizer
    ) 
    val_dataset = ClinicalNoteDataset(
        val_features, val_labels, vocab_instance, 
        max_context_len, max_distance,
        entity_category_map,
        tokenizer=tokenizer
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
        model_full_path = os.path.join(project_root, 'custom_model_training/models', best_model_name)
        # Ensure directory exists before saving
        os.makedirs(os.path.dirname(model_full_path), exist_ok=True)
        
        # Create a dictionary to save both model state and hyperparameters
        # Create a simplified version of the config with only serializable data
        serializable_config = {
            'ENTITY_MODE': best_config.get('ENTITY_MODE'),
            'MAX_DISTANCE': best_config.get('MAX_DISTANCE'),
            'MAX_CONTEXT_LEN': best_config.get('MAX_CONTEXT_LEN'),
            'EMBEDDING_DIM': best_config.get('EMBEDDING_DIM'),
            'HIDDEN_DIM': best_config.get('HIDDEN_DIM'),
            'BATCH_SIZE': best_config.get('BATCH_SIZE'),
            'LEARNING_RATE': best_config.get('LEARNING_RATE'),
            'NUM_EPOCHS': best_config.get('NUM_EPOCHS'),
            'DROPOUT': best_config.get('DROPOUT'),
            'USE_DISTANCE_FEATURE': best_config.get('USE_DISTANCE_FEATURE'),
            'USE_POSITION_FEATURE': best_config.get('USE_POSITION_FEATURE'),
            'ENTITY_CATEGORY_EMBEDDING_DIM': best_config.get('ENTITY_CATEGORY_EMBEDDING_DIM'),
            'USE_WEIGHTED_LOSS': best_config.get('USE_WEIGHTED_LOSS', False),
            'POS_WEIGHT': best_config.get('POS_WEIGHT'),
            'USE_PRETRAINED_EMBEDDINGS': best_config.get('USE_PRETRAINED_EMBEDDINGS', False),
            'PRETRAINED_EMBEDDINGS_PATH': best_config.get('PRETRAINED_EMBEDDINGS_PATH', '')
        }
        
        save_dict = {
            'hyperparameters': serializable_config,
            'model_state_dict': best_model_state,
            'vocab_size': train_dataset.vocab.n_words,
            'best_threshold': best_metrics['best_val_threshold']
        }
        
        torch.save(save_dict, model_full_path)
        print(f"Best model and config saved to: {model_full_path}")
        
        # Plot training curves for the best model only
        # Extract dataset name from model name
        dataset_name = os.path.splitext(best_model_name)[0].replace('_custom', '')
        plot_save_path = os.path.join(project_root, 'custom_model_training/plots', 
                                     f"{dataset_name}_custom")
        
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
    
    # Log only the best model at the end of grid search
    if best_model_state is not None:
        # Extract dataset name from best model name
        dataset_name = os.path.basename(best_config['TRAINING_DATA_PATH'])
        entity_mode = best_config['ENTITY_MODE']
        
        # Create hyperparams dict for logging (reuse the serializable_config)
        hyperparams_for_logging = {
            'ENTITY_MODE': best_config.get('ENTITY_MODE'),
            'MAX_DISTANCE': best_config.get('MAX_DISTANCE'),
            'MAX_CONTEXT_LEN': best_config.get('MAX_CONTEXT_LEN'),
            'EMBEDDING_DIM': best_config.get('EMBEDDING_DIM'),
            'HIDDEN_DIM': best_config.get('HIDDEN_DIM'),
            'BATCH_SIZE': best_config.get('BATCH_SIZE'),
            'LEARNING_RATE': best_config.get('LEARNING_RATE'),
            'NUM_EPOCHS': best_config.get('NUM_EPOCHS'),
            'DROPOUT': best_config.get('DROPOUT'),
            'USE_DISTANCE_FEATURE': best_config.get('USE_DISTANCE_FEATURE'),
            'USE_POSITION_FEATURE': best_config.get('USE_POSITION_FEATURE'),
            'ENTITY_CATEGORY_EMBEDDING_DIM': best_config.get('ENTITY_CATEGORY_EMBEDDING_DIM'),
            'USE_WEIGHTED_LOSS': best_config.get('USE_WEIGHTED_LOSS', False),
            'POS_WEIGHT': best_config.get('POS_WEIGHT'),  # This will now have the auto-calculated value
            'USE_PRETRAINED_EMBEDDINGS': best_config.get('USE_PRETRAINED_EMBEDDINGS', False),
            'PRETRAINED_EMBEDDINGS_PATH': best_config.get('PRETRAINED_EMBEDDINGS_PATH', ''),
            'train_examples': len(train_features),
            'val_examples': len(val_features),
            'positive_examples_pct': labels_info['positive_pct']
        }
        
        # Add a debug print to verify POS_WEIGHT is being passed correctly
        if best_config.get('USE_WEIGHTED_LOSS', False):
            print(f"Logging POS_WEIGHT: {best_config.get('POS_WEIGHT')}")
        
        # Log the best model
        log_training_run(
            model_full_path,
            hyperparams_for_logging,
            best_metrics,
            dataset_name,
            entity_mode,
            os.path.join(project_root, 'custom_model_training'),
            'custom_model_training_log.csv'
        )

    print(f"Best hyperparameters:")
    for key, value in best_config.items():
        if isinstance(getattr(training_config_custom, key, None), list):
            print(f"  {key}: {value}")
    print(f"{'='*80}")

    print("Done!")

if __name__ == "__main__":
    train()