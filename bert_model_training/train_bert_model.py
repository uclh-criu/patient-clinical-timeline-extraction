import os
import sys
import time
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

# Adjust relative paths for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import configuration
import config
import bert_model_training.training_config_bert as training_config

# Import utility functions
from bert_model_training.training_utils_bert import prepare_bert_training_data, train_bert_model
from utils.training_utils import generate_hyperparameter_grid, log_training_run, plot_training_curves
from utils.inference_eval_utils import load_and_prepare_data

def train_with_config(hyperparams, train_dataset, val_dataset, tokenizer):
    """
    Train a BERT model with the given hyperparameter configuration.
    
    Args:
        hyperparams (dict): Dictionary of hyperparameters
        train_dataset: Training dataset
        val_dataset: Validation dataset
        tokenizer: BERT tokenizer
        
    Returns:
        tuple: (best_val_acc, model_path, metrics)
    """
    print(f"\n{'='*80}")
    print(f"Training with configuration:")
    # Print only the hyperparameters that are being tuned (those that are lists in the original config)
    for key, value in hyperparams.items():
        if isinstance(getattr(training_config, key, None), list):
            print(f"  {key}: {value}")
    print(f"{'='*80}\n")
    
    # Get entity mode from config
    entity_mode = hyperparams['ENTITY_MODE']
    
    # Get dataset name from training data path
    dataset_path = hyperparams['BERT_TRAINING_DATA_PATH']
    dataset_name = os.path.basename(dataset_path)
    
    # Create model name based on dataset and entity mode
    model_name = f"bert_{os.path.splitext(dataset_name)[0]}_{entity_mode}.pt"
    
    # Ensure model directory exists
    model_dir = os.path.join(project_root, 'bert_model_training/bert_model')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparams['BERT_BATCH_SIZE'],
        shuffle=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=hyperparams['BERT_BATCH_SIZE'],
            shuffle=False
        )
    
    # Initialize model
    print(f"Loading pre-trained model: {hyperparams['BERT_PRETRAINED_MODEL']}")
    model = AutoModelForSequenceClassification.from_pretrained(
        hyperparams['BERT_PRETRAINED_MODEL'],
        num_labels=1,  # Binary classification
        hidden_dropout_prob=hyperparams['BERT_DROPOUT']
    )
    
    # Add special tokens for entity marking
    special_tokens = {'additional_special_tokens': ['[E1]', '[/E1]', '[E2]', '[/E2]']}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Set up optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=hyperparams['BERT_LEARNING_RATE']
    )
    
    # Calculate total training steps
    total_steps = len(train_loader) * hyperparams['BERT_NUM_TRAIN_EPOCHS']
    
    # Create scheduler with linear warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),  # 10% of total steps for warmup
        num_training_steps=total_steps
    )
    
    # Train the model
    print(f"Starting training for {hyperparams['BERT_NUM_TRAIN_EPOCHS']} epochs...")
    metrics = train_bert_model(
        model,
        tokenizer,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        hyperparams['BERT_NUM_TRAIN_EPOCHS'],
        device,
        model_path
    )
    
    # Plot training curves
    curves_path = os.path.join(model_dir, f"{os.path.splitext(model_name)[0]}_training_curves")
    plot_training_curves(metrics, curves_path)
    
    # Log the training run
    hyperparams_for_logging = {
        'ENTITY_MODE': entity_mode,
        'BERT_PRETRAINED_MODEL': hyperparams['BERT_PRETRAINED_MODEL'],
        'BERT_MAX_SEQ_LENGTH': hyperparams['BERT_MAX_SEQ_LENGTH'],
        'BERT_BATCH_SIZE': hyperparams['BERT_BATCH_SIZE'],
        'BERT_LEARNING_RATE': hyperparams['BERT_LEARNING_RATE'],
        'BERT_NUM_TRAIN_EPOCHS': hyperparams['BERT_NUM_TRAIN_EPOCHS'],
        'BERT_DROPOUT': hyperparams['BERT_DROPOUT'],
        'train_examples': len(train_dataset),
        'val_examples': len(val_dataset) if val_dataset else 0
    }
    
    log_training_run(
        model_path,
        hyperparams_for_logging,
        metrics,
        dataset_name,
        entity_mode,
        os.path.join(project_root, 'bert_model_training'),
        'bert_model_training_log.csv'
    )
    
    return metrics['best_val_acc'], model_path, metrics

def main():
    """
    Main function to train the BERT model with grid search.
    """
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Generate hyperparameter grid
    print("Generating hyperparameter grid for training...")
    hyperparameter_grid = generate_hyperparameter_grid(training_config)
    
    # Print summary of all hyperparameter combinations
    print(f"\n{'='*80}")
    print(f"Grid Search Summary: {len(hyperparameter_grid)} hyperparameter combinations")
    print(f"{'='*80}")
    for i, params in enumerate(hyperparameter_grid):
        print(f"Combination {i+1}:")
        for key, value in params.items():
            if isinstance(getattr(training_config, key, None), list):
                print(f"  {key}: {value}")
    print(f"{'='*80}\n")
    
    # Get the first configuration to use for data loading
    first_config = hyperparameter_grid[0]
    entity_mode = first_config['ENTITY_MODE']
    
    # Set the entity mode in the main config
    import config as main_config  # Import here to avoid name collision
    main_config.ENTITY_MODE = entity_mode
    
    # Get the full path for the training data
    training_data_path = os.path.join(project_root, first_config['BERT_TRAINING_DATA_PATH'])
    
    # Ensure the data file exists
    if not os.path.exists(training_data_path):
        print(f"Error: Training data file not found at {training_data_path}")
        return
    
    print(f"Loading training data from: {training_data_path}")
    
    # Load data once for all hyperparameter combinations
    train_dataset, val_dataset, tokenizer = prepare_bert_training_data(
        training_data_path, 
        first_config['BERT_PRETRAINED_MODEL'],
        first_config['BERT_MAX_SEQ_LENGTH'],
        data_split_mode='train'
    )
    
    # Check if we have training data
    if train_dataset is None or len(train_dataset) == 0:
        print("ERROR: No training examples were created. Check your data format.")
        return
    
    # Track the best model across all runs
    best_val_acc = 0
    best_model_path = None
    best_config = None
    best_metrics = None
    
    # Train with each hyperparameter combination
    start_time = time.time()
    for i, config in enumerate(hyperparameter_grid):
        print(f"\nTraining run {i+1}/{len(hyperparameter_grid)}")
        val_acc, model_path, metrics = train_with_config(config, train_dataset, val_dataset, tokenizer)
        
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
        if isinstance(getattr(training_config, key, None), list):
            print(f"  {key}: {value}")
    print(f"{'='*80}")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main() 