"""
Shared training utilities for both custom and BERT models.
These functions are model-agnostic and can be used by any training pipeline.
"""

import os
import csv
import datetime
import itertools
import matplotlib.pyplot as plt
import torch

def generate_hyperparameter_grid(config_module):
    """
    Generate a grid of hyperparameters for grid search.
    
    Args:
        config_module: The config module containing hyperparameters
        
    Returns:
        list: List of dictionaries, each containing a unique combination of hyperparameters
    """
    # If grid search is disabled, return a single configuration with the first value of each list
    if not getattr(config_module, 'ENABLE_GRID_SEARCH', True):
        config = {}
        for key, value in vars(config_module).items():
            if key.startswith('__'):
                continue
            if isinstance(value, list):
                config[key] = value[0]  # Take the first value
            else:
                config[key] = value
        return [config]
    
    # Identify hyperparameters that are lists
    param_grid = {}
    for key, value in vars(config_module).items():
        if key.startswith('__'):
            continue
        if isinstance(value, list) and len(value) > 0:
            param_grid[key] = value
    
    # If no lists are found, return the original config
    if not param_grid:
        return [vars(config_module)]
    
    # Generate all combinations of hyperparameters
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    # Create a list of dictionaries with all combinations
    grid = []
    for combo in combinations:
        config = {}
        # Copy all parameters from config_module
        for key, value in vars(config_module).items():
            if key.startswith('__'):
                continue
            config[key] = value
        
        # Override with the current combination
        for i, key in enumerate(keys):
            config[key] = combo[i]
        
        grid.append(config)
    
    return grid

def log_training_run(model_path, hyperparams, metrics, dataset_name, entity_mode, log_dir, log_filename='training_log.csv'):
    """
    Log training run details to a CSV file.
    
    Args:
        model_path (str): Path to the saved model
        hyperparams (dict): Dictionary of hyperparameters used for training
        metrics (dict): Dictionary of training metrics (e.g., val_acc, val_loss)
        dataset_name (str): Name of the dataset used for training
        entity_mode (str): Entity mode used for training ('diagnosis_only' or 'multi_entity')
        log_dir (str): Directory to save the log file
        log_filename (str): Name of the log file (default: 'training_log.csv')
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Define the log file path
    log_file = os.path.join(log_dir, log_filename)
    
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Extract model filename from path
    model_filename = os.path.basename(model_path)
    
    # Prepare the log entry
    log_entry = {
        'timestamp': timestamp,
        'dataset': dataset_name,
        'entity_mode': entity_mode,
        'model_filename': model_filename,
        'val_accuracy': metrics.get('val_acc', 0),
        'val_loss': metrics.get('val_loss', 0),
        'train_loss': metrics.get('train_loss', 0),
        'epochs': metrics.get('epochs', 0),
    }
    
    # Add all hyperparameters to the log entry
    for key, value in hyperparams.items():
        log_entry[key] = value
    
    # Check if the log file exists
    file_exists = os.path.isfile(log_file)
    
    # Write to the CSV file
    try:
        with open(log_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            
            # Write header if file doesn't exist
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(log_entry)
        
        print(f"Training run logged to {log_file}")
    except Exception as e:
        print(f"Error logging training run: {e}")

def plot_training_curves(metrics, save_path, show_plot=False):
    """
    Plot training curves for model training.
    
    Args:
        metrics (dict): Dictionary containing training metrics (train_losses, val_losses, val_accs)
        save_path: Path to save the plot
        show_plot: Whether to display the plot (default: False)
    """
    # Make sure the save path has a proper extension
    if not save_path.endswith('.png'):
        # If no .png extension, add it
        save_path = save_path + '.png'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    train_losses = metrics['train_losses']
    val_losses = metrics['val_losses']
    val_accs = metrics['val_accs']
    
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(epochs, val_accs, 'g-')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    
    plt.tight_layout()
    
    # Ensure directory exists before saving plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    print(f"Training curves saved to {save_path}") 