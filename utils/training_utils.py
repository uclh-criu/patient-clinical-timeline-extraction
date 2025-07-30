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
    
    # Determine which model type this is based on the log filename
    is_bert_model = 'bert' in log_filename.lower()
    
    # Define columns based on model type
    if is_bert_model:
        # BERT model columns
        columns = [
            'timestamp', 'dataset', 'entity_mode', 'model_filename',
            'val_accuracy', 'val_f1', 'val_precision', 'val_recall', 'val_threshold',
            'val_loss', 'train_loss', 'epochs',
            # Common parameters
            'ENTITY_MODE',
            # BERT model parameters
            'BERT_PRETRAINED_MODEL', 'BERT_MAX_SEQ_LENGTH', 'BERT_BATCH_SIZE',
            'BERT_LEARNING_RATE', 'BERT_NUM_TRAIN_EPOCHS', 'BERT_DROPOUT',
            # Dataset statistics
            'train_examples', 'val_examples', 'positive_examples_pct'
        ]
    else:
        # Custom model columns
        columns = [
            'timestamp', 'dataset', 'entity_mode', 'model_filename',
            'val_accuracy', 'val_f1', 'val_precision', 'val_recall', 'val_threshold',
            'val_loss', 'train_loss', 'epochs',
            # Common parameters
            'ENTITY_MODE',
            # Custom model parameters
            'MAX_DISTANCE', 'MAX_CONTEXT_LEN', 'EMBEDDING_DIM', 'HIDDEN_DIM',
            'BATCH_SIZE', 'LEARNING_RATE', 'NUM_EPOCHS', 'DROPOUT',
            'USE_DISTANCE_FEATURE', 'USE_POSITION_FEATURE', 'ENTITY_CATEGORY_EMBEDDING_DIM',
            'USE_WEIGHTED_LOSS', 'POS_WEIGHT',
            # Dataset statistics
            'train_examples', 'val_examples', 'positive_examples_pct'
        ]
    
    # Prepare the log entry with standard metrics
    log_entry = {
        'timestamp': timestamp,
        'dataset': dataset_name,
        'entity_mode': entity_mode,
        'model_filename': model_filename,
        'val_accuracy': metrics.get('best_val_acc', 0),
        'val_f1': metrics.get('best_val_f1', 0),
        'val_precision': metrics.get('best_val_precision', 0),
        'val_recall': metrics.get('best_val_recall', 0),
        'val_threshold': metrics.get('best_val_threshold', 0.5),
        'val_loss': metrics.get('val_loss', 0),
        'train_loss': metrics.get('train_loss', 0),
        'epochs': metrics.get('epochs', 0),
    }
    
    # Add hyperparameters to the log entry
    for key, value in hyperparams.items():
        log_entry[key] = value
    
    # Create an ordered dictionary with model-specific columns
    ordered_log_entry = {}
    for column in columns:
        ordered_log_entry[column] = log_entry.get(column, '')
    
    # Check if the log file exists
    file_exists = os.path.isfile(log_file)
    
    # Write to the CSV file
    try:
        with open(log_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            
            # Write header if file doesn't exist
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(ordered_log_entry)
        
        print(f"Training run logged to {log_file}")
    except Exception as e:
        print(f"Error logging training run: {e}")

def plot_training_curves(metrics, save_path, title_suffix="", show_plot=False):
    """
    Plot training curves for model training.
    
    Args:
        metrics (dict): Dictionary containing training metrics (train_losses, val_losses, val_accs, val_f1s, etc.)
        save_path: Path to save the plot
        title_suffix (str): Optional suffix to add to the plot titles
        show_plot: Whether to display the plot (default: False)
    """
    # Make sure the save path has a proper extension
    if not save_path.endswith('.png'):
        # If no .png extension, add it
        save_path = save_path + '.png'
    
    # Create a figure with subplots - add an extra row for thresholds
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    
    # Plot losses
    train_losses = metrics['train_losses']
    val_losses = metrics['val_losses']
    val_accs = metrics['val_accs']
    
    epochs = range(1, len(train_losses) + 1)
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss')
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss')
    axes[0, 0].set_title(f'Training and Validation Loss{title_suffix}')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Plot accuracy
    axes[0, 1].plot(epochs, val_accs, 'g-')
    axes[0, 1].set_title(f'Validation Accuracy{title_suffix}')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Accuracy (%)')
    
    # Plot F1-score if available
    if 'val_f1s' in metrics:
        val_f1s = metrics['val_f1s']
        axes[1, 0].plot(epochs, val_f1s, 'm-')
        axes[1, 0].set_title(f'Validation F1-Score{title_suffix}')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('F1-Score')
        
        # Mark the best F1-score
        best_epoch_idx = val_f1s.index(max(val_f1s))
        best_f1 = val_f1s[best_epoch_idx]
        axes[1, 0].plot(best_epoch_idx + 1, best_f1, 'ro', markersize=8)
        axes[1, 0].annotate(f'Best: {best_f1:.4f}', 
                           xy=(best_epoch_idx + 1, best_f1),
                           xytext=(best_epoch_idx + 1 - 0.5, best_f1 + 0.05),
                           fontsize=10)
    
    # Plot precision and recall if available
    if 'val_precisions' in metrics and 'val_recalls' in metrics:
        val_precisions = metrics['val_precisions']
        val_recalls = metrics['val_recalls']
        axes[1, 1].plot(epochs, val_precisions, 'c-', label='Precision')
        axes[1, 1].plot(epochs, val_recalls, 'y-', label='Recall')
        axes[1, 1].set_title(f'Validation Precision and Recall{title_suffix}')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
    
    # Plot thresholds if available
    if 'val_thresholds' in metrics:
        val_thresholds = metrics['val_thresholds']
        axes[2, 0].plot(epochs, val_thresholds, 'k-')
        axes[2, 0].set_title(f'Best F1 Thresholds{title_suffix}')
        axes[2, 0].set_xlabel('Epochs')
        axes[2, 0].set_ylabel('Threshold')
        axes[2, 0].set_ylim([0, 1])
        
        # Mark the best threshold
        best_epoch_idx = val_f1s.index(max(val_f1s)) if 'val_f1s' in metrics else 0
        best_threshold = val_thresholds[best_epoch_idx]
        axes[2, 0].plot(best_epoch_idx + 1, best_threshold, 'ro', markersize=8)
        axes[2, 0].annotate(f'Best: {best_threshold:.2f}', 
                           xy=(best_epoch_idx + 1, best_threshold),
                           xytext=(best_epoch_idx + 1 - 0.5, best_threshold + 0.05),
                           fontsize=10)
    
    # Keep the last subplot empty or use it for additional information
    axes[2, 1].axis('off')  # Turn off axis
    
    # Add a text box with key metrics
    if 'best_val_f1' in metrics:
        best_f1 = metrics['best_val_f1']
        best_precision = metrics.get('best_val_precision', 0)
        best_recall = metrics.get('best_val_recall', 0)
        best_acc = metrics.get('best_val_acc', 0)
        best_threshold = metrics.get('best_val_threshold', 0.5)
        
        info_text = (f"Best F1-Score: {best_f1:.4f}\n"
                    f"Precision: {best_precision:.4f}\n"
                    f"Recall: {best_recall:.4f}\n"
                    f"Accuracy: {best_acc:.2f}%\n"
                    f"Threshold: {best_threshold:.2f}")
        
        axes[2, 1].text(0.1, 0.5, info_text, fontsize=12,
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Add overall title
    fig.suptitle(f'Training Metrics{title_suffix}', fontsize=16)
    plt.subplots_adjust(top=0.92)  # Make room for the suptitle
    
    # Ensure directory exists before saving plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    print(f"Training curves saved to {save_path}") 