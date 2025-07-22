import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

# Add parent directory to path to allow importing from other modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import configuration
import config
import model_training.training_config as training_config
from model_training.BertEntityPairDataset import BertEntityPairDataset

# Import utility functions
from utils.training_utils import prepare_bert_training_data, train_bert_model, plot_training_curves

def train():
    """Train the BERT model for relation extraction."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get training data path from training_config
    data_path = os.path.join(project_root, training_config.BERT_TRAINING_DATA_PATH)
    print(f"Using training data from: {data_path}")
    
    # Ensure model directory exists
    model_path = os.path.join(project_root, config.BERT_MODEL_PATH)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Prepare data
    train_dataset, val_dataset, tokenizer = prepare_bert_training_data(
        data_path, 
        training_config.BERT_PRETRAINED_MODEL,
        training_config.BERT_MAX_SEQ_LENGTH
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.BERT_BATCH_SIZE,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.BERT_BATCH_SIZE,
        shuffle=False
    )
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        training_config.BERT_PRETRAINED_MODEL,
        num_labels=2  # Binary classification
    )
    
    # Resize token embeddings to account for new special tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Move model to device
    model.to(device)
    
    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=training_config.BERT_LEARNING_RATE)
    
    total_steps = len(train_loader) * training_config.BERT_NUM_TRAIN_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Train the model
    print(f"Starting training for {training_config.BERT_NUM_TRAIN_EPOCHS} epochs...")
    train_losses, val_losses, val_accuracies = train_bert_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        training_config.BERT_NUM_TRAIN_EPOCHS,
        device,
        model_path
    )
    
    # Save tokenizer separately
    tokenizer.save_pretrained(model_path)
    print(f"Tokenizer saved to {model_path}")
    
    # Plot training curves
    plot_path = os.path.join(model_path, 'training_curves.png')
    plot_training_curves(train_losses, val_losses, val_accuracies, plot_path)
    
    print("Training complete!")
    print(f"Model and tokenizer saved to {model_path}")
    print(f"Training curves saved to {plot_path}")

if __name__ == "__main__":
    train() 