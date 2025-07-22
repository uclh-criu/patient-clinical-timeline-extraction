import os
import sys
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from transformers import get_linear_schedule_with_warmup

# Adjust relative paths for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import configuration
import config
import model_training.training_config as training_config
from model_training.BertEntityPairDataset import BertEntityPairDataset

# Import utility functions
from utils.training_utils import prepare_bert_training_data, train_bert_model, plot_training_curves

def main():
    """
    Main function to train the BERT model for entity-date relationship extraction.
    """
    print(f"Using device: {training_config.DEVICE}")
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Get training data path from training_config
    data_path = os.path.join(project_root, training_config.BERT_TRAINING_DATA_PATH)
    print(f"Using training data from: {data_path}")
    
    # Ensure the data file exists
    if not os.path.exists(data_path):
        print(f"Error: Training data file not found at {data_path}")
        return

    # Ensure model directory exists
    model_path = os.path.join(project_root, config.BERT_MODEL_PATH)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Prepare data
    train_dataset, val_dataset, tokenizer = prepare_bert_training_data(
        data_path, 
        training_config.BERT_PRETRAINED_MODEL,
        training_config.BERT_MAX_SEQ_LENGTH
    )
    
    # Check if we have training data
    if train_dataset is None or len(train_dataset) == 0:
        print("ERROR: No training examples were created. Check your data format.")
        return
        
    print(f"Created {len(train_dataset)} training examples and {len(val_dataset) if val_dataset else 0} validation examples")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.BERT_BATCH_SIZE,
        shuffle=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config.BERT_BATCH_SIZE,
            shuffle=False
        )
    
    # Initialize model
    print(f"Loading pre-trained model: {training_config.BERT_PRETRAINED_MODEL}")
    model = AutoModelForSequenceClassification.from_pretrained(
        training_config.BERT_PRETRAINED_MODEL,
        num_labels=1  # Binary classification
    )
    
    # Add special tokens for entity marking
    special_tokens = {'additional_special_tokens': ['[E1]', '[/E1]', '[E2]', '[/E2]']}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # Move model to device
    model.to(training_config.DEVICE)
    
    # Set up optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=training_config.BERT_LEARNING_RATE
    )
    
    # Calculate total training steps
    total_steps = len(train_loader) * training_config.BERT_NUM_TRAIN_EPOCHS
    
    # Create scheduler with linear warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),  # 10% of total steps for warmup
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
        training_config.DEVICE,
        model_path
    )
    
    # Plot training curves
    curves_path = os.path.join(os.path.dirname(model_path), "bert_training_curves.png")
    plot_training_curves(train_losses, val_losses, val_accuracies, curves_path)
    
    print(f"Training completed. Model saved to {model_path}")
    print(f"Training curves saved to {curves_path}")

if __name__ == "__main__":
    main() 