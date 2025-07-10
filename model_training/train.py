import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

# Adjust relative paths for imports since train.py is in model_training/
# Get the parent directory (project root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import from our modules using adjusted paths
# Files within the same directory (model_training)
from DiagnosisDateRelationModel import DiagnosisDateRelationModel 
from model_training.Vocabulary import Vocabulary
from ClinicalNoteDataset import ClinicalNoteDataset
import training_config 

# Files from other top-level directories
from utils.training_utils import load_and_prepare_data
from utils.training_utils import train_model, plot_training_curves
from config import DEVICE, MODEL_PATH, VOCAB_PATH

def train():
    print(f"Using device: {DEVICE}")
    
    # Ensure the training directory exists for outputs
    model_full_path = os.path.join(project_root, MODEL_PATH)
    vocab_full_path = os.path.join(project_root, VOCAB_PATH) # Use updated config path
    output_dir = os.path.dirname(model_full_path) 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Step 1: Load synthetic data from CSV
    synthetic_csv_path = os.path.join(project_root, 'data/synthetic.csv')
    
    if os.path.exists(synthetic_csv_path):
        print(f"Loading synthetic data from CSV: {synthetic_csv_path}")
        df = pd.read_csv(synthetic_csv_path)
        
        if 'note' not in df.columns:
            print("Error: CSV file does not contain a 'note' column")
            sys.exit(1)
            
        # Extract clinical notes from CSV
        clinical_notes = df['note'].tolist()
        print(f"Loaded {len(clinical_notes)} clinical notes from CSV")
        
        # Create dataset in the expected format for load_and_prepare_data
        dataset = []
        for i, row in df.iterrows():
            # Parse the gold_standard JSON from the CSV
            gold_standard = json.loads(row['gold_standard']) if 'gold_standard' in df.columns else []
            
            # Get pre-extracted disorders and dates
            extracted_disorders = row['extracted_disorders'] if 'extracted_disorders' in df.columns else []
            formatted_dates = row['formatted_dates'] if 'formatted_dates' in df.columns else []
            
            dataset.append({
                'clinical_note': row['note'],
                'ground_truth': gold_standard,
                'extracted_disorders': extracted_disorders,
                'formatted_dates': formatted_dates
            })
            
        print(f"Created dataset with {len(dataset)} entries")
    else:
        print(f"Error: Synthetic CSV not found at {synthetic_csv_path}")
        sys.exit(1)
    
    print(f"Training dataset contains {len(dataset)} clinical notes")
    
    # Step 2: Prepare data for model training
    print("Loading and preparing data...")
    # Use training_config settings here
    features, labels, vocab_instance = load_and_prepare_data(
        dataset, training_config.MAX_DISTANCE, Vocabulary
    )
    print(f"Loaded {len(features)} examples with vocabulary size {vocab_instance.n_words}")
    
    # Save vocabulary to the path defined in config (now model_training/vocab.pt)
    # Ensure the directory exists (it should as it's the same as output_dir)
    os.makedirs(os.path.dirname(vocab_full_path), exist_ok=True)
    torch.save(vocab_instance, vocab_full_path) 
    print(f"Saved vocabulary to {vocab_full_path}")
    
    # Check class balance
    if len(labels) > 0:
        positive = sum(labels)
        negative = len(labels) - positive
        print(f"Class distribution: {positive} positive examples ({positive/len(labels)*100:.1f}%), {negative} negative examples ({negative/len(labels)*100:.1f}%)")
    else:
        print("Warning: No examples found in the dataset!")
        sys.exit(1)  # Use sys.exit
    
    # Step 3: Create train / val / test datasets 
    # Note: These are splits within the training portion (80%) of the data
    train_features, val_features, train_labels, val_labels = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    print(f"Train: {len(train_features)}, Validation: {len(val_features)}")
    
    # Create datasets using training_config settings
    train_dataset = ClinicalNoteDataset(
        train_features, train_labels, vocab_instance, 
        training_config.MAX_CONTEXT_LEN, training_config.MAX_DISTANCE
    ) 
    val_dataset = ClinicalNoteDataset(
        val_features, val_labels, vocab_instance, 
        training_config.MAX_CONTEXT_LEN, training_config.MAX_DISTANCE
    )
    
    # Create data loaders using training_config batch size
    train_loader = DataLoader(train_dataset, batch_size=training_config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_config.BATCH_SIZE)
    
    # Step 4: Initialize and train model using training_config settings
    model = DiagnosisDateRelationModel(
        vocab_size=vocab_instance.n_words, 
        embedding_dim=training_config.EMBEDDING_DIM,
        hidden_dim=training_config.HIDDEN_DIM
    ).to(DEVICE)
    
    # Loss function and optimizer using training_config learning rate
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=training_config.LEARNING_RATE)
    
    # Train model using training_config epochs
    print("Training model...")
    train_losses, val_losses, val_accs = train_model(
        model, train_loader, val_loader, optimizer, criterion, 
        training_config.NUM_EPOCHS, DEVICE, model_full_path
    )
    
    # Plot training curves
    plot_save_path = os.path.join(output_dir, 'training_curves.png')
    plot_training_curves(train_losses, val_losses, val_accs, plot_save_path)
    
    print("Done!")

if __name__ == "__main__":
    train() 