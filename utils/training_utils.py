import json
import sys
import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import config # Import the config module

# Add parent directory to path to allow importing from models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Clean and preprocess text for model input
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Replace special characters
    text = re.sub(r'[^\w\s\.]', ' ', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Load dataset, process, and convert to the format needed
def load_and_prepare_data(file_or_dataset, MAX_DISTANCE, VocabClass=None):
    """
    Loads and prepares data for model training.
    
    Args:
        file_or_dataset: Either a file path to a JSON dataset or the dataset itself
        MAX_DISTANCE: Maximum distance (in characters) between diagnosis and date to include
        VocabClass: Optional vocabulary class to build vocabulary
        
    Returns:
        tuple: (features, labels, vocab) - processed data and vocabulary instance
    """
    # VocabClass is optional, only used if we need to build the vocab here
    if isinstance(file_or_dataset, str):
        # It's a file path
        with open(file_or_dataset, 'r') as f:
            dataset = json.load(f)
    else:
        # It's already a dataset
        dataset = file_or_dataset
    
    all_features = []
    all_labels = []
    vocab = VocabClass() if VocabClass else None
    
    # Add debug counters
    total_notes = 0
    notes_with_entities = 0
    total_examples = 0
    
    for entry in dataset:
        total_notes += 1
        clinical_note = entry['clinical_note']
        ground_truth = entry['ground_truth']
        
        # Use pre-extracted disorders and dates if available
        extracted_disorders = entry.get('extracted_disorders', [])
        formatted_dates = entry.get('formatted_dates', [])
        
        # Convert extracted disorders to the format we need
        diagnoses = []
        if extracted_disorders:
            for disorder in json.loads(extracted_disorders) if isinstance(extracted_disorders, str) else extracted_disorders:
                if isinstance(disorder, dict):
                    label = disorder.get('label', '')
                    start = disorder.get('start', 0)
                    diagnoses.append((label, start))
        
        # Convert formatted dates to the format we need
        dates = []
        if formatted_dates:
            for date in json.loads(formatted_dates) if isinstance(formatted_dates, str) else formatted_dates:
                if isinstance(date, dict):
                    parsed = date.get('parsed', '')
                    original = date.get('original', '')
                    start = date.get('start', 0)
                    dates.append((parsed, original, start))
        
        if diagnoses and dates:
            notes_with_entities += 1
        
        # Create a ground truth mapping
        gt_relations = {}
        for section in ground_truth:
            section_date = section['date']
            section_diagnoses = [diag['diagnosis'] for diag in section['diagnoses']]
            
            for diagnosis in section_diagnoses:
                # Ground truth keys use normalized (lowercase, underscore) diagnosis and YYYY-MM-DD date
                key = (diagnosis.strip().lower(), section_date.strip())
                gt_relations[key] = 1
        
        # Build features and labels using actual ground truth
        features = []
        labels = []
        
        # Dates list now contains (parsed_date, raw_date_str, date_pos)
        for diagnosis, diag_pos in diagnoses:
            for parsed_date, date_str, date_pos in dates:
                distance = abs(diag_pos - date_pos)
                
                # Skip if too far apart
                if distance > MAX_DISTANCE:
                    continue
                    
                # Process the context
                start_pos = max(0, min(diag_pos, date_pos) - 50)
                end_pos = min(len(clinical_note), max(diag_pos, date_pos) + 100)
                context = clinical_note[start_pos:end_pos]
                context = preprocess_text(context)
                
                # Add to vocab if building one
                if vocab:
                    vocab.add_sentence(context)
                
                # Create feature (using raw date_str for context)
                feature = {
                    'diagnosis': diagnosis, 
                    'date': date_str, # Keep original raw date string for feature context
                    'context': context,
                    'distance': distance,
                    'diag_pos_rel': diag_pos - start_pos,
                    'date_pos_rel': date_pos - start_pos,
                    'diag_before_date': 1 if diag_pos < date_pos else 0
                }
                
                # Look up in ground truth relations using normalized diagnosis and parsed date
                key = (diagnosis.strip().lower(), parsed_date) # Use the parsed_date here
                label = 1 if key in gt_relations else 0
                
                features.append(feature)
                labels.append(label)
                total_examples += 1
        
        all_features.extend(features)
        all_labels.extend(labels)
    
    # Print debug info
    print(f"Processed {total_notes} clinical notes")
    print(f"Notes with entities: {notes_with_entities}/{total_notes}")
    print(f"Total examples generated: {total_examples}")
    
    # Return vocab instance only if it was created
    return all_features, all_labels, vocab if VocabClass else None

# Training function
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device, model_save_path):
    # model_save_path should be the full path from config
    train_losses = []
    val_losses = []
    val_accs = []
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            # Move tensors to device
            context = batch['context'].to(device)
            distance = batch['distance'].to(device)
            diag_before = batch['diag_before'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(context, distance, diag_before)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move tensors to device
                context = batch['context'].to(device)
                distance = batch['distance'].to(device)
                diag_before = batch['diag_before'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = model(context, distance, diag_before)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        val_acc = 100 * correct / total
        val_accs.append(val_acc)
        
        # Save the best model to the specified path
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Ensure directory exists before saving
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return train_losses, val_losses, val_accs

# Plot training progress
def plot_training_curves(train_losses, val_losses, val_accs, save_path):
    # Get the directory path and create a proper image path
    dir_path = os.path.dirname(save_path)
    base_name = os.path.splitext(os.path.basename(save_path))[0]
    plot_path = os.path.join(dir_path, f"{base_name}_training_curves.png")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
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
    os.makedirs(dir_path, exist_ok=True)
    plt.savefig(plot_path)
    plt.show()
    
    print(f"Training curves saved to {plot_path}")