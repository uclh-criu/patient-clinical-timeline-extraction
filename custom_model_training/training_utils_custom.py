import json
import sys
import os
import re
import torch
import matplotlib.pyplot as plt
import config # Import the config module
import ast  # Add ast module for literal_eval
import csv
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score

# Add parent directory to path to allow importing from models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def find_best_f1_threshold(probs, labels, threshold_step=0.01):
    """
    Find the best F1 score by sweeping through different thresholds.
    
    Args:
        probs (np.array): Array of probabilities from the model.
        labels (np.array): Array of true labels.
        threshold_step (float): The step size for sweeping thresholds.
        
    Returns:
        tuple: (best_f1, best_precision, best_recall, best_accuracy, best_threshold)
    """
    best_f1 = 0
    best_threshold = 0.5  # Default threshold
    best_precision = 0
    best_recall = 0
    best_accuracy = 0
    
    # Sweep through a range of thresholds
    for threshold in np.arange(0.1, 0.9, threshold_step):
        preds = (probs >= threshold).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision_score(labels, preds, zero_division=0)
            best_recall = recall_score(labels, preds, zero_division=0)
            best_accuracy = np.mean(preds == labels) * 100
    
    return best_f1, best_precision, best_recall, best_accuracy, best_threshold

# Add a helper function to sanitize strings with datetime objects
def sanitize_datetime_strings(s):
    """
    Replace datetime.date objects in string with ISO format date strings
    Example: datetime.date(2019, 4, 18) -> "2019-04-18"
    """
    if not isinstance(s, str):
        return s
        
    # Pattern to match datetime.date(YYYY, MM, DD)
    pattern = r'datetime\.date\((\d+),\s*(\d+),\s*(\d+)\)'
    
    # Replace with "YYYY-MM-DD" format
    def replace_date(match):
        year = match.group(1)
        month = match.group(2).zfill(2)  # Ensure 2 digits
        day = match.group(3).zfill(2)    # Ensure 2 digits
        return f'"{year}-{month}-{day}"'
    
    return re.sub(pattern, replace_date, s)

def prepare_custom_training_data(dataset_path_or_data, max_distance, vocab_class=None, data_split_mode='all', 
                                prepared_data=None, relationship_gold=None):
    """
    Prepare data for training the custom model.
    
    Args:
        dataset_path_or_data: Either a file path to a CSV dataset or the dataset itself
        max_distance: Maximum distance (in characters) between diagnosis and date to include
        vocab_class: Optional vocabulary class to build vocabulary
        data_split_mode (str): How to split the data. Options:
            - 'train': Use only the training portion (first TRAINING_SET_RATIO)
            - 'test': Use only the testing portion (remaining 1-TRAINING_SET_RATIO)
            - 'all': Use all data without splitting (default)
        prepared_data: Pre-loaded prepared data (to avoid loading from file again)
        relationship_gold: Pre-loaded relationship gold data (to avoid loading from file again)
        
    Returns:
        tuple: (features, labels, vocab) - processed data and vocabulary instance
    """
    from utils.inference_eval_utils import preprocess_note_for_prediction, transform_python_to_json
    
    # Initialize vocabulary if needed
    vocab = vocab_class() if vocab_class else None
    
    # If prepared_data and relationship_gold are provided, use them directly
    if prepared_data is not None and relationship_gold is not None:
        print("Using pre-loaded data (skipping file loading)")
    elif isinstance(dataset_path_or_data, str):
        # It's a file path, use the canonical function to load it
        print(f"Loading data from file: {dataset_path_or_data} with split mode: {data_split_mode}")
        
        from utils.inference_eval_utils import load_and_prepare_data as canonical_load
        
        # Make sure we have the necessary config attributes for training
        if not hasattr(config, 'RELATIONSHIP_GOLD_COLUMN'):
            setattr(config, 'RELATIONSHIP_GOLD_COLUMN', 'relationship_gold')
        if not hasattr(config, 'DATES_COLUMN'):
            setattr(config, 'DATES_COLUMN', 'formatted_dates')
        
        # Check if we have a custom ENTITY_MODE in training_config_custom
        try:
            # Try to import training_config_custom
            import custom_model_training.training_config_custom as training_config_custom
            if hasattr(training_config_custom, 'ENTITY_MODE'):
                # Use the ENTITY_MODE from training_config_custom
                entity_mode = training_config_custom.ENTITY_MODE
                print(f"Using ENTITY_MODE from training_config_custom: {entity_mode}")
                setattr(config, 'ENTITY_MODE', entity_mode)
            else:
                # If not defined in training_config_custom, use the default or existing value
                if not hasattr(config, 'ENTITY_MODE'):
                    setattr(config, 'ENTITY_MODE', 'diagnosis_only')
        except ImportError:
            # If training_config_custom can't be imported, use the default or existing value
            if not hasattr(config, 'ENTITY_MODE'):
                setattr(config, 'ENTITY_MODE', 'diagnosis_only')
        
        if not hasattr(config, 'ENABLE_RELATIVE_DATE_EXTRACTION'):
            setattr(config, 'ENABLE_RELATIVE_DATE_EXTRACTION', False)
        
        # Handle different return signatures based on ENTITY_MODE
        try:
            result = canonical_load(dataset_path_or_data, None, config, data_split_mode=data_split_mode)
            
            # In disorder_only mode, load_and_prepare_data returns only 2 values
            # In multi_entity mode, it returns 4 values
            if len(result) == 2:
                prepared_data, relationship_gold = result
                entity_gold = None
                pa_likelihood_gold = None
            else:
                prepared_data, entity_gold, relationship_gold, pa_likelihood_gold = result
            
            # Check if prepared_data is empty (which would cause ZeroDivisionError)
            if not prepared_data:
                print("Error: No data returned from load_and_prepare_data. This could be due to an empty dataset or an issue with the data split.")
                print("Please check your dataset and ensure it contains enough records for the requested split.")
                return [], [], vocab
                
        except ZeroDivisionError:
            print("Error: ZeroDivisionError occurred during data loading. This is likely due to an empty dataset after splitting.")
            print("Please use a larger dataset or set data_split_mode='all' to use all available data.")
            return [], [], vocab
        except Exception as e:
            print(f"Error loading data: {e}")
            return [], [], vocab
    else:
        # It's already a dataset, process it directly
        print("Processing in-memory dataset")
        prepared_data = []
        relationship_gold = []
        entity_gold = None
        
        for i, entry in enumerate(dataset_path_or_data):
            clinical_note = entry['clinical_note']
            
            # Get relationship_gold data and convert to canonical format
            gold_data = entry.get('relationship_gold', [])
            if isinstance(gold_data, str):
                # Find the first '[' which should be the start of the JSON array
                json_start = gold_data.find('[')
                if json_start >= 0:
                    gold_data = gold_data[json_start:]
                
                # Parse the JSON string
                try:
                    # Sanitize and parse gold_data if it's a string
                    sanitized_gold = sanitize_datetime_strings(gold_data)
                    gold_data = json.loads(sanitized_gold)
                except json.JSONDecodeError:
                    try:
                        gold_data = ast.literal_eval(sanitized_gold)
                    except (ValueError, SyntaxError):
                        gold_data = []
            
            # Process gold data into canonical format
            for rel in gold_data:
                if isinstance(rel, dict):
                    if 'entity_label' in rel and 'date' in rel:
                        relationship_gold.append({
                            'note_id': i,
                            'patient_id': entry.get('patient_id'),
                            'entity_label': rel['entity_label'].lower(),
                            'entity_category': rel.get('entity_category', 'diagnosis'),
                            'date': rel['date']
                        })
                    elif 'diagnosis' in rel and 'date' in rel:
                        # Legacy format
                        relationship_gold.append({
                            'note_id': i,
                            'patient_id': entry.get('patient_id'),
                            'entity_label': rel['diagnosis'].lower(),
                            'entity_category': 'diagnosis',  # Default for legacy format
                            'date': rel['date']
                        })
            
            # Process extracted entities
            extracted_disorders = entry.get('extracted_disorders', [])
            formatted_dates = entry.get('formatted_dates', [])
            
            # Convert to the format expected by preprocess_note_for_prediction
            disorders = []
            dates = []
            
            # Process disorders
            if extracted_disorders:
                try:
                    # Handle different formats
                    if isinstance(extracted_disorders, list):
                        disorders_list = extracted_disorders
                    elif isinstance(extracted_disorders, str) and extracted_disorders.strip():
                        disorders_json = transform_python_to_json(extracted_disorders)
                        disorders_list = json.loads(disorders_json)
                    else:
                        disorders_list = []
                        
                    for disorder in disorders_list:
                        if isinstance(disorder, dict):
                            label = disorder.get('label', '')
                            start = disorder.get('start', 0)
                            disorders.append((label, start))
                except Exception as e:
                    print(f"Warning: Could not parse extracted_disorders: {e}")
            
            # Process dates
            if formatted_dates:
                try:
                    # Handle different formats
                    if isinstance(formatted_dates, list):
                        dates_list = formatted_dates
                    elif isinstance(formatted_dates, str) and formatted_dates.strip():
                        dates_json = transform_python_to_json(formatted_dates)
                        dates_list = json.loads(dates_json)
                    else:
                        dates_list = []
                        
                    for date in dates_list:
                        if isinstance(date, dict):
                            parsed = date.get('parsed', '')
                            original = date.get('original', '')
                            start = date.get('start', 0)
                            dates.append((parsed, original, start))
                except Exception as e:
                    print(f"Warning: Could not parse formatted_dates: {e}")
            
            # Add to prepared data
            prepared_data.append({
                'patient_id': entry.get('patient_id'),
                'note_id': i,
                'note': clinical_note,
                'entities': (disorders, dates)
            })
    
    # Create a set of gold standard relationships for quick lookup
    gold_relationships = set()
    entity_categories = {}  # Map of (entity_label, date) -> entity_category
    
    for rel in relationship_gold:
        entity_label = rel['entity_label'].lower()
        date = rel['date']
        gold_relationships.add((entity_label, date))
        
        # Store the entity category for each (entity_label, date) pair
        entity_categories[(entity_label, date)] = rel.get('entity_category', 'diagnosis').lower()
    
    # Print detailed diagnostic information about the data
    print("\n===== GENERATING FEATURES FROM PREPARED DATA =====")
    print(f"Found {len(gold_relationships)} gold standard relationships")
    print(f"Found {len(prepared_data)} notes with entity data")
    
    # Determine if we're in multi-entity mode
    entity_mode = getattr(config, 'ENTITY_MODE', 'diagnosis_only')
    is_multi_entity = entity_mode == 'multi_entity'
    print(f"Using entity mode: {entity_mode}")
    
    # Generate features and labels
    all_features = []
    all_labels = []
    total_examples = 0
    total_positive = 0
    
    for note_entry in prepared_data:
        note_text = note_entry['note']
        disorders, dates = note_entry['entities']
        patient_id = note_entry.get('patient_id')  # Get patient_id from note entry
        note_id = note_entry.get('note_id')
        
        # Generate features using preprocess_note_for_prediction
        note_features = preprocess_note_for_prediction(note_text, disorders, dates, max_distance)
        
        # Assign labels based on gold relationships
        for feature in note_features:
            diagnosis = feature['diagnosis']
            parsed_date = feature['date']
            context = feature['context']
            
            # Add to vocab if building one
            if vocab:
                vocab.add_sentence(context)
            
            # Look up in gold relationships
            key = (diagnosis.strip().lower(), parsed_date)
            label = 1 if key in gold_relationships else 0
            
            # Add entity category information for multi-entity mode
            if is_multi_entity and key in entity_categories:
                feature['entity_category'] = entity_categories[key]
            else:
                feature['entity_category'] = 'diagnosis'  # Default
            
            # Add patient_id and note_id to feature for patient-aware splitting
            feature['patient_id'] = patient_id
            feature['note_id'] = note_id
            
            all_features.append(feature)
            all_labels.append(label)
            total_examples += 1
            if label == 1:
                total_positive += 1
    
    # Print debug info
    print(f"Generated {total_examples} examples")
    print(f"Positive examples: {total_positive}/{total_examples} ({total_positive/max(1,total_examples)*100:.1f}%)")
    
    # Count examples by entity category
    if is_multi_entity:
        category_counts = {}
        for feature in all_features:
            category = feature.get('entity_category', 'diagnosis')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        print("\nExamples by entity category:")
        for category, count in category_counts.items():
            print(f"  {category}: {count} examples ({count/len(all_features)*100:.1f}%)")
    
    return all_features, all_labels, vocab

# Training function
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device, model_save_path=None):
    """
    Train the model and track the best version based on validation F1-score.
    
    Args:
        model: The PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: The optimizer to use
        criterion: The loss function
        epochs: Number of epochs to train for
        device: Device to train on (cpu or cuda)
        model_save_path: Path to save the best model (not used anymore, kept for compatibility)
        
    Returns:
        tuple: (best_model_state, metrics) - The state dict of the best model and metrics dictionary
    """
    train_losses = []
    val_losses = []
    val_accs = []
    val_f1s = []
    val_precisions = []
    val_recalls = []
    val_thresholds = []  # Store the best threshold for each epoch
    best_val_f1 = 0
    best_model_state = None  # Store the best model state in memory
    
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
            
            # Handle entity category if available (for multi-entity mode)
            entity_category = batch.get('entity_category')
            if entity_category is not None:
                entity_category = entity_category.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(context, distance, diag_before, entity_category)
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
        all_outputs = []  # Changed from all_preds to store raw outputs
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move tensors to device
                context = batch['context'].to(device)
                distance = batch['distance'].to(device)
                diag_before = batch['diag_before'].to(device)
                labels = batch['label'].to(device)
                
                # Handle entity category if available (for multi-entity mode)
                entity_category = batch.get('entity_category')
                if entity_category is not None:
                    entity_category = entity_category.to(device)
                
                # Forward pass
                outputs = model(context, distance, diag_before, entity_category)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                # Store raw outputs and labels for threshold sweeping
                # If using BCEWithLogitsLoss, model outputs are logits. Apply sigmoid to get probabilities.
                if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                    probs = torch.sigmoid(outputs)
                else:
                    probs = outputs  # Assume model already outputs probabilities
                
                all_outputs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate metrics with threshold sweeping
        all_outputs = np.array(all_outputs)
        all_labels = np.array(all_labels)
        
        # Find the best threshold and corresponding metrics
        val_f1, val_precision, val_recall, val_acc, best_threshold = find_best_f1_threshold(all_outputs, all_labels)
        
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_thresholds.append(best_threshold)
        
        # Store the best model state in memory based on F1-score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()  # Create a copy of the state dict
            print(f"New best validation F1-score found: {val_f1:.4f} (at threshold {best_threshold:.2f}) (P: {val_precision:.4f}, R: {val_recall:.4f}, Acc: {val_acc:.2f}%)")
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f} (Thresh: {best_threshold:.2f}), Val Acc: {val_acc:.2f}%')
    
    # Find the index of the best F1 score
    best_epoch_idx = val_f1s.index(best_val_f1)
    
    # Return metrics as a dictionary for logging
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'val_f1s': val_f1s,
        'val_precisions': val_precisions,
        'val_recalls': val_recalls,
        'val_thresholds': val_thresholds,
        'best_val_f1': best_val_f1,
        'best_val_acc': val_accs[best_epoch_idx],
        'best_val_precision': val_precisions[best_epoch_idx],
        'best_val_recall': val_recalls[best_epoch_idx],
        'best_val_threshold': val_thresholds[best_epoch_idx],
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'final_val_acc': val_accs[-1],
        'final_val_f1': val_f1s[-1],
        'final_val_precision': val_precisions[-1],
        'final_val_recall': val_recalls[-1],
        'final_val_threshold': val_thresholds[-1],
        'epochs': epochs,
        'train_loss': train_losses[-1],  # For consistency with log_training_run
        'val_loss': val_losses[-1],      # For consistency with log_training_run
    }
    
    return best_model_state, metrics