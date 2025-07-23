import json
import sys
import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import config # Import the config module
import ast  # Add ast module for literal_eval

# Add parent directory to path to allow importing from models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import preprocess_text from extraction_utils to avoid duplication
from utils.extraction_utils import preprocess_text

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

# This function has been replaced by prepare_custom_training_data
# which uses the canonical load_and_prepare_data from extraction_utils.py
def load_and_prepare_data(file_or_dataset, MAX_DISTANCE, VocabClass=None):
    """
    DEPRECATED: Use prepare_custom_training_data instead.
    This function is kept for backward compatibility but will be removed in the future.
    """
    return prepare_custom_training_data(file_or_dataset, MAX_DISTANCE, VocabClass)

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
def plot_training_curves(train_losses, val_losses, val_accs, save_path, show_plot=False):
    """
    Plot training curves for model training.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        val_accs: List of validation accuracies
        save_path: Path to save the plot
        show_plot: Whether to display the plot (default: False)
    """
        # Check if save_path needs to be modified to include _training_curves.png
    if not save_path.endswith('.png'):
        dir_path = os.path.dirname(save_path)
        base_name = os.path.splitext(os.path.basename(save_path))[0]
        save_path = os.path.join(dir_path, f"{base_name}_training_curves.png")
    
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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    print(f"Training curves saved to {save_path}")

# --- BERT-specific functions --- #

def prepare_bert_training_data(csv_path, pretrained_model_name, max_seq_length):
    """
    Prepare data for training the BERT model.
    
    Args:
        csv_path: Path to the CSV dataset
        pretrained_model_name: Name or path of the pretrained BERT model
        max_seq_length: Maximum sequence length for tokenization
        
    Returns:
        tuple: (train_dataset, val_dataset, tokenizer) - datasets and tokenizer
    """
    import os
    from model_training.BertEntityPairDataset import BertEntityPairDataset
    
    # Import the canonical load_and_prepare_data function
    from utils.extraction_utils import load_and_prepare_data
    
    # Ensure config has the necessary attributes for BERT training
    if not hasattr(config, 'RELATIONSHIP_GOLD_COLUMN'):
        setattr(config, 'RELATIONSHIP_GOLD_COLUMN', 'relationship_gold')
    if not hasattr(config, 'REAL_DATA_DATES_COLUMN'):
        setattr(config, 'REAL_DATA_DATES_COLUMN', 'formatted_dates')
    if not hasattr(config, 'ENTITY_MODE'):
        setattr(config, 'ENTITY_MODE', 'disorder_only')
    if not hasattr(config, 'ENABLE_RELATIVE_DATE_EXTRACTION'):
        setattr(config, 'ENABLE_RELATIVE_DATE_EXTRACTION', False)
    
    # Load the data using the canonical function
    # Handle different return signatures based on ENTITY_MODE
    result = load_and_prepare_data(csv_path, None, config)
    
    # In disorder_only mode, load_and_prepare_data returns only 2 values
    # In multi_entity mode, it returns 4 values
    if len(result) == 2:
        prepared_data, relationship_gold = result
    else:
        prepared_data, _, relationship_gold, _ = result
    
    if not prepared_data:
        print("Error: Failed to load data for BERT training.")
        return None, None, None
    
    # Initialize tokenizer
    print(f"Loading tokenizer: {pretrained_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    
    # Add special tokens for entity marking
    special_tokens = {'additional_special_tokens': ['[E1]', '[/E1]', '[E2]', '[/E2]']}
    tokenizer.add_special_tokens(special_tokens)
    
    # Create a mapping of gold standard relationships for quick lookup
    gold_relationships = set()
    for rel in relationship_gold:
        # Handle both disorder_only and multi_entity modes
        if 'diagnosis' in rel:
            gold_relationships.add((rel['note_id'], rel['diagnosis'].lower(), rel['date']))
        elif 'entity_label' in rel:
            gold_relationships.add((rel['note_id'], rel['entity_label'].lower(), rel['date']))
    
    print(f"Found {len(gold_relationships)} gold standard relationships")
    
    # Create entity pairs for BERT training
    entity_pairs = []
    
    for note_entry in prepared_data:
        note_id = note_entry['note_id']
        text = note_entry['note']
        entities_list, dates = note_entry['entities']
        
        # Skip if either entities or dates are empty
        if not entities_list or not dates:
            continue
        
        # Process each entity-date pair
        for entity in entities_list:
            # Handle both dict and tuple formats for entities
            if isinstance(entity, dict):
                entity_label = entity.get('label', '')
                entity_start = entity.get('start', 0)
                entity_end = entity.get('end', entity_start + len(entity_label))
                entity_category = entity.get('category', 'disorder')
            else:
                # Legacy format: (label, position)
                entity_label, entity_start = entity
                entity_end = entity_start + len(entity_label)
                entity_category = 'disorder'  # Default category for legacy format
            
            for date_tuple in dates:
                # Unpack the date tuple: (parsed_date, raw_date_str, position)
                parsed_date, date_str, date_start = date_tuple
                date_end = date_start + len(date_str)
                
                # Create entity dictionaries
                entity_dict = {
                    'text': entity_label,
                    'start': entity_start,
                    'end': entity_end,
                    'category': entity_category
                }
                
                date_dict = {
                    'text': date_str,
                    'start': date_start,
                    'end': date_end,
                    'parsed': parsed_date
                }
                
                # Check if this is a gold standard relationship
                is_gold = False
                if (note_id, entity_label.lower(), parsed_date) in gold_relationships:
                    is_gold = True
                
                # Add to entity pairs
                entity_pairs.append({
                    'text': text,
                    'entity1': entity_dict,
                    'entity2': date_dict,
                    'label': 1 if is_gold else 0
                })
    
    print(f"Created {len(entity_pairs)} entity pairs for training")
    
    # Check class balance
    positive = sum(1 for pair in entity_pairs if pair['label'] == 1)
    negative = len(entity_pairs) - positive
    print(f"Class distribution: {positive} positive examples ({positive/len(entity_pairs)*100:.1f}%), "
          f"{negative} negative examples ({negative/len(entity_pairs)*100:.1f}%)")
    
    # Balance the dataset if needed (downsample negative examples)
    if negative > 5 * positive:  # If negative examples are more than 5x positive ones
        import random
        print("Downsampling negative examples to balance the dataset...")
        positive_pairs = [pair for pair in entity_pairs if pair['label'] == 1]
        negative_pairs = [pair for pair in entity_pairs if pair['label'] == 0]
        
        # Keep all positive examples and randomly sample negative examples
        sampled_negative = random.sample(negative_pairs, min(len(negative_pairs), 5 * len(positive_pairs)))
        entity_pairs = positive_pairs + sampled_negative
        
        # Shuffle the pairs
        random.shuffle(entity_pairs)
        
        # Print new class balance
        positive = sum(1 for pair in entity_pairs if pair['label'] == 1)
        negative = len(entity_pairs) - positive
        print(f"New class distribution: {positive} positive examples ({positive/len(entity_pairs)*100:.1f}%), "
              f"{negative} negative examples ({negative/len(entity_pairs)*100:.1f}%)")
    
    # Create dataset
    dataset = BertEntityPairDataset.create_from_pairs(entity_pairs, tokenizer, max_seq_length)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"Split into {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
    
    return train_dataset, val_dataset, tokenizer

def train_bert_model(model, tokenizer, train_loader, val_loader, optimizer, scheduler, num_epochs, device, model_path):
    """
    Train the BERT model for entity-date relationship extraction.
    
    Args:
        model: The BERT model
        tokenizer: The BERT tokenizer to save alongside the model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        device: Device to train on
        model_path: Path to save the model
        
    Returns:
        tuple: (train_losses, val_losses, val_accuracies)
    """
    from tqdm import tqdm
    import numpy as np
    import os
    
    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Initialize best validation accuracy
    best_val_accuracy = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        total_train_loss = 0
        train_steps = 0
        
        # Use tqdm for progress bar
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(**{k: v for k, v in batch.items() if k != 'label'})
            logits = outputs.logits
            
            # Calculate loss
            loss = torch.nn.BCEWithLogitsLoss()(logits.view(-1), batch['label'])
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            total_train_loss += loss.item()
            train_steps += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate average training loss
        avg_train_loss = total_train_loss / train_steps
        train_losses.append(avg_train_loss)
        
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        if val_loader:
            model.eval()
            total_val_loss = 0
            val_steps = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    # Move batch to device
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = model(**{k: v for k, v in batch.items() if k != 'label'})
                    logits = outputs.logits
                    
                    # Calculate loss
                    loss = torch.nn.BCEWithLogitsLoss()(logits.view(-1), batch['label'])
                    
                    # Update metrics
                    total_val_loss += loss.item()
                    val_steps += 1
                    
                    # Calculate accuracy
                    predictions = (torch.sigmoid(logits.view(-1)) > 0.5).float()
                    correct += (predictions == batch['label']).sum().item()
                    total += len(batch['label'])
            
            # Calculate average validation loss and accuracy
            avg_val_loss = total_val_loss / val_steps
            val_losses.append(avg_val_loss)
            
            val_accuracy = correct / total * 100
            val_accuracies.append(val_accuracy)
            
            print(f"Validation loss: {avg_val_loss:.4f}")
            print(f"Validation accuracy: {val_accuracy:.2f}%")
            
            # Save the best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                print(f"New best validation accuracy: {best_val_accuracy:.4f}")
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                
                # Save model and tokenizer
                model.save_pretrained(model_path)
                tokenizer.save_pretrained(model_path)
                print(f"Model and tokenizer saved to {model_path}")
    
    return train_losses, val_losses, val_accuracies

# plot_bert_training_curves has been consolidated with plot_training_curves

def prepare_custom_training_data(dataset_path_or_data, max_distance, vocab_class=None):
    """
    Prepare data for training the custom model.
    
    Args:
        dataset_path_or_data: Either a file path to a CSV dataset or the dataset itself
        max_distance: Maximum distance (in characters) between diagnosis and date to include
        vocab_class: Optional vocabulary class to build vocabulary
        
    Returns:
        tuple: (features, labels, vocab) - processed data and vocabulary instance
    """
    from utils.extraction_utils import load_and_prepare_data as canonical_load
    from utils.extraction_utils import preprocess_note_for_prediction, transform_python_to_json
    
    # Initialize vocabulary if needed
    vocab = vocab_class() if vocab_class else None
    
    if isinstance(dataset_path_or_data, str):
        # It's a file path, use the canonical function to load it
        print(f"Loading data from file: {dataset_path_or_data}")
        
        # Make sure we have the necessary config attributes for training
        if not hasattr(config, 'RELATIONSHIP_GOLD_COLUMN'):
            setattr(config, 'RELATIONSHIP_GOLD_COLUMN', 'relationship_gold')
        if not hasattr(config, 'REAL_DATA_DATES_COLUMN'):
            setattr(config, 'REAL_DATA_DATES_COLUMN', 'formatted_dates')
        if not hasattr(config, 'ENTITY_MODE'):
            setattr(config, 'ENTITY_MODE', 'disorder_only')
        if not hasattr(config, 'ENABLE_RELATIVE_DATE_EXTRACTION'):
            setattr(config, 'ENABLE_RELATIVE_DATE_EXTRACTION', False)
            
        # Handle different return signatures based on ENTITY_MODE
        result = canonical_load(dataset_path_or_data, None, config)
        
        # In disorder_only mode, load_and_prepare_data returns only 2 values
        # In multi_entity mode, it returns 4 values
        if len(result) == 2:
            prepared_data, relationship_gold = result
            entity_gold = None
            pa_likelihood_gold = None
        else:
            prepared_data, entity_gold, relationship_gold, pa_likelihood_gold = result
        
        if not prepared_data:
            print("Error: Failed to load data for training.")
            return [], [], vocab
    else:
        # It's already a dataset, process it directly
        print("Processing in-memory dataset")
        prepared_data = []
        relationship_gold = []
        
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
                            'entity_category': rel.get('entity_category', 'disorder'),
                            'date': rel['date']
                        })
                    elif 'diagnosis' in rel and 'date' in rel:
                        # Legacy format
                        relationship_gold.append({
                            'note_id': i,
                            'patient_id': entry.get('patient_id'),
                            'entity_label': rel['diagnosis'].lower(),
                            'entity_category': 'disorder',  # Default for legacy format
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
    for rel in relationship_gold:
        gold_relationships.add((rel['entity_label'].lower(), rel['date']))
    
    # Print detailed diagnostic information about the data
    print("\n===== DETAILED DATA DIAGNOSTICS =====")
    print(f"Found {len(gold_relationships)} gold standard relationships")
    print(f"Found {len(prepared_data)} notes with entity data")
    
    # Show sample of the relationship_gold data
    print("\nSample of raw relationship_gold data:")
    if relationship_gold and len(relationship_gold) > 0:
        for i, rel in enumerate(relationship_gold[:3]):
            print(f"  {i+1}. {rel}")
    
    # Show sample of the parsed entities
    print("\nSample of parsed entities:")
    if prepared_data and len(prepared_data) > 0:
        sample_note = prepared_data[0]
        disorders, dates = sample_note['entities']
        
        print(f"  Disorders ({len(disorders)}):")
        for i, disorder in enumerate(disorders[:5]):
            print(f"    {i+1}. {disorder}")
            
        print(f"  Dates ({len(dates)}):")
        for i, date in enumerate(dates[:5]):
            print(f"    {i+1}. {date}")
    else:
        print("  No entities found in prepared data")
    
    # Show sample of gold relationships
    if gold_relationships:
        print("\nSample gold relationships (first 5):")
        for i, (entity, date) in enumerate(list(gold_relationships)[:5]):
            print(f"  {i+1}. Entity: '{entity}', Date: '{date}'")
    else:
        print("\nWARNING: No gold standard relationships found. Check your data format.")
    
    # Generate features and labels
    all_features = []
    all_labels = []
    total_examples = 0
    total_positive = 0
    
    for note_entry in prepared_data:
        note_text = note_entry['note']
        disorders, dates = note_entry['entities']
        
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
            
            all_features.append(feature)
            all_labels.append(label)
            total_examples += 1
            if label == 1:
                total_positive += 1
    
    # Print debug info
    print(f"Generated {total_examples} examples")
    print(f"Positive examples: {total_positive}/{total_examples} ({total_positive/max(1,total_examples)*100:.1f}%)")
    
    return all_features, all_labels, vocab