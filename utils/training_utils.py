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

# Clean and preprocess text for model input
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Replace special characters
    text = re.sub(r'[^\w\s\.]', ' ', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

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
    total_positive = 0
    
    print("\n===== DIAGNOSTICS FOR LABEL ASSIGNMENT =====")
    print("This will help identify why we're not getting positive examples")
    
    for entry in dataset:
        total_notes += 1
        clinical_note = entry['clinical_note']
        
        # Get relationship_gold data
        relationship_gold = entry.get('relationship_gold', [])
        
        # If it's a string, try to extract the JSON part
        if isinstance(relationship_gold, str):
            # Find the first '[' which should be the start of the JSON array
            json_start = relationship_gold.find('[')
            if json_start >= 0:
                relationship_gold = relationship_gold[json_start:]
                print(f"\n--- CLINICAL NOTE #{total_notes} ---")
                print(f"Extracted JSON from relationship_gold: {relationship_gold[:50]}...")
            
            # Parse the JSON string
            try:
                # Sanitize and parse relationship_gold if it's a string
                sanitized_gold = sanitize_datetime_strings(relationship_gold)
                relationship_gold = json.loads(sanitized_gold)
            except json.JSONDecodeError:
                try:
                    relationship_gold = ast.literal_eval(sanitized_gold)
                except (ValueError, SyntaxError):
                    # If parsing fails, use an empty list as fallback
                    print(f"Warning: Could not parse relationship_gold: {relationship_gold[:100]}...")
                    relationship_gold = []
        
        # Use pre-extracted disorders and dates if available
        extracted_disorders = entry.get('extracted_disorders', [])
        formatted_dates = entry.get('formatted_dates', [])
        
        # Convert extracted disorders to the format we need
        diagnoses = []
        if extracted_disorders:
            try:
                # First try to parse as a list directly
                if not isinstance(extracted_disorders, str):
                    disorders_list = extracted_disorders
                # Then try ast.literal_eval which handles both single and double quotes
                elif extracted_disorders.strip():
                    disorders_list = ast.literal_eval(extracted_disorders)
                else:
                    disorders_list = []
                    
                for disorder in disorders_list:
                    if isinstance(disorder, dict):
                        label = disorder.get('label', '')
                        start = disorder.get('start', 0)
                        diagnoses.append((label, start))
            except (ValueError, SyntaxError) as e:
                # If that fails, try json.loads as a fallback
                try:
                    if isinstance(extracted_disorders, str):
                        disorders_list = json.loads(extracted_disorders)
                        for disorder in disorders_list:
                            if isinstance(disorder, dict):
                                label = disorder.get('label', '')
                                start = disorder.get('start', 0)
                                diagnoses.append((label, start))
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse extracted_disorders: {extracted_disorders[:100]}...")
        
        # Create a mapping from original date strings to parsed dates
        date_mapping = {}
        
        # Convert formatted dates to the format we need
        dates = []
        if formatted_dates:
            try:
                # First try to parse as a list directly
                if not isinstance(formatted_dates, str):
                    dates_list = formatted_dates
                # For strings, sanitize datetime.date objects first
                elif isinstance(formatted_dates, str):
                    # Replace datetime.date objects with ISO format strings
                    sanitized_dates = sanitize_datetime_strings(formatted_dates)
                    
                    if sanitized_dates.strip():
                        try:
                            # Try parsing the sanitized string
                            dates_list = ast.literal_eval(sanitized_dates)
                        except (ValueError, SyntaxError):
                            # If that fails, try json.loads
                            dates_list = json.loads(sanitized_dates)
                    else:
                        dates_list = []
                else:
                    dates_list = []
                    
                for date in dates_list:
                    if isinstance(date, dict):
                        # Handle both string dates and original datetime objects
                        parsed = date.get('parsed', '')
                        # If parsed is still a string representation of datetime.date, extract just the date
                        if isinstance(parsed, str) and parsed.startswith('datetime.date'):
                            # Extract YYYY-MM-DD from the string
                            match = re.search(r'datetime\.date\((\d+),\s*(\d+),\s*(\d+)\)', parsed)
                            if match:
                                year, month, day = match.groups()
                                parsed = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                        
                        original = date.get('original', '')
                        start = date.get('start', 0)
                        
                        # Add to our date mapping - normalize by replacing slashes with hyphens
                        if original and parsed:
                            normalized_original = original.replace('/', '-')
                            date_mapping[normalized_original] = parsed
                        
                        dates.append((parsed, original, start))
            except Exception as e:
                print(f"Warning: Could not parse formatted_dates: {formatted_dates[:100]}... Error: {str(e)}")
        
        if diagnoses and dates:
            notes_with_entities += 1
        
        # Create a ground truth mapping
        gt_relations = {}
        for section in relationship_gold:
            section_date = section.get('date', '')
            section_diagnoses = [diag['diagnosis'] for diag in section.get('diagnoses', [])]
            
            # Normalize the date format using our mapping if available
            normalized_section_date = section_date.replace('/', '-')
            standardized_date = date_mapping.get(normalized_section_date, section_date)
            
            for diagnosis in section_diagnoses:
                # Ground truth keys use normalized (lowercase, underscore) diagnosis and standardized date
                key = (diagnosis.strip().lower(), standardized_date.strip())
                gt_relations[key] = 1
        
        # Print ground truth relations for the first few notes
        if total_notes <= 2 and gt_relations:
            print(f"\n--- CLINICAL NOTE #{total_notes} ---")
            print(f"Ground Truth Relations ({len(gt_relations)} total):")
            for i, (key, _) in enumerate(gt_relations.items()):
                if i < 10:  # Show only first 10 for brevity
                    print(f"  {key}")
                else:
                    print(f"  ... and {len(gt_relations) - 10} more")
                    break
            
            # Also print the date mapping for debugging
            print("\nDate Mapping:")
            for i, (original, parsed) in enumerate(date_mapping.items()):
                if i < 10:  # Show only first 10 for brevity
                    print(f"  '{original}' -> '{parsed}'")
                else:
                    print(f"  ... and {len(date_mapping) - 10} more")
                    break
        
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
                
                # Print diagnostic info for the first few examples
                if total_notes <= 2 and total_examples < 10:
                    print(f"\nCandidate pair #{total_examples+1}:")
                    print(f"  Diagnosis: '{diagnosis}' -> Normalized: '{diagnosis.strip().lower()}'")
                    print(f"  Date: '{date_str}' -> Parsed: '{parsed_date}'")
                    print(f"  Lookup key: {key}")
                    print(f"  Found in ground truth: {key in gt_relations}")
                    print(f"  Assigned label: {label}")
                
                features.append(feature)
                labels.append(label)
                total_examples += 1
                if label == 1:
                    total_positive += 1
        
        all_features.extend(features)
        all_labels.extend(labels)
    
    # Print debug info
    print("\n===== SUMMARY =====")
    print(f"Processed {total_notes} clinical notes")
    print(f"Notes with entities: {notes_with_entities}/{total_notes}")
    print(f"Total examples generated: {total_examples}")
    print(f"Positive examples: {total_positive}/{total_examples} ({total_positive/max(1,total_examples)*100:.1f}%)")
    
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
    Load and prepare data for training the BERT model.
    
    Args:
        csv_path: Path to the CSV dataset
        pretrained_model_name: Name of the pretrained model for tokenization
        max_seq_length: Maximum sequence length for tokenization
        
    Returns:
        tuple: (train_dataset, val_dataset, tokenizer)
    """
    print(f"Loading data from {csv_path}...")
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    
    # Add special tokens for entity marking
    special_tokens = {'additional_special_tokens': ['[E1]', '[/E1]', '[E2]', '[/E2]']}
    tokenizer.add_special_tokens(special_tokens)
    
    # Process each row to create examples
    all_examples = []
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing notes"):
        # Get the text
        text = row.get(config.REAL_DATA_TEXT_COLUMN, '')
        if not text or pd.isna(text):
            continue
            
        # Get relationship gold data
        relationship_gold = row.get('relationship_gold', '')
        if isinstance(relationship_gold, str) and relationship_gold:
            # Find the first '[' which should be the start of the JSON array
            json_start = relationship_gold.find('[')
            if json_start >= 0:
                relationship_gold = relationship_gold[json_start:]
                
            # Parse the JSON string
            try:
                # Sanitize and parse relationship_gold if it's a string
                sanitized_gold = sanitize_datetime_strings(relationship_gold)
                gold_data = json.loads(sanitized_gold)
            except json.JSONDecodeError:
                try:
                    gold_data = eval(sanitized_gold)
                except (ValueError, SyntaxError):
                    gold_data = []
        else:
            gold_data = []
            
        # Create a set of gold standard relationships
        gold_relationships = set()
        for rel in gold_data:
            if isinstance(rel, dict):
                if 'entity_label' in rel and 'date' in rel:
                    gold_relationships.add((rel['entity_label'].lower(), rel['date']))
                elif 'diagnosis' in rel and 'date' in rel:
                    # Legacy format
                    gold_relationships.add((rel['diagnosis'].lower(), rel['date']))
                    
        # Get extracted entities
        extracted_disorders = row.get('extracted_disorders', '')
        if isinstance(extracted_disorders, str) and extracted_disorders:
            try:
                from utils.extraction_utils import transform_python_to_json
                disorders_json = transform_python_to_json(extracted_disorders)
                disorders_list = json.loads(disorders_json)
            except (json.JSONDecodeError, ValueError, SyntaxError):
                disorders_list = []
        else:
            disorders_list = []
            
        # Get extracted dates
        formatted_dates = row.get('formatted_dates', '')
        if isinstance(formatted_dates, str) and formatted_dates:
            try:
                from utils.extraction_utils import transform_python_to_json
                dates_json = transform_python_to_json(formatted_dates)
                dates_list = json.loads(dates_json)
            except (json.JSONDecodeError, ValueError, SyntaxError):
                dates_list = []
        else:
            dates_list = []
            
        # Skip if no entities or dates
        if not disorders_list or not dates_list:
            continue
            
        # Process disorders
        disorders = []
        for disorder in disorders_list:
            if isinstance(disorder, dict):
                label = disorder.get('label', '')
                start = disorder.get('start', 0)
                end = start + len(label) if 'end' not in disorder else disorder.get('end')
                category = disorder.get('category', 'disorder')
                
                if label and start >= 0:
                    disorders.append({
                        'label': label.lower(),
                        'start': start,
                        'end': end,
                        'category': category
                    })
                    
        # Process dates
        dates = []
        for date in dates_list:
            if isinstance(date, dict):
                parsed = date.get('parsed', '')
                original = date.get('original', '')
                start = date.get('start', 0)
                end = start + len(original) if original else start
                
                if parsed and original and start >= 0:
                    dates.append({
                        'parsed': parsed,
                        'original': original,
                        'start': start,
                        'end': end
                    })
                    
        # Create examples for all possible pairs
        for disorder in disorders:
            for date in dates:
                # Check if this is a positive example
                is_positive = (disorder['label'], date['parsed']) in gold_relationships
                
                # Skip if entities overlap
                if (disorder['start'] <= date['start'] and disorder['end'] >= date['end']) or \
                   (date['start'] <= disorder['start'] and date['end'] >= disorder['end']):
                    continue
                    
                # Create marked text
                marked_text = list(text)
                
                # Insert markers in reverse order (end to start) to avoid changing positions
                if disorder['start'] > date['start']:
                    # Date is first
                    marked_text.insert(disorder['end'], "[/E1]")
                    marked_text.insert(disorder['start'], "[E1]")
                    marked_text.insert(date['end'], "[/E2]")
                    marked_text.insert(date['start'], "[E2]")
                else:
                    # Disorder is first
                    marked_text.insert(date['end'], "[/E2]")
                    marked_text.insert(date['start'], "[E2]")
                    marked_text.insert(disorder['end'], "[/E1]")
                    marked_text.insert(disorder['start'], "[E1]")
                    
                marked_text = ''.join(marked_text)
                
                # Get context window around the entities
                min_pos = min(disorder['start'], date['start'])
                max_pos = max(disorder['end'], date['end'])
                
                # Get a window of text that includes both entities plus context
                context_start = max(0, min_pos - 200)
                context_end = min(len(text), max_pos + 200)
                context = marked_text[context_start:context_end]
                
                # Calculate character distance between entities
                dist1 = abs(disorder['start'] - date['end'])
                dist2 = abs(date['start'] - disorder['end'])
                char_distance = min(dist1, dist2)
                
                # Create example
                example = {
                    'text': context,
                    'disorder': disorder['label'],
                    'date': date['parsed'],
                    'distance': char_distance / 1000.0,  # Normalize distance
                    'label': 1 if is_positive else 0
                }
                
                all_examples.append(example)
                
    print(f"Created {len(all_examples)} examples")
    
    # Count positive and negative examples
    positive_count = sum(1 for ex in all_examples if ex['label'] == 1)
    negative_count = len(all_examples) - positive_count
    print(f"Positive examples: {positive_count} ({positive_count/len(all_examples)*100:.1f}%)")
    print(f"Negative examples: {negative_count} ({negative_count/len(all_examples)*100:.1f}%)")
    
    # Handle class imbalance by downsampling negative examples if needed
    if negative_count > 2 * positive_count:
        print("Downsampling negative examples...")
        positive_examples = [ex for ex in all_examples if ex['label'] == 1]
        negative_examples = [ex for ex in all_examples if ex['label'] == 0]
        
        # Sample twice as many negative examples as positive ones
        sampled_negative = random.sample(negative_examples, min(2 * positive_count, negative_count))
        balanced_examples = positive_examples + sampled_negative
        random.shuffle(balanced_examples)
        
        all_examples = balanced_examples
        print(f"After balancing: {len(all_examples)} examples")
        print(f"Positive examples: {len(positive_examples)} ({len(positive_examples)/len(all_examples)*100:.1f}%)")
        print(f"Negative examples: {len(sampled_negative)} ({len(sampled_negative)/len(all_examples)*100:.1f}%)")
    
    # Split into train and validation sets
    train_examples, val_examples = train_test_split(all_examples, test_size=0.2, random_state=42, 
                                                   stratify=[ex['label'] for ex in all_examples])
    
    print(f"Train set: {len(train_examples)} examples")
    print(f"Validation set: {len(val_examples)} examples")
    
    # Import BertEntityPairDataset here to avoid circular imports
    from model_training.BertEntityPairDataset import BertEntityPairDataset
    
    # Create datasets
    train_dataset = BertEntityPairDataset(train_examples, tokenizer, max_length=max_seq_length)
    val_dataset = BertEntityPairDataset(val_examples, tokenizer, max_length=max_seq_length)
    
    return train_dataset, val_dataset, tokenizer

def train_bert_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device, model_path):
    """
    Train the BERT model for relation extraction.
    
    Args:
        model: The BERT model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        device: Device to train on (cuda or cpu)
        model_path: Path to save the best model
        
    Returns:
        tuple: (train_losses, val_losses, val_accuracies) - Lists of training losses, validation losses and validation accuracies
    """
    # Training loop
    best_val_accuracy = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].long().to(device)  # Convert to long for CrossEntropyLoss
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].long().to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()
                
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
        val_accuracy = correct / total
        avg_val_loss = val_loss / len(val_loader)
        val_accuracies.append(val_accuracy)
        val_losses.append(avg_val_loss)
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Average validation loss: {avg_val_loss:.4f}")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"New best validation accuracy: {best_val_accuracy:.4f}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(model_path)
            print(f"Model saved to {model_path}")
    
    return train_losses, val_losses, val_accuracies

# plot_bert_training_curves has been consolidated with plot_training_curves