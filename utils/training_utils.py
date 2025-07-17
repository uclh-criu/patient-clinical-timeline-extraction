import json
import sys
import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
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