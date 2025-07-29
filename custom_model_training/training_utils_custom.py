import json
import sys
import os
import re
import torch
import matplotlib.pyplot as plt
import config # Import the config module
import ast  # Add ast module for literal_eval

# Add parent directory to path to allow importing from models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def prepare_custom_training_data(dataset_path_or_data, max_distance, vocab_class=None, data_split_mode='all'):
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
        
    Returns:
        tuple: (features, labels, vocab) - processed data and vocabulary instance
    """
    from utils.inference_eval_utils import load_and_prepare_data as canonical_load
    from utils.inference_eval_utils import preprocess_note_for_prediction, transform_python_to_json
    
    # Initialize vocabulary if needed
    vocab = vocab_class() if vocab_class else None
    
    if isinstance(dataset_path_or_data, str):
        # It's a file path, use the canonical function to load it
        print(f"Loading data from file: {dataset_path_or_data} with split mode: {data_split_mode}")
        
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
            category = entity_categories.get((entity, date), 'diagnosis')
            print(f"  {i+1}. Entity: '{entity}', Category: '{category}', Date: '{date}'")
    else:
        print("\nWARNING: No gold standard relationships found. Check your data format.")
    
    # Generate features and labels
    all_features = []
    all_labels = []
    total_examples = 0
    total_positive = 0
    
    # Determine if we're in multi-entity mode
    entity_mode = getattr(config, 'ENTITY_MODE', 'diagnosis_only')
    is_multi_entity = entity_mode == 'multi_entity'
    print(f"\nUsing entity mode: {entity_mode}")
    
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
            
            # Add entity category information for multi-entity mode
            if is_multi_entity and key in entity_categories:
                feature['entity_category'] = entity_categories[key]
            else:
                feature['entity_category'] = 'diagnosis'  # Default
            
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
        correct = 0
        total = 0
        
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
    # Make sure the save path has a proper extension
    if not save_path.endswith('.png'):
        # If no .png extension, add it
        save_path = save_path + '.png'
    
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