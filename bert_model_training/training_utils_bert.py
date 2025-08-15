import sys
import os
import torch
from transformers import AutoTokenizer
import config # Import the config module
import bert_model_training.training_config_bert as training_config_bert

# Add parent directory to path to allow importing from models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def prepare_bert_training_data(csv_path, pretrained_model_name, max_seq_length, data_split_mode='all'):
    """
    Prepare data for training the BERT model.
    
    Args:
        csv_path: Path to the CSV dataset
        pretrained_model_name: Name or path of the pretrained BERT model
        max_seq_length: Maximum sequence length for tokenization
        data_split_mode (str): How to split the data. Options:
            - 'train': Use only the training portion (first TRAINING_SET_RATIO)
            - 'test': Use only the testing portion (remaining 1-TRAINING_SET_RATIO)
            - 'all': Use all data without splitting (default)
        
    Returns:
        tuple: (train_dataset, val_dataset, tokenizer) - datasets and tokenizer
    """

    from BertEntityPairDataset import BertEntityPairDataset
    
    # Import the canonical load_and_prepare_data function
    from utils.inference_eval_utils import load_and_prepare_data
    
    # Ensure config has the necessary attributes for BERT training
    if not hasattr(config, 'RELATIONSHIP_GOLD_COLUMN'):
        setattr(config, 'RELATIONSHIP_GOLD_COLUMN', 'relationship_gold')
    if not hasattr(config, 'DATES_COLUMN'):
        setattr(config, 'DATES_COLUMN', 'formatted_dates')
    
    # Set entity mode from training_config_bert
    entity_mode = training_config_bert.ENTITY_MODE
    setattr(config, 'ENTITY_MODE', entity_mode)
    
    if not hasattr(config, 'ENABLE_RELATIVE_DATE_EXTRACTION'):
        setattr(config, 'ENABLE_RELATIVE_DATE_EXTRACTION', False)
    
    # Load the data using the canonical function with the specified data split mode
    print(f"Loading data with split mode: {data_split_mode}")
    print(f"Using entity mode: {entity_mode}")
    
    try:
        # Handle different return signatures based on ENTITY_MODE
        if entity_mode == 'diagnosis_only':
            prepared_data, relationship_gold = load_and_prepare_data(
                csv_path, None, config, data_split_mode=data_split_mode
            )
        else:
            prepared_data, entity_gold, relationship_gold, _ = load_and_prepare_data(
                csv_path, None, config, data_split_mode=data_split_mode
            )
    except ZeroDivisionError:
        print("Error: ZeroDivisionError occurred during data loading. This is likely due to an empty dataset after splitting.")
        print("Trying again with data_split_mode='all' to use all available data...")
        
        if entity_mode == 'diagnosis_only':
            prepared_data, relationship_gold = load_and_prepare_data(
                csv_path, None, config, data_split_mode='all'
            )
        else:
            prepared_data, entity_gold, relationship_gold, _ = load_and_prepare_data(
                csv_path, None, config, data_split_mode='all'
            )
    
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
    gold_relationships = {}  # (note_id, entity_label, date) -> entity_category
    
    for rel in relationship_gold:
        # Handle both diagnosis_only and multi_entity modes
        if 'diagnosis' in rel:
            key = (rel['note_id'], rel['diagnosis'].lower(), rel['date'])
            gold_relationships[key] = 'diagnosis'
        elif 'entity_label' in rel:
            key = (rel['note_id'], rel['entity_label'].lower(), rel['date'])
            gold_relationships[key] = rel.get('entity_category', 'diagnosis').lower()
    
    print(f"Found {len(gold_relationships)} gold standard relationships")
    
    # Create entity pairs for BERT training
    entity_pairs = []
    
    # Track entity categories for statistics
    entity_category_counts = {
        'diagnosis': 0,
        'symptom': 0,
        'procedure': 0,
        'medication': 0,
        'unknown': 0
    }
    
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
                entity_category = entity.get('category', 'diagnosis').lower()
            else:
                # Handle both tuple formats: (label, position) or (label, position, category)
                if len(entity) == 2:
                    # Legacy format: (label, position)
                    entity_label, entity_start = entity
                    entity_end = entity_start + len(entity_label)
                    entity_category = 'diagnosis'  # Default category for legacy format
                elif len(entity) == 3:
                    # Multi-entity format: (label, position, category)
                    entity_label, entity_start, entity_category = entity
                    entity_end = entity_start + len(entity_label)
                else:
                    print(f"Warning: Unexpected entity format in prepare_bert_training_data: {entity}. Skipping.")
                    continue
            
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
                key = (note_id, entity_label.lower(), parsed_date)
                
                if key in gold_relationships:
                    is_gold = True
                    # Use the category from gold standard if available
                    entity_category = gold_relationships[key]
                
                # Update category counts
                if is_gold:
                    if entity_category in entity_category_counts:
                        entity_category_counts[entity_category] += 1
                    else:
                        entity_category_counts['unknown'] += 1
                
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
    print(f"Class distribution: {positive} positive examples ({positive/len(entity_pairs)*100:.1f}%), {negative} negative examples ({negative/len(entity_pairs)*100:.1f}%)")
    
    # Print entity category distribution for positive examples
    if entity_mode == 'multi_entity':
        print("Entity category distribution for positive examples:")
        for category, count in entity_category_counts.items():
            if count > 0:
                print(f"  {category}: {count} examples ({count/positive*100:.1f}%)")
    
    # Balance the dataset if needed (downsample negative examples)
    if negative > 5 * positive:  # If negative examples are more than 5x positive ones
        import random
        print("Downsampling negative examples to balance the dataset...")
        positive_pairs = [pair for pair in entity_pairs if pair['label'] == 1]
        negative_pairs = [pair for pair in entity_pairs if pair['label'] == 0]
        
        # Randomly select a subset of negative examples (5x the number of positive examples)
        random.seed(42)  # For reproducibility
        sampled_negative_pairs = random.sample(negative_pairs, min(5 * len(positive_pairs), len(negative_pairs)))
        
        # Combine positive and sampled negative examples
        entity_pairs = positive_pairs + sampled_negative_pairs
        print(f"After balancing: {len(positive_pairs)} positive examples, {len(sampled_negative_pairs)} negative examples")
    
    # Split into training and validation sets
    from sklearn.model_selection import train_test_split
    
    train_pairs, val_pairs = train_test_split(entity_pairs, test_size=0.2, random_state=42, stratify=[p['label'] for p in entity_pairs])
    
    print(f"Total entity pairs: {len(entity_pairs)}. Splitting into {len(train_pairs)} training examples and {len(val_pairs)} validation examples.")
    
    # Create datasets
    train_dataset = BertEntityPairDataset.create_from_pairs(train_pairs, tokenizer, max_seq_length)
    val_dataset = BertEntityPairDataset.create_from_pairs(val_pairs, tokenizer, max_seq_length)
    
    return train_dataset, val_dataset, tokenizer

def train_bert_model(model, tokenizer, train_loader, val_loader, optimizer, scheduler, num_epochs, device, model_save_path):
    """
    Train the BERT model.
    
    Args:
        model: The BERT model
        tokenizer: The BERT tokenizer
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train for
        device: Device to train on (cpu or cuda)
        model_save_path: Directory path to save the best model (Hugging Face format)
        
    Returns:
        dict: Dictionary containing training metrics
    """
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    # Training metrics
    train_losses = []
    val_losses = []
    val_accs = []
    val_f1s = []
    best_val_f1 = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_steps = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Handle entity category if available
            entity_category = batch.get('entity_category')
            if entity_category is not None:
                entity_category = entity_category.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Different models might have different forward signatures
            try:
                if entity_category is not None:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        entity_category=entity_category
                    )
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
            except TypeError:
                # If model doesn't accept entity_category, use standard signature
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            # Get logits and compute loss
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Handle different output shapes
            if len(logits.shape) > 1 and logits.shape[1] > 1:
                # Multi-class classification
                loss = F.cross_entropy(logits, labels.long())
            else:
                # Binary classification
                loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_steps += 1
        
        # Calculate average training loss
        avg_train_loss = train_loss / train_steps
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Handle entity category if available
                entity_category = batch.get('entity_category')
                if entity_category is not None:
                    entity_category = entity_category.to(device)
                
                # Forward pass
                try:
                    if entity_category is not None:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            entity_category=entity_category
                        )
                    else:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                except TypeError:
                    # If model doesn't accept entity_category, use standard signature
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                
                # Get logits and compute loss
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                # Handle different output shapes
                if len(logits.shape) > 1 and logits.shape[1] > 1:
                    # Multi-class classification
                    loss = F.cross_entropy(logits, labels.long())
                    preds = torch.argmax(logits, dim=1)
                else:
                    # Binary classification
                    loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels)
                    preds = (torch.sigmoid(logits.view(-1)) > 0.5).float()
                
                val_loss += loss.item()
                val_steps += 1
                
                # Collect predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        avg_val_loss = val_loss / val_steps
        val_losses.append(avg_val_loss)
        
        # Calculate accuracy
        val_acc = accuracy_score(all_labels, all_preds) * 100
        val_accs.append(val_acc)
        
        # Calculate precision, recall, and F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        val_f1s.append(f1)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Track the best model state
        if f1 > best_val_f1:
            best_val_f1 = f1
            print(f"New best model with validation F1-score: {f1:.4f}")
            print(f"  Validation Accuracy: {val_acc:.2f}%")
            print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")
            
            # Store the model state dict (not saving to disk yet)
            best_model_state = model.state_dict().copy()
    
    # Return metrics
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'val_f1s': val_f1s,
        'best_val_f1': best_val_f1,
        'best_val_acc': val_accs[val_f1s.index(best_val_f1)] if val_f1s else 0,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'final_val_acc': val_accs[-1],
        'final_val_f1': val_f1s[-1] if val_f1s else 0,
        'epochs': num_epochs,
        'train_loss': train_losses[-1],  # For consistency with log_training_run
        'val_loss': val_losses[-1],      # For consistency with log_training_run
        'val_acc': val_accs[-1],         # For consistency with log_training_run
        'val_f1': val_f1s[-1] if val_f1s else 0  # For consistency with log_training_run
    }
    
    return best_model_state, metrics

# plot_bert_training_curves has been consolidated with plot_training_curves