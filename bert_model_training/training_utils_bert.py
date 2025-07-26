import sys
import os
import torch
from transformers import AutoTokenizer
import config # Import the config module

# Add parent directory to path to allow importing from models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    from BertEntityPairDataset import BertEntityPairDataset
    
    # Import the canonical load_and_prepare_data function
    from utils.inference_utils import load_and_prepare_data
    
    # Ensure config has the necessary attributes for BERT training
    if not hasattr(config, 'RELATIONSHIP_GOLD_COLUMN'):
        setattr(config, 'RELATIONSHIP_GOLD_COLUMN', 'relationship_gold')
    if not hasattr(config, 'DATES_COLUMN'):
        setattr(config, 'DATES_COLUMN', 'formatted_dates')
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