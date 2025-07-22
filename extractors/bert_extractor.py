import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from extractors.base_extractor import BaseRelationExtractor
import model_training.training_config as training_config

class BertExtractor(BaseRelationExtractor):
    """
    Relation extractor that uses a fine-tuned BERT model (e.g., BioBERT) 
    to identify entity-date relationships in clinical notes.
    """
    
    def __init__(self, config):
        """
        Initialize the BERT extractor.
        
        Args:
            config: Main configuration object (config.py).
        """
        self.config = config
        self.model_path = config.BERT_MODEL_PATH
        self.pretrained_model = training_config.BERT_PRETRAINED_MODEL
        self.max_seq_length = training_config.BERT_MAX_SEQ_LENGTH
        self.confidence_threshold = training_config.BERT_CONFIDENCE_THRESHOLD
        self.device = config.DEVICE if hasattr(config, 'DEVICE') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.name = "BERT"
        self.debug = getattr(config, 'MODEL_DEBUG_MODE', False)
        
        self.model = None
        self.tokenizer = None
        
    def load(self):
        """
        Load the BERT model and tokenizer.
        
        Returns:
            bool: True if successfully loaded, False otherwise.
        """
        try:
            # Check if fine-tuned model exists
            if os.path.exists(self.model_path):
                print(f"Loading fine-tuned BERT model from: {self.model_path}")
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            else:
                # If no fine-tuned model exists, load the base pre-trained model
                print(f"No fine-tuned model found at {self.model_path}. Loading base pre-trained model: {self.pretrained_model}")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.pretrained_model, 
                    num_labels=2  # Binary classification: relationship or no relationship
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
                print("Note: Using a pre-trained model without fine-tuning may result in poor performance.")
                
            self.model.to(self.device)
            self.model.eval()
            print(f"Successfully loaded {self.name} model")
            return True
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            self.model = None
            self.tokenizer = None
            return False
    
    def extract(self, text, entities=None):
        """
        Extract relationships between medical entities and dates in the text using BERT.
        
        Args:
            text (str): The clinical note text.
            entities (tuple, optional): A tuple of (entities_list, dates) if already extracted.
                
        Returns:
            list: A list of dictionaries, each representing a relationship:
                {
                    'entity_label': str,     # The entity text
                    'entity_category': str,  # The entity category
                    'date': str,             # The date text
                    'confidence': float      # A confidence score between 0 and 1
                }
        """
        if self.model is None or self.tokenizer is None:
            print("BERT model or tokenizer not loaded. Call load() first.")
            return []
            
        if entities is None:
            print("Error: entities parameter is required for BERT extraction.")
            return []
            
        entities_list, dates = entities
        
        # Convert entities to a format we can use
        disorders = []
        for entity in entities_list:
            if isinstance(entity, dict):
                entity_label = entity.get('label', '')
                entity_start = entity.get('start', 0)
                entity_end = entity.get('end', 0)
                entity_category = entity.get('category', 'unknown')
                
                # For now, only consider disorders if we're in disorder_only mode
                if hasattr(self.config, 'ENTITY_MODE') and self.config.ENTITY_MODE == 'disorder_only':
                    if entity_category.lower() == 'disorder':
                        disorders.append({
                            'label': entity_label,
                            'start': entity_start,
                            'end': entity_end,
                            'category': entity_category
                        })
                else:
                    # In multi-entity mode, consider all entities
                    disorders.append({
                        'label': entity_label,
                        'start': entity_start,
                        'end': entity_end,
                        'category': entity_category
                    })
        
        # Format dates
        formatted_dates = []
        for date_tuple in dates:
            parsed_date, original_date, start_pos = date_tuple
            
            # Find the end position by adding the length of the original date string
            end_pos = start_pos + len(original_date)
            
            formatted_dates.append({
                'parsed': parsed_date,
                'original': original_date,
                'start': start_pos,
                'end': end_pos
            })
        
        if self.debug:
            print(f"Found {len(disorders)} disorders and {len(formatted_dates)} dates")
            
        # Generate all possible pairs for classification
        pairs = []
        for disorder in disorders:
            for date in formatted_dates:
                # Calculate character distance between entities
                dist1 = abs(disorder['start'] - date['end'])
                dist2 = abs(date['start'] - disorder['end'])
                char_distance = min(dist1, dist2)
                
                # Determine if disorder comes before date in the text
                disorder_before_date = 1 if disorder['start'] < date['start'] else 0
                
                pairs.append({
                    'disorder': disorder,
                    'date': date,
                    'char_distance': char_distance,
                    'disorder_before_date': disorder_before_date
                })
        
        if not pairs:
            if self.debug:
                print("No entity-date pairs found for classification")
            return []
            
        # Prepare inputs for BERT
        inputs = []
        for pair in pairs:
            # Create a formatted input with special markers for the entities
            # We'll use [E1] and [/E1] to mark the disorder entity
            # and [E2] and [/E2] to mark the date entity
            
            disorder = pair['disorder']
            date = pair['date']
            
            # Handle the case where one entity is inside another
            if (disorder['start'] <= date['start'] and disorder['end'] >= date['end']) or \
               (date['start'] <= disorder['start'] and date['end'] >= disorder['end']):
                # Entities overlap, skip this pair
                continue
                
            # Create a copy of the text and insert markers
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
            
            inputs.append({
                'text': context,
                'disorder': disorder,
                'date': date
            })
            
        if self.debug:
            print(f"Prepared {len(inputs)} inputs for BERT classification")
            if inputs:
                print(f"Sample input: {inputs[0]['text'][:100]}...")
                
        # Process inputs in batches
        batch_size = training_config.BERT_BATCH_SIZE  # Use batch size from training config
        all_predictions = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            batch_texts = [inp['text'] for inp in batch]
            
            # Tokenize the inputs
            encoded_inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**encoded_inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                positive_probs = probabilities[:, 1].cpu().numpy()  # Probability of positive relationship
                
            # Store predictions
            for j, prob in enumerate(positive_probs):
                all_predictions.append({
                    'disorder': batch[j]['disorder'],
                    'date': batch[j]['date'],
                    'confidence': float(prob)
                })
                
        # Filter predictions by confidence threshold
        filtered_predictions = [p for p in all_predictions if p['confidence'] >= self.confidence_threshold]
        
        if self.debug:
            print(f"Found {len(filtered_predictions)} relationships above confidence threshold {self.confidence_threshold}")
            
        # Format the output
        relationships = []
        for pred in filtered_predictions:
            relationships.append({
                'entity_label': pred['disorder']['label'],
                'entity_category': pred['disorder']['category'],
                'date': pred['date']['parsed'],  # Use the parsed date
                'confidence': pred['confidence']
            })
            
        return relationships
