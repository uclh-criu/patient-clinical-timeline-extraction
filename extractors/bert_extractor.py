import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from extractors.base_extractor import BaseRelationExtractor

class BertExtractor(BaseRelationExtractor):
    """
    Relation extractor that uses a fine-tuned BERT model to predict relationships
    between disorders and dates.
    """
    
    def __init__(self, config):
        """
        Initialize the BERT extractor.
        
        Args:
            config: The configuration object or dict containing BERT_MODEL_PATH and other parameters.
        """
        super().__init__(config)  # Call the parent constructor
        self.model_path = config.BERT_MODEL_PATH
        self.name = "BERT"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.debug = getattr(config, 'MODEL_DEBUG_MODE', False)
        
        # Get BERT parameters from training config
        try:
            from bert_model_training import training_config_bert
            self.pretrained_model_name = training_config_bert.BERT_PRETRAINED_MODEL
            self.max_seq_length = training_config_bert.BERT_MAX_SEQ_LENGTH
            self.confidence_threshold = getattr(config, 'BERT_CONFIDENCE_THRESHOLD', 0.185)  # Get from main config
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not load training_config_bert. Using default values. Error: {e}")
            self.pretrained_model_name = 'dmis-lab/biobert-base-cased-v1.1'
            self.max_seq_length = 512
            self.confidence_threshold = 0.185  # Lower threshold
        
        # Initialize model and tokenizer as None (will be loaded in load())
        self.model = None
        self.tokenizer = None
    
    def load(self):
        """
        Load the BERT model and tokenizer.
        
        Returns:
            bool: True if loading was successful, False otherwise.
        """
        try:
            print(f"Loading BERT model from {self.model_path}")
            
            # Check if the model path exists
            if os.path.exists(self.model_path):
                # Load the fine-tuned model and tokenizer
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            else:
                # If model doesn't exist, load the pre-trained model
                print(f"Fine-tuned model not found at {self.model_path}. Loading pre-trained model {self.pretrained_model_name}")
                self.model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_model_name, num_labels=1)
                self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
                
                # Add special tokens for entity marking if they don't exist
                special_tokens = {'additional_special_tokens': ['[E1]', '[/E1]', '[E2]', '[/E2]']}
                self.tokenizer.add_special_tokens(special_tokens)
                self.model.resize_token_embeddings(len(self.tokenizer))
            
            # Move model to the appropriate device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            print(f"BERT model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            return False
    
    def extract(self, text, entities=None, note_id=None, patient_id=None):
        """
        Extract relationships between entities and dates using the BERT model.
        
        Args:
            text (str): The clinical note text.
            entities (tuple, optional): A tuple of (entities_list, dates) if already extracted.
            note_id (int, optional): The ID of the note being processed.
            patient_id (str, optional): The ID of the patient the note belongs to.
                
        Returns:
            list: A list of dictionaries, each representing a relationship:
                {
                    'entity_label': str,     # The entity text
                    'entity_category': str,  # The entity category
                    'date': str,             # The date text
                    'confidence': float,     # Model confidence score
                }
        """
        if self.model is None or self.tokenizer is None:
            print("Error: BERT model or tokenizer not loaded. Call load() first.")
            return []
        
        if entities is None:
            print("Error: entities parameter is required for BERT extractor.")
            return []
        
        entities_list, dates = entities
        relationships = []
        
        # Skip if either entities or dates are empty
        if not entities_list or not dates:
            return relationships
        
        # Process each entity-date pair
        for entity in entities_list:
            # Handle both dict and tuple formats for entities
            if isinstance(entity, dict):
                entity_label = entity.get('label', '')
                entity_pos = entity.get('start', 0)
                entity_category = entity.get('category', 'disorder')
            else:
                # Handle both tuple formats: (label, position) or (label, position, category)
                if len(entity) == 2:
                    # Legacy format: (label, position)
                    entity_label, entity_pos = entity
                    entity_category = 'disorder'  # Default category for legacy format
                elif len(entity) == 3:
                    # Multi-entity format: (label, position, category)
                    entity_label, entity_pos, entity_category = entity
                else:
                    print(f"Warning: Unexpected entity format: {entity}. Skipping.")
                    continue
            
            for date_tuple in dates:
                # Unpack the date tuple: (parsed_date, raw_date_str, position)
                parsed_date, date_str, date_pos = date_tuple
                
                # Create input text with special tokens to mark entities
                # Find the entity and date in the text
                entity_start = entity_pos
                entity_end = entity_start + len(entity_label)
                date_start = date_pos
                date_end = date_start + len(date_str)
                
                # Create a context window around the entity and date
                context_start = max(0, min(entity_start, date_start) - 100)
                context_end = min(len(text), max(entity_end, date_end) + 100)
                
                # Extract the context and insert special tokens
                context = text[context_start:context_end]
                
                # Adjust positions relative to the context
                rel_entity_start = entity_start - context_start
                rel_entity_end = entity_end - context_start
                rel_date_start = date_start - context_start
                rel_date_end = date_end - context_start
                
                # Handle overlapping entities (unlikely but possible)
                if rel_entity_start <= rel_date_end and rel_date_start <= rel_entity_end:
                    # Entities overlap, skip this pair
                    continue
                
                # Insert special tokens to mark the entities
                # We need to handle the case where entity comes before date or vice versa
                if rel_entity_start < rel_date_start:
                    # Entity comes first
                    marked_text = (
                        context[:rel_entity_start] + 
                        "[E1]" + context[rel_entity_start:rel_entity_end] + "[/E1]" + 
                        context[rel_entity_end:rel_date_start] + 
                        "[E2]" + context[rel_date_start:rel_date_end] + "[/E2]" + 
                        context[rel_date_end:]
                    )
                else:
                    # Date comes first
                    marked_text = (
                        context[:rel_date_start] + 
                        "[E2]" + context[rel_date_start:rel_date_end] + "[/E2]" + 
                        context[rel_date_end:rel_entity_start] + 
                        "[E1]" + context[rel_entity_start:rel_entity_end] + "[/E1]" + 
                        context[rel_entity_end:]
                    )
                
                # Tokenize and prepare input
                inputs = self.tokenizer(
                    marked_text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt"
                )
                
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Perform inference
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    
                    # Convert logits to probabilities using sigmoid
                    probabilities = torch.sigmoid(logits)
                    confidence = probabilities.item()
                
                # DIAGNOSTIC PRINT: Show confidence for each pair
                #if self.debug:
                    #print(f"  - BERT Pair: ('{entity_label}', '{parsed_date}') -> Confidence: {confidence:.4f}")
                
                # Only include relationships above the confidence threshold
                if confidence >= self.confidence_threshold:
                    relationships.append({
                        'entity_label': entity_label,
                        'entity_category': entity_category,
                        'date': parsed_date,
                        'confidence': confidence
                    })
        
        # DIAGNOSTIC SUMMARY
        if self.debug and len(entities_list) > 0 and len(dates) > 0:
            print(f"\n===== BERT PREDICTION SUMMARY =====")
            print(f"Total entity-date pairs considered: {len(entities_list) * len(dates)}")
            print(f"Pairs above confidence threshold ({self.confidence_threshold}): {len(relationships)}")
            print(f"Current confidence threshold: {self.confidence_threshold}")
            if len(relationships) == 0:
                print(f"WARNING: No predictions above threshold. Consider lowering the confidence threshold.")
            print(f"==============================\n")
        
        return relationships
