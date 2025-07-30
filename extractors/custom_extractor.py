import os
import torch
import sys

# Add project root to path to allow importing from other directories
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from extractors.base_extractor import BaseRelationExtractor
from custom_model_training.DiagnosisDateRelationModel import DiagnosisDateRelationModel
from utils.inference_eval_utils import preprocess_note_for_prediction, create_prediction_dataset, predict_relationships

class CustomExtractor(BaseRelationExtractor):
    """
    Relation extractor that uses the custom-trained PyTorch neural network 
    model (`DiagnosisDateRelationModel`) to identify entity-date relationships.
    """
    
    def __init__(self, config):
        """
        Initialize the custom PyTorch model extractor.
        
        Args:
            config: Main configuration object (config.py).
        """
        self.config = config 
        self.model_path = config.MODEL_PATH
        self.vocab_path = config.VOCAB_PATH
        # Use specific prediction parameters from main config
        self.pred_max_distance = getattr(config, 'PREDICTION_MAX_DISTANCE', 500)
        self.pred_max_context_len = getattr(config, 'PREDICTION_MAX_CONTEXT_LEN', 512)
        self.device = config.DEVICE
        self.name = "Custom (PyTorch NN)"
        self.debug = getattr(config, 'MODEL_DEBUG_MODE', False)
        self.threshold = getattr(config, 'CUSTOM_CONFIDENCE_THRESHOLD', 0.5)
        
        # Track all unique entity categories seen during extraction
        self.seen_categories = set()
        self.seen_raw_to_normalized = {}  # Track raw to normalized category mappings
        
        # Use the centralized category mappings from config
        self.category_mapping = config.CATEGORY_MAPPINGS
        
        self.model = None
        self.vocab = None 
        self.model_config = None
        
    def load(self):
        """
        Load the trained custom model and vocabulary.
        
        Returns:
            bool: True if successfully loaded, False otherwise.
        """
        try:
            if not os.path.exists(self.model_path) or not os.path.exists(self.vocab_path):
                print(f"Error: Custom model file {self.model_path} or vocabulary file {self.vocab_path} not found.")
                print(f"Please run model_training/train.py first.")
                return False
            
            print(f"Loading vocabulary from: {self.vocab_path}")
            # Handle different PyTorch versions - safe_globals was removed in newer versions
            try:
                # Try the newer PyTorch approach
                self.vocab = torch.load(self.vocab_path, weights_only=False)
            except TypeError:
                # For older PyTorch versions that don't have weights_only
                self.vocab = torch.load(self.vocab_path)
            
            if not hasattr(self.vocab, 'n_words'):
                 print("Error: Loaded vocabulary object does not have 'n_words' attribute.")
                 self.vocab = None
                 return False
            print(f"Vocabulary loaded successfully. Size: {self.vocab.n_words} words.")
            
            print(f"Loading custom model from: {self.model_path}")
            
            # Load the saved model checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Check if the checkpoint contains hyperparameters
            if isinstance(checkpoint, dict) and 'hyperparameters' in checkpoint:
                # New format - contains hyperparameters
                self.model_config = checkpoint['hyperparameters']
                model_state_dict = checkpoint['model_state_dict']
                vocab_size = checkpoint.get('vocab_size', self.vocab.n_words)
                
                # Get the best threshold if available
                if 'best_threshold' in checkpoint:
                    self.threshold = checkpoint['best_threshold']
                    print(f"Using optimal threshold from saved model: {self.threshold:.2f}")
                else:
                    print(f"No threshold found in model file. Using default threshold: {self.threshold:.2f}")
                
                # Initialize model with saved hyperparameters
                self.model = DiagnosisDateRelationModel(
                    vocab_size=vocab_size,
                    embedding_dim=self.model_config['EMBEDDING_DIM'],
                    hidden_dim=self.model_config['HIDDEN_DIM'],
                    apply_sigmoid=not self.model_config.get('USE_WEIGHTED_LOSS', False)
                ).to(self.device)
                
                self.model.load_state_dict(model_state_dict)
            else:
                # Old format - just the model state dict
                print("Warning: Model file doesn't contain hyperparameters. Using default values.")
                from custom_model_training.training_config_custom import EMBEDDING_DIM, HIDDEN_DIM
                
                # First try to load the model with default parameters
                try:
                    # Try with different embedding dimensions
                    for embedding_dim in [100, 128, 200, 256]:
                        for hidden_dim in [128, 256]:
                            try:
                                print(f"Trying with embedding_dim={embedding_dim}, hidden_dim={hidden_dim}")
                                self.model = DiagnosisDateRelationModel(
                                    vocab_size=self.vocab.n_words,
                                    embedding_dim=embedding_dim,
                                    hidden_dim=hidden_dim
                                ).to(self.device)
                                
                                self.model.load_state_dict(checkpoint)
                                print(f"Successfully loaded model with parameters: embedding_dim={embedding_dim}, hidden_dim={hidden_dim}")
                                
                                # Create a model config for compatibility
                                self.model_config = {
                                    'EMBEDDING_DIM': embedding_dim,
                                    'HIDDEN_DIM': hidden_dim,
                                    'USE_WEIGHTED_LOSS': False
                                }
                                
                                return True
                            except Exception:
                                # Try next combination
                                pass
                    
                    # If we get here, none of the combinations worked
                    raise ValueError("None of the parameter combinations worked")
                except Exception as e:
                    print(f"Error loading with default parameters: {e}")
                    print("Trying to create a compatible model structure...")
                    
                    # If loading fails, try to infer the parameters from the checkpoint
                    # This is a bit hacky but necessary for backward compatibility
                    # Try to infer model parameters from the checkpoint
                    inferred_params = {}
                    
                    # Check for embedding dimensions
                    if 'embedding.weight' in checkpoint:
                        inferred_params['vocab_size'], inferred_params['embedding_dim'] = checkpoint['embedding.weight'].shape
                        print(f"Inferred embedding_dim: {inferred_params['embedding_dim']}")
                    
                    # Try to infer hidden_dim from various weights
                    if 'conv1.weight' in checkpoint:
                        inferred_params['hidden_dim'] = checkpoint['conv1.weight'].shape[0]
                        print(f"Inferred hidden_dim from conv1.weight: {inferred_params['hidden_dim']}")
                    elif 'lstm.weight_ih_l0' in checkpoint:
                        # LSTM weights have 4x the hidden dimension
                        inferred_params['hidden_dim'] = checkpoint['lstm.weight_ih_l0'].shape[0] // 4
                        print(f"Inferred hidden_dim from lstm.weight_ih_l0: {inferred_params['hidden_dim']}")
                    elif 'fc2.weight' in checkpoint:
                        inferred_params['hidden_dim'] = checkpoint['fc2.weight'].shape[1]
                        print(f"Inferred hidden_dim from fc2.weight: {inferred_params['hidden_dim']}")
                    
                    # If we have enough information, try to create the model
                    if 'embedding_dim' in inferred_params and 'hidden_dim' in inferred_params:
                        print(f"Creating model with inferred parameters: embedding_dim={inferred_params['embedding_dim']}, hidden_dim={inferred_params['hidden_dim']}")
                        
                        # Create model with inferred parameters
                        try:
                            self.model = DiagnosisDateRelationModel(
                                vocab_size=self.vocab.n_words,
                                embedding_dim=inferred_params['embedding_dim'],
                                hidden_dim=inferred_params['hidden_dim']
                            ).to(self.device)
                            
                            # Try to load the state dictionary
                            self.model.load_state_dict(checkpoint)
                            print("Successfully loaded model with inferred parameters")
                            
                            # Create a simple model config for compatibility
                            self.model_config = {
                                'EMBEDDING_DIM': inferred_params['embedding_dim'],
                                'HIDDEN_DIM': inferred_params['hidden_dim'],
                                'USE_WEIGHTED_LOSS': False
                            }
                            
                            return True
                        except Exception as e2:
                            print(f"Error loading with inferred parameters: {e2}")
                            return False
                    else:
                        print("Could not infer all required parameters from checkpoint")
                        print(f"Available parameters: {list(inferred_params.keys())}")
                        print(f"Missing parameters: {'embedding_dim' if 'embedding_dim' not in inferred_params else ''} {'hidden_dim' if 'hidden_dim' not in inferred_params else ''}")
                        return False
            
            self.model.eval()
            print(f"Successfully loaded {self.name}")
            return True
        except Exception as e:
            print(f"Error loading custom model: {e}")
            self.model = None
            self.vocab = None
            return False
    
    def extract(self, text, entities=None, note_id=None, patient_id=None):
        """
        Extract relationships using the custom PyTorch model.
        
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
                    'confidence': float      # Model prediction confidence
                }
        """
        if self.model is None or self.vocab is None:
            print("Custom Model or Vocabulary not loaded. Call load() first.")
            return []
        
        if entities is None:
            print("Error: entities parameter is required for CSV data processing.")
            return []
        
        entities_list, dates = entities
        
        # Collect unique entity categories for later analysis
        for entity in entities_list:
            if isinstance(entity, tuple) and len(entity) > 2:
                self.seen_categories.add(entity[2])
        
        # Convert the new entity format to the format expected by the model
        diagnoses = []
        for entity in entities_list:
            if isinstance(entity, dict):
                # New format: dict with label, start, etc.
                entity_label = entity.get('label', '')
                entity_pos = entity.get('start', 0)
                entity_category = entity.get('category', 'unknown')
                diagnoses.append((entity_label, entity_pos, entity_category))
            elif isinstance(entity, tuple) and len(entity) >= 2:
                # Legacy format: tuple of (label, position)
                entity_label, entity_pos = entity[0], entity[1]
                entity_category = entity[2] if len(entity) > 2 else 'disorder'
                diagnoses.append((entity_label, entity_pos, entity_category))
            elif isinstance(entity, list) and len(entity) >= 2:
                # List format: [label, position]
                entity_label, entity_pos = entity[0], entity[1]
                entity_category = entity[2] if len(entity) > 2 else 'disorder'
                diagnoses.append((entity_label, entity_pos, entity_category))
        
        # Process dates to ensure they are in the correct format
        formatted_dates = []
        for date in dates:
            if isinstance(date, tuple) and len(date) >= 3:
                # Already in correct format: (parsed_date, original_date, position)
                formatted_dates.append(date)
            elif isinstance(date, tuple) and len(date) == 2:
                # Tuple with just (parsed_date, position)
                parsed_date, position = date
                formatted_dates.append((parsed_date, parsed_date, position))
            elif isinstance(date, str):
                # Just a string, assume it's both parsed and original, with position 0
                formatted_dates.append((date, date, 0))
        
        # Extract just the position information needed for preprocess_note_for_prediction
        diagnoses_for_model = [(d[0], d[1]) for d in diagnoses]
        
        features = preprocess_note_for_prediction(text, diagnoses_for_model, formatted_dates, self.pred_max_distance)
        
        # Pass the confirmed lists to the next function
        test_data = create_prediction_dataset(features, self.vocab, self.device, 
                                              self.pred_max_distance, self.pred_max_context_len)
        
        # Create a dictionary to store all predictions for each entity
        entity_predictions = {}
        self.model.eval()
        
        prediction_count = 0
        
        with torch.no_grad():
            for data in test_data:
                # Get entity category ID (0=diagnosis, 1=symptom, 2=procedure, 3=medication)
                feature = data['feature']
                diagnosis = feature['diagnosis']
                date = feature['date']
                
                # Find the corresponding entity category
                entity_category = 'disorder'  # Default
                entity_category_id = 0  # Default to diagnosis (0)
                for entity_label, _, raw_category in diagnoses:
                    if entity_label == diagnosis:
                        # Map the raw category to one of our four simplified categories
                        raw_category_lower = raw_category.lower()
                        
                        # Use the mapping dictionary to normalize the category
                        if raw_category_lower in self.category_mapping:
                            normalized_category = self.category_mapping[raw_category_lower]
                        else:
                            # Default to 'diagnosis' for unknown categories
                            normalized_category = 'diagnosis'
                            
                        # Track the mapping for debugging
                        if raw_category_lower not in self.seen_raw_to_normalized:
                            self.seen_raw_to_normalized[raw_category_lower] = normalized_category
                            
                        entity_category = raw_category
                        
                        # Map normalized category to ID
                        if normalized_category == 'symptom':
                            entity_category_id = 1
                        elif normalized_category == 'procedure':
                            entity_category_id = 2
                        elif normalized_category == 'medication':
                            entity_category_id = 3
                        else:  # Default to diagnosis
                            entity_category_id = 0
                        break
                
                # Convert to tensor
                entity_category_tensor = torch.tensor([entity_category_id], dtype=torch.long, device=self.device)
                
                # Track the mapping of raw category to ID for later analysis
                if entity_category not in self.seen_categories:
                    self.seen_categories.add(entity_category)
                
                # Pass entity category to model
                try:
                    output = self.model(data['context'], data['distance'], data['diag_before'], entity_category_tensor)
                except Exception:
                    # Fallback if entity_category is not supported
                    output = self.model(data['context'], data['distance'], data['diag_before'])
                
                # Check if we need to apply sigmoid (if model is configured to output raw logits)
                if hasattr(self.model, 'apply_sigmoid') and not self.model.apply_sigmoid:
                    import torch.nn.functional as F
                    prob = F.sigmoid(output).item()
                    #if self.debug:
                        #print(f"Applied sigmoid to raw logit {output.item():.4f} -> probability {prob:.4f}")
                else:
                    prob = output.item()
                
                prediction_count += 1
                
                # Find the corresponding entity category
                entity_category = 'disorder'  # Default
                for entity_label, _, category in diagnoses:
                    if entity_label == diagnosis:
                        entity_category = category
                        break
                
                # Create a unique key that includes both label and category
                entity_key = f"{diagnosis}|{entity_category}"
                
                # Store this prediction for the entity
                if entity_key not in entity_predictions:
                    entity_predictions[entity_key] = []
                
                entity_predictions[entity_key].append({
                    'date': date,
                    'confidence': prob
                })
        
        # For each entity, select the date with the highest confidence
        relationships = []
        
        # Disable verbose diagnostic output as we now have a unified output format
        # in the run_extraction function
        
        for entity_key, predictions in entity_predictions.items():
            if predictions:
                # Split the key back into label and category
                entity_label, entity_category = entity_key.split('|', 1)
                
                # Sort predictions by confidence (highest first)
                best_prediction = max(predictions, key=lambda x: x['confidence'])
                
                # Only include predictions that meet the threshold
                if best_prediction['confidence'] >= self.threshold:
                    # Add the best prediction to our results
                    relationships.append({
                        'entity_label': entity_label,
                        'entity_category': entity_category,
                        'date': best_prediction['date'],
                        'confidence': best_prediction['confidence']
                    })
                # We no longer need to print rejected predictions here
        
        return relationships 