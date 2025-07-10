import os
import torch
from extractors.base_extractor import BaseRelationExtractor
from model_training.DiagnosisDateRelationModel import DiagnosisDateRelationModel
from model_training.Vocabulary import Vocabulary
from model_training.training_config import EMBEDDING_DIM, HIDDEN_DIM
from utils.extraction_utils import preprocess_note_for_prediction, create_prediction_dataset, predict_relationships

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
        
        self.model = None
        self.vocab = None 
        
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
            self.model = DiagnosisDateRelationModel(
                vocab_size=self.vocab.n_words,
                embedding_dim=EMBEDDING_DIM,
                hidden_dim=HIDDEN_DIM
            ).to(self.device)
            
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
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
        
        # Convert the new entity format to the format expected by the model
        diagnoses = []
        for entity in entities_list:
            if isinstance(entity, dict):
                # New format: dict with label, start, etc.
                entity_label = entity.get('label', '')
                entity_pos = entity.get('start', 0)
                entity_category = entity.get('category', 'unknown')
                diagnoses.append((entity_label, entity_pos, entity_category))
            else:
                # Legacy format: tuple of (label, position)
                entity_label, entity_pos = entity
                diagnoses.append((entity_label, entity_pos, 'disorder'))  # Default category
        
        # Extract just the position information needed for preprocess_note_for_prediction
        diagnoses_for_model = [(d[0], d[1]) for d in diagnoses]
        
        features = preprocess_note_for_prediction(text, diagnoses_for_model, dates, self.pred_max_distance)
        # Pass the confirmed lists to the next function
        test_data = create_prediction_dataset(features, self.vocab, self.device, 
                                              self.pred_max_distance, self.pred_max_context_len)
        
        # Create a dictionary to store all predictions for each entity
        entity_predictions = {}
        self.model.eval()
        
        with torch.no_grad():
            for data in test_data:
                output = self.model(data['context'], data['distance'], data['diag_before'])
                prob = output.item()
                
                feature = data['feature']
                diagnosis = feature['diagnosis']
                date = feature['date']
                
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
        for entity_key, predictions in entity_predictions.items():
            if predictions:
                # Split the key back into label and category
                entity_label, entity_category = entity_key.split('|', 1)
                
                # Sort predictions by confidence (highest first)
                best_prediction = max(predictions, key=lambda x: x['confidence'])
                
                # Add the best prediction to our results
                relationships.append({
                    'entity_label': entity_label,
                    'entity_category': entity_category,
                    'date': best_prediction['date'],
                    'confidence': best_prediction['confidence']
                })
        
        return relationships 