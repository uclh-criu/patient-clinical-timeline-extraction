import os
import torch
from extractors.base_extractor import BaseRelationExtractor
from custom_model_training.DiagnosisDateRelationModel import DiagnosisDateRelationModel
from custom_model_training.Vocabulary import Vocabulary
from custom_model_training.training_config_custom import EMBEDDING_DIM, HIDDEN_DIM
from utils.inference_utils import preprocess_note_for_prediction, create_prediction_dataset, predict_relationships

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
        
        # DIAGNOSTIC: Print the extracted entities and dates
        # Comment out the extracted entities and dates section
        """
        if self.debug:
            print("\n===== DIAGNOSTIC: EXTRACTED ENTITIES AND DATES =====")
            print(f"Number of diagnoses: {len(diagnoses_for_model)}")
            for i, (label, pos) in enumerate(diagnoses_for_model[:5]):  # Show first 5
                print(f"  Diagnosis {i+1}: '{label}' at position {pos}")
                # Show the text snippet around this diagnosis
                start = max(0, pos - 20)
                end = min(len(text), pos + 20)
                snippet = text[start:end].replace('\n', ' ')
                print(f"     Context: '...{snippet}...'")
            
            print(f"Number of dates: {len(dates)}")
            for i, (parsed_date, date_str, pos) in enumerate(dates[:5]):  # Show first 5
                print(f"  Date {i+1}: '{date_str}' at position {pos}, parsed as '{parsed_date}'")
                # Show the text snippet around this date
                start = max(0, pos - 20)
                end = min(len(text), pos + 20)
                snippet = text[start:end].replace('\n', ' ')
                print(f"     Context: '...{snippet}...'")
        """
        
        features = preprocess_note_for_prediction(text, diagnoses_for_model, dates, self.pred_max_distance)
        
        # DIAGNOSTIC: Print the generated features (candidate pairs)
        # Comment out the redundant candidate pairs section
        """
        if self.debug:
            print("\n===== DIAGNOSTIC: GENERATED CANDIDATE PAIRS =====")
            print(f"Number of candidate pairs: {len(features)}")
            for i, feature in enumerate(features[:3]):  # Show first 3 pairs
                print(f"\nCandidate Pair {i+1}:")
                print(f"  Diagnosis: '{feature['diagnosis']}'")
                print(f"  Date: '{feature['date']}'")
                print(f"  Distance (words): {feature['distance']}")
                print(f"  Diagnosis before date: {feature['diag_before_date']}")
                print(f"  Context snippet (truncated): '{feature['context'][:100]}...'")
            if len(features) > 3:
                print(f"  ... and {len(features) - 3} more candidate pairs")
        """
        
        # Pass the confirmed lists to the next function
        test_data = create_prediction_dataset(features, self.vocab, self.device, 
                                              self.pred_max_distance, self.pred_max_context_len)
        
        # Create a dictionary to store all predictions for each entity
        entity_predictions = {}
        self.model.eval()
        
        # Comment out the MODEL PREDICTIONS section
        """
        if self.debug:
            print("\n===== DIAGNOSTIC: MODEL PREDICTIONS =====")
        """
        
        prediction_count = 0
        
        with torch.no_grad():
            for data in test_data:
                # DIAGNOSTIC: Print the input tensors (shapes and values)
                #if self.debug and prediction_count < 3:  # Only print first 3 for brevity
                    #print(f"\nPrediction {prediction_count + 1}:")
                    #print(f"  Context tensor shape: {data['context'].shape}")
                    #print(f"  Distance value: {data['distance'].item()}")
                    #print(f"  Diag_before value: {data['diag_before'].item()}")
                
                output = self.model(data['context'], data['distance'], data['diag_before'])
                prob = output.item()
                
                feature = data['feature']
                diagnosis = feature['diagnosis']
                date = feature['date']
                
                # Comment out the model's prediction prints
                """
                # DIAGNOSTIC: Print the model's prediction for ALL pairs, not just the first 3
                if self.debug:
                    print(f"  Prediction for '{diagnosis}' and '{date}': {prob:.4f}")
                """
                
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
        
        if self.debug:
            print("\n===== DIAGNOSTIC: FINAL RELATIONSHIP SELECTIONS =====")
            # Remove the full note text print since it's now in the comparison section
        
        for entity_key, predictions in entity_predictions.items():
            if predictions:
                # Split the key back into label and category
                entity_label, entity_category = entity_key.split('|', 1)
                
                # Sort predictions by confidence (highest first)
                best_prediction = max(predictions, key=lambda x: x['confidence'])
                
                # DIAGNOSTIC: Print the best prediction for each entity
                if self.debug:
                    print(f"Entity: '{entity_label}' ({entity_category})")
                    print(f"  Best date: '{best_prediction['date']}' with confidence: {best_prediction['confidence']:.4f}")
                    if len(predictions) > 1:
                        print(f"  (Selected from {len(predictions)} candidate dates)")
                
                # Add the best prediction to our results
                relationships.append({
                    'entity_label': entity_label,
                    'entity_category': entity_category,
                    'date': best_prediction['date'],
                    'confidence': best_prediction['confidence']
                })
        
        if self.debug:
            print("\n===== END OF DIAGNOSTIC OUTPUT =====\n")
        
        return relationships 