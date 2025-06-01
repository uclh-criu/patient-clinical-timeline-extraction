import os
import torch
from extractors.base_extractor import BaseRelationExtractor
from model_training.DiagnosisDateRelationModel import DiagnosisDateRelationModel
from model_training.Vocabulary import Vocabulary
from model_training.training_config import EMBEDDING_DIM, HIDDEN_DIM
from utils.training_utils import preprocess_note_for_prediction, create_prediction_dataset

class CustomExtractor(BaseRelationExtractor):
    """
    Relation extractor that uses the custom-trained PyTorch neural network 
    model (`DiagnosisDateRelationModel`) to identify diagnosis-date relationships.
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
            with torch.serialization.safe_globals([Vocabulary]):
                self.vocab = torch.load(self.vocab_path, weights_only=False)
            
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
    
    def extract(self, text, entities=None):
        """
        Extract relationships using the custom PyTorch model.
        
        Args:
            text (str): The clinical note text.
            entities (tuple, optional): A tuple of (diagnoses, dates) if already extracted.
            
        Returns:
            list: A list of dictionaries, each representing a relationship:
                {
                    'diagnosis': str,      # The diagnosis text
                    'date': str,           # The date text
                    'confidence': float    # Model prediction confidence
                }
        """
        if self.model is None or self.vocab is None:
            print("Custom Model or Vocabulary not loaded. Call load() first.")
            return []
        
        if entities is None:
            print("Error: entities parameter is required for CSV data processing.")
            return []
        
        diagnoses, dates = entities
        
        features = preprocess_note_for_prediction(text, self.pred_max_distance)
        # Pass the confirmed lists to the next function
        test_data = create_prediction_dataset(features, self.vocab, self.device, 
                                              self.pred_max_distance, self.pred_max_context_len)
        
        # Create a dictionary to store all predictions for each diagnosis
        diagnosis_predictions = {}
        self.model.eval()
        
        with torch.no_grad():
            for data in test_data:
                output = self.model(data['context'], data['distance'], data['diag_before'])
                prob = output.item()
                
                feature = data['feature']
                diagnosis = feature['diagnosis']
                date = feature['date']
                
                # Store this prediction for the diagnosis
                if diagnosis not in diagnosis_predictions:
                    diagnosis_predictions[diagnosis] = []
                
                diagnosis_predictions[diagnosis].append({
                    'date': date,
                    'confidence': prob
                })
        
        # For each diagnosis, select the date with the highest confidence
        relationships = []
        for diagnosis, predictions in diagnosis_predictions.items():
            if predictions:
                # Sort predictions by confidence (highest first)
                best_prediction = max(predictions, key=lambda x: x['confidence'])
                
                # Add the best prediction to our results
                relationships.append({
                    'diagnosis': diagnosis,
                    'date': best_prediction['date'],
                    'confidence': best_prediction['confidence']
                })
        
        return relationships 