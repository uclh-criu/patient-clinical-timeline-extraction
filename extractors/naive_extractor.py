import os
from extractors.base_extractor import BaseRelationExtractor
from utils.extraction_utils import extract_entities
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

class NaiveExtractor(BaseRelationExtractor):
    """
    Relation extractor that uses character proximity as the basis for matching.
    
    This is a naive approach that assigns each diagnosis to the closest date
    within a configured maximum distance.
    """
    
    def __init__(self, config):
        """
        Initialize the naive (proximity-based) extractor.
        
        Args:
            config: The configuration object or dict containing PROXIMITY_MAX_DISTANCE parameter.
        """
        self.max_distance = config.PROXIMITY_MAX_DISTANCE if hasattr(config, 'PROXIMITY_MAX_DISTANCE') else 200
        self.name = "Naive (Proximity)"
    
    def load(self):
        """
        Nothing to load for this rule-based extractor.
        """
        return True
    
    def extract(self, text, entities=None):
        """
        Extract relationships between diagnoses and dates based on proximity.
        
        Args:
            text (str): The clinical note text.
            entities (tuple, optional): A tuple of (diagnoses, dates) if already extracted.
                
        Returns:
            list: A list of dictionaries, each representing a relationship:
                {
                    'diagnosis': str,        # The diagnosis text
                    'date': str,             # The date text
                    'confidence': float,     # Always 1.0 for this method
                    'distance': int          # Character distance
                }
        """
        # Extract entities if not provided
        if entities is None:
            diagnoses, dates = extract_entities(text)
        else:
            diagnoses, dates = entities
        
        # Find relationships
        relationships = []
        
        for diagnosis, diag_pos in diagnoses:
            closest_date = None
            min_distance = float('inf')
            
            # Unpack the 3 elements: parsed_date, raw date_str, position
            for parsed_date, date_str, date_pos in dates:
                distance = abs(diag_pos - date_pos)
                
                if distance < min_distance and distance <= self.max_distance:
                    min_distance = distance
                    closest_date = date_str # Use the raw date string found nearby
            
            if closest_date:
                relationships.append({
                    'diagnosis': diagnosis,
                    'date': closest_date,
                    'confidence': 1.0,  # Always 1.0 for rule-based
                    'distance': min_distance
                })
        
        return relationships
    
    # Removed redundant evaluate method 