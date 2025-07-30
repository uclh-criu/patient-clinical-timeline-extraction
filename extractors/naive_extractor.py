import os
from extractors.base_extractor import BaseRelationExtractor
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

class NaiveExtractor(BaseRelationExtractor):
    """
    Relation extractor that uses character proximity as the basis for matching.
    
    This is a naive approach that assigns each entity to the closest date
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
    
    def extract(self, text, entities=None, note_id=None, patient_id=None):
        """
        Extract relationships between entities and dates based on proximity.
        
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
                    'confidence': float,     # Always 1.0 for this method
                    'distance': int          # Character distance
                }
        """
        # Entities are required for CSV data processing
        if entities is None:
            print("Error: entities parameter is required for CSV data processing.")
            return []
        
        entities_list, dates = entities
        
        # Find relationships
        relationships = []
        
        # Handle both old format (tuple) and new format (dict)
        for entity in entities_list:
            # Check if entity is in the new dict format or old tuple format
            if isinstance(entity, dict):
                entity_label = entity.get('label', '')
                entity_pos = entity.get('start', 0)
                entity_category = entity.get('category', 'unknown')
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
            
            closest_date = None
            min_distance = float('inf')
            
            # Unpack the 3 elements: parsed_date, raw date_str, position
            for parsed_date, date_str, date_pos in dates:
                distance = abs(entity_pos - date_pos)
                
                if distance < min_distance and distance <= self.max_distance:
                    min_distance = distance
                    closest_date = parsed_date # Use the formatted date
            
            if closest_date:
                relationships.append({
                    'entity_label': entity_label,
                    'entity_category': entity_category,
                    'date': closest_date,
                    'confidence': 1.0,  # Always 1.0 for rule-based
                    'distance': min_distance
                })
        
        return relationships
    
    # Removed redundant evaluate method 