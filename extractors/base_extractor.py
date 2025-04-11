from abc import ABC, abstractmethod

class BaseRelationExtractor(ABC):
    """
    Abstract base class for all relation extraction methods.
    
    Any class that implements this interface should be able to extract
    relationships between diagnoses and dates in clinical notes.
    """
    
    @abstractmethod
    def extract(self, text, entities=None):
        """
        Extract relationships between diagnoses and dates in the text.
        
        Args:
            text (str): The clinical note text.
            entities (tuple, optional): A tuple of (diagnoses, dates) if already extracted.
                If None, the implementation should extract entities itself.
                
        Returns:
            list: A list of dictionaries, each representing a relationship:
                {
                    'diagnosis': str,  # The diagnosis text
                    'date': str,       # The date text
                    'confidence': float # A confidence score between 0 and 1
                }
        """
        pass
    
    @abstractmethod
    def load(self):
        """
        Load any necessary models or resources.
        
        Returns:
            bool: True if successfully loaded, False otherwise.
        """
        pass