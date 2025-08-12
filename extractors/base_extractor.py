from abc import ABC, abstractmethod

class BaseRelationExtractor(ABC):
    """
    Abstract base class for all relation extraction methods.
    
    Any class that implements this interface should be able to extract
    relationships between medical entities and dates in clinical notes.
    """
    
    def __init__(self, config):
        """
        Initialize the base extractor with a configuration object.
        
        Args:
            config: The configuration object for the run.
        """
        self.config = config
        self.name = "Base Extractor"

    @abstractmethod
    def extract(self, text, entities=None):
        """
        Extract relationships between medical entities and dates in the text.
        
        Args:
            text (str): The clinical note text.
            entities (tuple, optional): A tuple of (entities_list, dates) if already extracted.
                If None, the implementation should extract entities itself.
                
        Returns:
            list: A list of dictionaries, each representing a relationship:
                {
                    'entity_label': str,     # The entity text
                    'entity_category': str,  # The entity category
                    'date': str,             # The date text
                    'confidence': float      # A confidence score between 0 and 1
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