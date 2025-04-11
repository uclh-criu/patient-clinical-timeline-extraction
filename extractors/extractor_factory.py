from extractors.naive_extractor import NaiveExtractor
from extractors.custom_extractor import CustomExtractor
from extractors.relcat_extractor import RelcatExtractor
from extractors.llm_extractor import LLMExtractor

def create_extractor(method, config):
    """
    Factory function to create the appropriate relation extractor.
    
    Args:
        method (str): The extraction method to use ('custom', 'naive', 'relcat', or 'llm').
        config: The configuration object or dict.
        
    Returns:
        BaseRelationExtractor: An instance of the requested extractor class.
        
    Raises:
        ValueError: If the method is not recognized.
    """
    method = method.lower()
    
    if method == 'custom':
        return CustomExtractor(config)
    elif method == 'naive':
        return NaiveExtractor(config)
    elif method == 'relcat':
        return RelcatExtractor(config)
    elif method == 'llm':
        return LLMExtractor(config)
    else:
        raise ValueError(f"Unknown extraction method: {method}. " +
                        "Valid options are: 'custom', 'naive', 'relcat', or 'llm'.") 