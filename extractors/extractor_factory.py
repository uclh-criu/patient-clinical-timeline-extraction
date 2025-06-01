from extractors.naive_extractor import NaiveExtractor
from extractors.custom_extractor import CustomExtractor
from extractors.relcat_extractor import RelcatExtractor
from extractors.openai_extractor import OpenAIExtractor
from extractors.llama_extractor import LlamaExtractor

def create_extractor(method, config):
    """
    Factory function to create the appropriate relation extractor.
    
    Args:
        method (str): The extraction method to use ('custom', 'naive', 'relcat', 'openai', or 'llama').
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
    elif method == 'openai':
        return OpenAIExtractor(config)
    elif method == 'llama':
        return LlamaExtractor(config)
    else:
        raise ValueError(f"Unknown extraction method: {method}. " +
                        "Valid options are: 'custom', 'naive', 'relcat', 'openai', or 'llama'.") 