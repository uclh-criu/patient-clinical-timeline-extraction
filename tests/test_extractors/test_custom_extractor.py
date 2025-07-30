import os
import sys
import torch

# Add project root to path to allow importing from other directories
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from extractors.custom_extractor import CustomExtractor
import config

# Create a simple config object
class TestConfig:
    def __init__(self):
        self.MODEL_PATH = 'custom_model_training/best_model.pt'
        self.VOCAB_PATH = 'custom_model_training/vocab.pt'
        self.PREDICTION_MAX_DISTANCE = 500
        self.PREDICTION_MAX_CONTEXT_LEN = 512
        self.DEVICE = torch.device('cpu')
        self.MODEL_DEBUG_MODE = True

# Create extractor and try to load it
extractor = CustomExtractor(TestConfig())
success = extractor.load()

print(f"Model loaded successfully: {success}")
if success:
    print(f"Using threshold: {extractor.threshold}")
    print(f"Model parameters:")
    if extractor.model_config:
        for key, value in extractor.model_config.items():
            print(f"  {key}: {value}")
    else:
        print("  No model config available")
        
    # Test inference with a simple example
    print("\nTesting inference with a simple example:")
    test_text = "Patient was diagnosed with pituitary adenoma on January 15, 2023. Follow-up scheduled for March 2023."
    
    # Create test entities in the format expected by the extractor
    entities_list = [
        {'label': 'pituitary adenoma', 'start': 28, 'end': 45, 'category': 'disorder'},
        {'label': 'follow-up', 'start': 47, 'end': 56, 'category': 'procedure'}
    ]
    dates = ['January 15, 2023', 'March 2023']
    test_entities = (entities_list, dates)
    
    relationships = extractor.extract(test_text, test_entities)
    
    print(f"\nExtracted {len(relationships)} relationships:")
    for rel in relationships:
        print(f"  {rel['entity_label']} ({rel['entity_category']}) -> {rel['date']} (confidence: {rel['confidence']:.4f})")