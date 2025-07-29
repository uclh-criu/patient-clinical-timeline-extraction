# Custom dataset for BERT entity-pair relation extraction
import torch
from torch.utils.data import Dataset

class BertEntityPairDataset(Dataset):
    """
    Dataset for BERT entity pair classification.
    Each example is a pair of entities (entity and date) with a binary label
    indicating whether they are related.
    """
    
    def __init__(self, examples, tokenizer, max_length=512):
        """
        Initialize the dataset.
        
        Args:
            examples (list): List of dictionaries containing:
                - text (str): The text with special tokens marking entities
                - label (int): 1 if the entities are related, 0 otherwise
                - entity_category (str, optional): Category of the entity (diagnosis, symptom, etc.)
            tokenizer: The BERT tokenizer
            max_length (int): Maximum sequence length
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Entity category mapping
        self.entity_category_map = {
            'diagnosis': 0,
            'symptom': 1,
            'procedure': 2,
            'medication': 3
        }
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """
        Get a single example from the dataset.
        
        Args:
            idx (int): Index of the example
            
        Returns:
            dict: Dictionary containing the tokenized input, label, and entity category
        """
        example = self.examples[idx]
        text = example['text']
        label = example['label']
        
        # Get entity category if available (default to 'diagnosis' if not)
        entity_category = example.get('entity_category', 'diagnosis')
        entity_category_id = self.entity_category_map.get(entity_category, 0)  # Default to diagnosis (0)
        
        # Tokenize the text with special tokens
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Remove the batch dimension added by the tokenizer
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Add the label and entity category
        encoding['label'] = torch.tensor(label, dtype=torch.float)
        encoding['entity_category'] = torch.tensor(entity_category_id, dtype=torch.long)
        
        return encoding
    
    @classmethod
    def create_from_pairs(cls, entity_pairs, tokenizer, max_length=512):
        """
        Create a dataset from a list of entity pairs.
        
        Args:
            entity_pairs (list): List of dictionaries containing:
                - text (str): The original text
                - entity1 (dict): First entity with 'text', 'start', 'end', and optional 'category'
                - entity2 (dict): Second entity with 'text', 'start', 'end'
                - label (int): 1 if the entities are related, 0 otherwise
            tokenizer: The BERT tokenizer
            max_length (int): Maximum sequence length
            
        Returns:
            BertEntityPairDataset: The created dataset
        """
        examples = []
        
        for pair in entity_pairs:
            text = pair['text']
            entity1 = pair['entity1']
            entity2 = pair['entity2']
            label = pair['label']
            
            # Get entity category if available
            entity_category = entity1.get('category', 'diagnosis').lower()
            
            # Insert special tokens to mark the entities
            # We need to handle the case where entity1 comes before entity2 or vice versa
            if entity1['start'] < entity2['start']:
                # Entity1 comes first
                marked_text = (
                    text[:entity1['start']] + 
                    "[E1]" + text[entity1['start']:entity1['end']] + "[/E1]" + 
                    text[entity1['end']:entity2['start']] + 
                    "[E2]" + text[entity2['start']:entity2['end']] + "[/E2]" + 
                    text[entity2['end']:]
                )
            else:
                # Entity2 comes first
                marked_text = (
                    text[:entity2['start']] + 
                    "[E2]" + text[entity2['start']:entity2['end']] + "[/E2]" + 
                    text[entity2['end']:entity1['start']] + 
                    "[E1]" + text[entity1['start']:entity1['end']] + "[/E1]" + 
                    text[entity1['end']:]
                )
            
            examples.append({
                'text': marked_text,
                'label': label,
                'entity_category': entity_category
            })
        
        return cls(examples, tokenizer, max_length) 