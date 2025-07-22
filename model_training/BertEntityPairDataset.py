# Custom dataset for BERT entity-pair relation extraction
import torch
from torch.utils.data import Dataset

class BertEntityPairDataset(Dataset):
    """Dataset for training BERT on entity-date relation extraction."""
    
    def __init__(self, examples, tokenizer, max_length=512):
        """
        Initialize the dataset.
        
        Args:
            examples: List of dictionaries containing examples
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize the text with special entity markers
        encoding = self.tokenizer(
            example['text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert to the expected format (remove batch dimension)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Create label tensor
        label = torch.tensor(example['label'], dtype=torch.float)
        
        # Optional: Add distance feature
        if 'distance' in example:
            distance = torch.tensor(example['distance'], dtype=torch.float)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'distance': distance,
                'label': label
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': label
            } 