# Custom dataset for clinical notes
import torch
from torch.utils.data import Dataset

class ClinicalNoteDataset(Dataset):
    def __init__(self, features, labels, vocab, MAX_CONTEXT_LEN, MAX_DISTANCE, entity_category_map=None):
        self.features = features
        self.labels = labels
        self.vocab = vocab
        self.MAX_CONTEXT_LEN = MAX_CONTEXT_LEN
        self.MAX_DISTANCE = MAX_DISTANCE
        
        # Entity category mapping (for multi-entity mode)
        # Default mapping if none provided - only the four specific categories
        self.entity_category_map = entity_category_map or {
            'diagnosis': 0,
            'symptom': 1,
            'procedure': 2,
            'medication': 3
        }

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        # Convert words to indices
        context_indices = []
        for word in feature['context'].split():
            if word in self.vocab.word2idx:
                context_indices.append(self.vocab.word2idx[word])
            else:
                context_indices.append(self.vocab.word2idx['<unk>'])
        
        # Pad or truncate to MAX_CONTEXT_LEN
        if len(context_indices) > self.MAX_CONTEXT_LEN:
            context_indices = context_indices[:self.MAX_CONTEXT_LEN]
        else:
            padding = [0] * (self.MAX_CONTEXT_LEN - len(context_indices))
            context_indices.extend(padding)
        
        # Additional features
        distance = min(feature['distance'] / self.MAX_DISTANCE, 1.0)  # Normalize
        diag_before = feature['diag_before_date']
        
        # Get entity category ID (default to 0 for diagnosis if not specified)
        entity_category = feature.get('entity_category', 'diagnosis').lower()
        entity_category_id = self.entity_category_map.get(entity_category, 0)  # Default to diagnosis (0) if not in map
        
        return {
            'context': torch.tensor(context_indices, dtype=torch.long),
            'distance': torch.tensor(distance, dtype=torch.float),
            'diag_before': torch.tensor(diag_before, dtype=torch.float),
            'entity_category': torch.tensor(entity_category_id, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float)
        }