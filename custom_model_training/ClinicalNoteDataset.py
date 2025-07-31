# Custom dataset for clinical notes
import torch
from torch.utils.data import Dataset

class ClinicalNoteDataset(Dataset):
    def __init__(self, features, labels, vocab, MAX_CONTEXT_LEN, MAX_DISTANCE, entity_category_map=None, tokenizer=None):
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
        
        # Determine the unknown token format used in this vocabulary
        # Check for different possible unknown token formats
        self.unk_token = None
        for unk_format in ['<unk>', '[UNK]', '<UNK>', 'unk', 'UNK']:
            if unk_format in self.vocab.word2idx:
                self.unk_token = unk_format
                break
        
        # If no unknown token found, use the first one as default or raise error
        if self.unk_token is None:
            if hasattr(self.vocab, 'word2idx') and len(self.vocab.word2idx) > 1:
                # Use the second token (usually the unknown token after padding)
                self.unk_token = list(self.vocab.word2idx.keys())[1]
                print(f"Warning: No standard unknown token found. Using '{self.unk_token}' as unknown token.")
            else:
                raise ValueError("Could not find an unknown token in the vocabulary.")
        
        # Store the external tokenizer if provided
        self.tokenizer = tokenizer
        
        # Determine the padding token
        self.pad_token = None
        for pad_format in ['<pad>', '[PAD]', '<PAD>', 'pad', 'PAD']:
            if pad_format in self.vocab.word2idx:
                self.pad_token = pad_format
                break
        
        # If no padding token found, use the first one as default
        if self.pad_token is None:
            if hasattr(self.vocab, 'word2idx') and len(self.vocab.word2idx) > 0:
                self.pad_token = list(self.vocab.word2idx.keys())[0]
                print(f"Warning: No standard padding token found. Using '{self.pad_token}' as padding token.")

    def __len__(self):
        return len(self.labels)
    
    def tokenize(self, text):
        """
        Tokenize text consistently with the vocabulary.
        If an external tokenizer is provided, use it. Otherwise, use simple whitespace tokenization.
        """
        if self.tokenizer:
            # Use the provided tokenizer
            tokens = self.tokenizer.tokenize(text)
            # Convert tokens to indices
            indices = []
            for token in tokens:
                if token in self.vocab.word2idx:
                    indices.append(self.vocab.word2idx[token])
                else:
                    indices.append(self.vocab.word2idx[self.unk_token])
            return indices
        else:
            # Simple whitespace tokenization
            indices = []
            for word in text.split():
                if word in self.vocab.word2idx:
                    indices.append(self.vocab.word2idx[word])
                else:
                    indices.append(self.vocab.word2idx[self.unk_token])
            return indices
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        # Convert text to token indices
        context_indices = self.tokenize(feature['context'])
        
        # Pad or truncate to MAX_CONTEXT_LEN
        if len(context_indices) > self.MAX_CONTEXT_LEN:
            context_indices = context_indices[:self.MAX_CONTEXT_LEN]
        else:
            # Get padding index (usually 0)
            pad_idx = self.vocab.word2idx[self.pad_token] if self.pad_token in self.vocab.word2idx else 0
            padding = [pad_idx] * (self.MAX_CONTEXT_LEN - len(context_indices))
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