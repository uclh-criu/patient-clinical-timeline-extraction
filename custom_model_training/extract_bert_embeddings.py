"""
Utility script to extract pre-trained embeddings from a BERT model.
This script loads a BERT model, extracts its token embeddings,
and saves them to a file that can be used by our custom model.
"""

import os
import sys
import torch
from transformers import AutoModel, AutoTokenizer

# Add project root to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_model_training.Vocabulary import Vocabulary
from custom_model_training.training_config_custom import VOCAB_PATH, PRETRAINED_EMBEDDINGS_PATH, EMBEDDING_DIM

def extract_bert_embeddings(bert_model_name="emilyalsentzer/Bio_ClinicalBERT", embedding_dim=128):
    """
    Extract token embeddings from a BERT model and save them to a file.
    
    Args:
        bert_model_name (str): Name of the BERT model to use
        embedding_dim (int): Dimension of the embeddings to extract
    """
    print(f"Loading BERT model: {bert_model_name}")
    
    # Load BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    model = AutoModel.from_pretrained(bert_model_name)
    
    # Get BERT's vocabulary size and embedding dimension
    bert_vocab_size = len(tokenizer.vocab)
    bert_embedding_dim = model.embeddings.word_embeddings.weight.shape[1]
    
    print(f"BERT vocabulary size: {bert_vocab_size}")
    print(f"BERT embedding dimension: {bert_embedding_dim}")
    
    # Get BERT's token embeddings
    bert_embeddings = model.embeddings.word_embeddings.weight.data
    
    # Load our custom vocabulary
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vocab_full_path = os.path.join(project_root, VOCAB_PATH)
    
    print(f"Loading custom vocabulary from: {vocab_full_path}")
    try:
        # Handle different PyTorch versions
        try:
            loaded_data = torch.load(vocab_full_path, weights_only=False)
        except TypeError:
            loaded_data = torch.load(vocab_full_path)
        
        # Check the type of loaded data and handle accordingly
        if isinstance(loaded_data, dict):
            print("Loaded a dictionary-style vocabulary, converting to Vocabulary object...")
            vocab = Vocabulary()
            
            # Reset the vocabulary to empty
            vocab.word2idx = {}
            vocab.idx2word = {}
            vocab.n_words = 0
            
            # Add special tokens first to ensure they have the expected indices
            special_tokens = ['<pad>', '<unk>', '<cls>', '<sep>', '<mask>']
            for token in special_tokens:
                if token.lower() in loaded_data:
                    vocab.add_word(token.lower())
            
            # Add all other tokens
            for word in loaded_data:
                if word not in vocab.word2idx:
                    vocab.add_word(word)
        elif hasattr(loaded_data, 'n_words'):
            # It's already a Vocabulary object
            vocab = loaded_data
        else:
            raise TypeError("Loaded data is neither a dictionary nor a Vocabulary object")
            
        print(f"Vocabulary loaded successfully. Size: {vocab.n_words} words")
    except Exception as e:
        print(f"Error loading vocabulary: {e}")
        return
    
    # Create a new embedding matrix for our custom vocabulary
    custom_embeddings = torch.randn(vocab.n_words, embedding_dim)
    
    # Initialize with BERT embeddings where possible
    bert_to_custom_map = {}
    for word, idx in vocab.word2idx.items():
        # Handle special tokens
        if word == '<pad>':
            bert_word = '[PAD]'
        elif word == '<unk>':
            bert_word = '[UNK]'
        elif word == '<cls>':
            bert_word = '[CLS]'
        elif word == '<sep>':
            bert_word = '[SEP]'
        elif word == '<mask>':
            bert_word = '[MASK]'
        else:
            bert_word = word
            
        # Check if the word exists in BERT's vocabulary
        if bert_word in tokenizer.vocab:
            bert_idx = tokenizer.vocab[bert_word]
            bert_to_custom_map[idx] = bert_idx
    
    # Count how many words were found in BERT's vocabulary
    found_count = len(bert_to_custom_map)
    print(f"Found {found_count} out of {vocab.n_words} words in BERT's vocabulary ({found_count/vocab.n_words*100:.2f}%)")
    
    # Copy BERT embeddings to our custom embedding matrix
    for custom_idx, bert_idx in bert_to_custom_map.items():
        # Get the BERT embedding for this token
        bert_embedding = bert_embeddings[bert_idx]
        
        # If BERT's embedding dimension matches our target dimension, use it directly
        if bert_embedding_dim == embedding_dim:
            custom_embeddings[custom_idx] = bert_embedding
        else:
            # Otherwise, we need to project it to our target dimension
            # For simplicity, we'll just take the first embedding_dim dimensions
            # In a real application, you might want to use a learned projection
            if bert_embedding_dim > embedding_dim:
                custom_embeddings[custom_idx] = bert_embedding[:embedding_dim]
            else:
                # If BERT's dimension is smaller, pad with zeros
                custom_embeddings[custom_idx, :bert_embedding_dim] = bert_embedding
    
    # Save the embeddings
    embeddings_full_path = os.path.join(project_root, PRETRAINED_EMBEDDINGS_PATH)
    os.makedirs(os.path.dirname(embeddings_full_path), exist_ok=True)
    torch.save(custom_embeddings, embeddings_full_path)
    print(f"Embeddings saved to: {embeddings_full_path}")

if __name__ == "__main__":
    # Get embedding dimension from config if available
    extract_bert_embeddings(embedding_dim=EMBEDDING_DIM)