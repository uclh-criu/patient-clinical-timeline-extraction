"""
Utility script to extract vocabulary from a BERT model.
This script loads a BERT tokenizer, extracts its vocabulary,
and saves it to a file that can be used by our custom model.
"""

import os
import sys
import torch
from transformers import AutoTokenizer

# Add project root to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_model_training.training_config_custom import VOCAB_PATH

def extract_bert_vocab(bert_model_name="emilyalsentzer/Bio_ClinicalBERT"):
    """
    Extract vocabulary from a BERT model and save it to a file.
    
    Args:
        bert_model_name (str): Name of the BERT model to use
    """
    print(f"Loading BERT tokenizer: {bert_model_name}")
    
    # Load BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    
    # Get BERT's vocabulary
    bert_vocab = tokenizer.vocab
    bert_vocab_size = len(bert_vocab)
    
    print(f"BERT vocabulary size: {bert_vocab_size}")
    
    # Save the vocabulary
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vocab_full_path = os.path.join(project_root, VOCAB_PATH)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(vocab_full_path), exist_ok=True)
    
    # Save the vocabulary as a dictionary
    torch.save(bert_vocab, vocab_full_path)
    print(f"Vocabulary saved to: {vocab_full_path}")

if __name__ == "__main__":
    extract_bert_vocab()