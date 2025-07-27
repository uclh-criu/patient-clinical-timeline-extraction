import os
import sys
import json
import torch
import pandas as pd
import re

# Adjust relative paths for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import from our modules
from custom_model_training.Vocabulary import Vocabulary
import custom_model_training.training_config_custom as training_config
from utils.inference_eval_utils import preprocess_text
from config import VOCAB_PATH

def build_vocabulary():
    """
    Build a vocabulary from the data specified in TRAINING_DATA_PATH in training_config.py
    and save it to the path specified in config.py.
    """
    print("Building vocabulary from training data source...")
    
    # Ensure the output directory exists
    vocab_full_path = os.path.join(project_root, VOCAB_PATH)
    os.makedirs(os.path.dirname(vocab_full_path), exist_ok=True)
    
    # Load the vocabulary data
    vocab_data_path = os.path.join(project_root, training_config.TRAINING_DATA_PATH)
    
    if not os.path.exists(vocab_data_path):
        print(f"Error: Training data file not found at {vocab_data_path}")
        sys.exit(1)
    
    print(f"Loading training data from: {vocab_data_path}")
    df = pd.read_csv(vocab_data_path)
    
    if 'note' not in df.columns:
        print("Error: CSV file does not contain a 'note' column")
        sys.exit(1)
    
    # Create a new vocabulary
    vocab = Vocabulary()
    
    # Process each note in the dataset
    print(f"Processing {len(df)} notes to build vocabulary...")
    
    for i, row in df.iterrows():
        note_text = row['note']
        
        # Preprocess the text (lowercase, remove punctuation, etc.)
        processed_text = preprocess_text(note_text)
        
        # Add all words to the vocabulary
        vocab.add_sentence(processed_text)
        
        # Print progress every 100 notes
        if (i + 1) % 100 == 0 or i == len(df) - 1:
            print(f"Processed {i+1}/{len(df)} notes. Current vocabulary size: {vocab.n_words} words")
    
    # Save the vocabulary
    torch.save(vocab, vocab_full_path)
    print(f"Vocabulary built successfully with {vocab.n_words} unique words")
    print(f"Saved vocabulary to: {vocab_full_path}")
    
    # Print some statistics about the vocabulary
    print("\nVocabulary Statistics:")
    print(f"Total unique words: {vocab.n_words}")
    
    # Print the most common words (top 20)
    sorted_words = sorted(vocab.word_count.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 20 most common words:")
    for word, count in sorted_words[:20]:
        print(f"  '{word}': {count} occurrences")

if __name__ == "__main__":
    build_vocabulary() 