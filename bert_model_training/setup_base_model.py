#!/usr/bin/env python3
"""
Setup script to download the base BERT model for isolated environments.

This script downloads the pre-trained BioBERT model from Hugging Face Hub
and saves it locally for use in environments without internet access.

Run this script when you have internet access, before deploying to an isolated environment.
"""

import os
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def setup_base_model():
    """Download and save the base BERT model locally."""
    
    # Model identifier from Hugging Face Hub
    model_name = 'dmis-lab/biobert-base-cased-v1.1'
    
    # Local directory to save the model
    local_model_dir = './bert_model_training/base_model'
    
    print(f"Setting up base BERT model for isolated environments...")
    print(f"Model: {model_name}")
    print(f"Local directory: {local_model_dir}")
    
    try:
        # Create the directory if it doesn't exist
        os.makedirs(local_model_dir, exist_ok=True)
        
        print("\nDownloading model...")
        # Download and save the model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=1  # Binary classification setup
        )
        model.save_pretrained(local_model_dir)
        print("✓ Model downloaded and saved")
        
        print("\nDownloading tokenizer...")
        # Download and save the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(local_model_dir)
        print("✓ Tokenizer downloaded and saved")
        
        print(f"\n✓ Setup complete! Base model saved to: {local_model_dir}")
        print("\nFiles created:")
        for file in os.listdir(local_model_dir):
            print(f"  - {file}")
        
        print(f"\nYou can now deploy this project to isolated environments.")
        print(f"The training script will use the local model instead of downloading from Hugging Face Hub.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("Make sure you have internet access and the transformers library installed.")
        return False

def check_if_model_exists():
    """Check if the base model already exists locally."""
    local_model_dir = './bert_model_training/base_model'
    
    if os.path.exists(local_model_dir):
        files = os.listdir(local_model_dir)
        required_files = ['config.json', 'tokenizer_config.json']
        has_model = any(f.startswith('model.') or f.startswith('pytorch_model.') for f in files)
        has_config = all(f in files for f in required_files)
        
        if has_model and has_config:
            print(f"✓ Base model already exists at: {local_model_dir}")
            print("Files found:")
            for file in sorted(files):
                print(f"  - {file}")
            return True
    
    return False

def main():
    """Main function."""
    print("=== BERT Base Model Setup for Isolated Environments ===\n")
    
    # Check if model already exists
    if check_if_model_exists():
        response = input("\nBase model already exists. Download again? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Keeping existing base model.")
            return
    
    # Download the model
    success = setup_base_model()
    
    if success:
        print("\n=== Next Steps ===")
        print("1. Deploy this entire project (including bert_model_training/base_model/) to your isolated environment")
        print("2. In isolated environment, you can:")
        print("   - Train new models: python bert_model_training/train_bert_model.py")
        print("   - Use existing fine-tuned models for inference")
        print("3. No internet connection required in isolated environment!")
    else:
        print("\n=== Troubleshooting ===")
        print("- Ensure you have internet access")
        print("- Install required packages: pip install transformers torch")
        print("- Try running again")

if __name__ == "__main__":
    main()
