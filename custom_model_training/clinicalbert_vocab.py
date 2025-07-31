from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
tokens = tokenizer("Patient has hypertension since 2010.", return_tensors="pt")

torch.save(tokenizer.get_vocab(), "custom_model_training/vocabs/clinicalbert_vocab.pt")