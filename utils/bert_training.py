import pandas as pd
import torch
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from utils import get_entity_date_pairs
from bert_extractor import preprocess_input

def build_gold_lookup(gold_relations):
    gold_map = {}
    for g in gold_relations:
        date_pos = g["date_position"]
        for diag in g.get("diagnoses", []):
            gold_map[(diag["position"], date_pos)] = "link"
    return gold_map

def get_label_for_pair(disorder_start, date_start, gold_map):
    key = (disorder_start, date_start)
    return gold_map.get(key, "no_link")

def create_training_pairs(samples, max_length=256):
    all_samples = []
    
    for sample in samples:
        # Build gold lookup for position-based labeling
        gold_map = build_gold_lookup(sample['relationship_gold'])
        
        # Get all disorder-date pairs
        for disorder in sample['entities_list']:
            for date in sample['dates']:
                # Get label using position-based method
                label_str = get_label_for_pair(disorder['start'], date['start'], gold_map)
                # Convert string label to integer
                label = 1 if label_str == 'link' else 0
                
                # Preprocess text
                processed = preprocess_input(sample['note_text'], disorder, date)
                processed['label'] = label
                
                # Add metadata for analysis
                processed['patient_id'] = sample.get('patient_id', '')
                processed['note_id'] = sample.get('note_id', '')
                processed['distance'] = abs(disorder['start'] - date['start'])
                
                all_samples.append(processed)
    
    return pd.DataFrame(all_samples)

def compute_class_weights(dataset, num_labels):
    if len(dataset) == 0:
        return torch.ones(num_labels)
    
    counts = Counter([int(x) for x in dataset['label']])
    total = sum(counts.values())
    weights = []
    
    for i in range(num_labels):
        c = max(1, counts.get(i, 0))
        w = total / (num_labels * c)
        weights.append(w)
    
    # Normalize so mean weight ~= 1
    mean_w = sum(weights) / len(weights)
    weights = [w / mean_w for w in weights]
    return torch.tensor(weights, dtype=torch.float)

def balance_classes(df, ratio=1.0, random_state=42):
    pos = df[df['label'] == 1]
    neg = df[df['label'] == 0]
    n_neg = int(len(pos) * ratio)
    neg_down = neg.sample(n=min(n_neg, len(neg)), random_state=random_state)
    return pd.concat([pos, neg_down]).sample(frac=1, random_state=random_state).reset_index(drop=True)

def handle_class_imbalance(df, method='weighted', ratio=1.0, random_state=42):
    if method == 'weighted':
        return df, compute_class_weights(df, 2)
    elif method == 'downsample':
        balanced_df = balance_classes(df, ratio, random_state)
        return balanced_df, None
    else:
        raise ValueError("Method must be 'weighted' or 'downsample'")

def add_special_tokens(tokenizer):
    special_tokens = {'additional_special_tokens': ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer

def tokenize_function(example, tokenizer, max_length=256):
    return tokenizer(example["marked_text"], truncation=True, padding="max_length", max_length=max_length)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }