"""
BERT training utilities for relation extraction.

This module provides functions for preparing training data, handling class imbalance,
and evaluating BERT models for relation extraction tasks.
"""

import pandas as pd
import torch
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from utils import get_entity_date_pairs
from bert_extractor import preprocess_input


# ============================================================================
# DATA PREPARATION
# ============================================================================

def build_gold_lookup(gold_relations):
    """
    Build position-based lookup for gold relations.
    
    This creates a mapping from (disorder_position, date_position) to "link"
    for robust position-based labeling.
    
    Args:
        gold_relations: List of gold relationship annotations
        
    Returns:
        Dictionary mapping (disorder_pos, date_pos) -> "link"
    """
    gold_map = {}
    for g in gold_relations:
        date_pos = g["date_position"]
        for diag in g.get("diagnoses", []):
            gold_map[(diag["position"], date_pos)] = "link"
    return gold_map


def get_label_for_pair(disorder_start, date_start, gold_map):
    """
    Get label for entity-date pair using position-based matching.
    
    Args:
        disorder_start: Start position of disorder entity
        date_start: Start position of date entity
        gold_map: Position-based lookup dictionary
        
    Returns:
        "link" if pair exists in gold annotations, "no_link" otherwise
    """
    key = (disorder_start, date_start)
    return gold_map.get(key, "no_link")


def create_training_pairs(samples, max_length=256):
    """
    Create training pairs using the best approach: full-text + position-based labeling.
    
    This is the main data preparation function that:
    1. Builds position-based gold lookups for each sample
    2. Creates all disorder-date pairs
    3. Labels pairs using position-based matching
    4. Preprocesses text with entity markers
    5. Adds metadata for analysis
    
    Args:
        samples: List of sample dictionaries from prepare_all_samples()
        max_length: Maximum sequence length for truncation
        
    Returns:
        DataFrame with columns: text, marked_text, ent1_start, ent1_end, 
        ent2_start, ent2_end, label, patient_id, note_id, distance
    """
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


# ============================================================================
# CLASS IMBALANCE HANDLING
# ============================================================================

def compute_class_weights(dataset, num_labels):
    """
    Compute class weights for weighted CrossEntropyLoss.
    
    This calculates inverse frequency weights to handle class imbalance
    while preserving all training data.
    
    Args:
        dataset: Dataset with 'label' column
        num_labels: Number of classes
        
    Returns:
        Tensor of class weights normalized to have mean ~= 1
    """
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
    """
    Downsample negative class to achieve a desired positive:negative ratio.
    
    Alternative to weighted loss for handling class imbalance.
    
    Args:
        df: DataFrame with 'label' column
        ratio: Ratio of negative to positive samples
        random_state: Random seed for reproducibility
        
    Returns:
        Balanced DataFrame
    """
    pos = df[df['label'] == 1]
    neg = df[df['label'] == 0]
    n_neg = int(len(pos) * ratio)
    neg_down = neg.sample(n=min(n_neg, len(neg)), random_state=random_state)
    return pd.concat([pos, neg_down]).sample(frac=1, random_state=random_state).reset_index(drop=True)


def handle_class_imbalance(df, method='weighted', ratio=1.0, random_state=42):
    """
    Handle class imbalance using specified method.
    
    Args:
        df: DataFrame with 'label' column
        method: 'weighted' (use class weights) or 'downsample' (balance classes)
        ratio: For downsampling, ratio of negative to positive samples
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (processed_df, class_weights)
    """
    if method == 'weighted':
        return df, compute_class_weights(df, 2)
    elif method == 'downsample':
        balanced_df = balance_classes(df, ratio, random_state)
        return balanced_df, None
    else:
        raise ValueError("Method must be 'weighted' or 'downsample'")


# ============================================================================
# TOKENIZATION
# ============================================================================

def add_special_tokens(tokenizer):
    """
    Add special tokens for entity and date marking.
    
    Args:
        tokenizer: HuggingFace tokenizer
        
    Returns:
        Updated tokenizer with special tokens
    """
    special_tokens = {'additional_special_tokens': ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer


def tokenize_function(example, tokenizer, max_length=256):
    """
    Tokenize function for datasets.
    
    Args:
        example: Dataset example with 'marked_text' column
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Tokenized example
    """
    return tokenizer(example["marked_text"], truncation=True, padding="max_length", max_length=max_length)


# ============================================================================
# EVALUATION
# ============================================================================

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for training.
    
    Args:
        eval_pred: Tuple of (logits, labels) from evaluation
        
    Returns:
        Dictionary of metrics: accuracy, f1_macro, f1_micro, f1_weighted
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }


# ============================================================================
# LEGACY FUNCTIONS (kept for backward compatibility)
# ============================================================================

def make_training_pairs(samples, gold_lookup, window_size=100, entity_marker='[E]', date_marker='[D]'):
    """
    Legacy function for creating training pairs with windowed approach.
    Use create_training_pairs() for new code.
    """
    rows = []
    for sample in samples:
        pairs = get_entity_date_pairs(sample['entities_list'], sample['dates'])
        for pair in pairs:
            entity = pair['entity']
            date = pair['date_info']
            text = preprocess_input(
                sample['note_text'], entity, date, window_size=window_size
            )
            label = 1 if gold_lookup(sample, pair) else 0
            rows.append({'text': text, 'label': label})
    return pd.DataFrame(rows)


def gold_lookup(sample, pair):
    """
    Legacy string-based gold lookup function.
    Use build_gold_lookup() and get_label_for_pair() for new code.
    """
    for rel in sample['relationship_gold']:
        if rel['date'] == pair['date']:
            for diag in rel['diagnoses']:
                if diag['diagnosis'] == pair['entity_label']:
                    return True
    return False