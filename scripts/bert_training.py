import pandas as pd
from sklearn.model_selection import train_test_split
from utils import get_entity_date_pairs
from bert_extractor import preprocess_input

def tokenize_function(example, tokenizer):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

def add_special_tokens(tokenizer, entity_marker='[E]', date_marker='[D]'):
    """
    Adds special tokens for entity and date marking to the tokenizer.
    """
    special_tokens = {'additional_special_tokens': [entity_marker, date_marker]}
    tokenizer.add_special_tokens(special_tokens)

def make_training_pairs(samples, gold_lookup, window_size=100, entity_marker='[E]', date_marker='[D]'):
    """
    Prepares training pairs using context window and entity/date marking.
    Returns: DataFrame with columns ['text', 'label']
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

def balance_classes(df, ratio=1.0, random_state=42):
    """
    Downsamples negative class to achieve a desired positive:negative ratio.
    """
    pos = df[df['label'] == 1]
    neg = df[df['label'] == 0]
    n_neg = int(len(pos) * ratio)
    neg_down = neg.sample(n=min(n_neg, len(neg)), random_state=random_state)
    return pd.concat([pos, neg_down]).sample(frac=1, random_state=random_state).reset_index(drop=True)

def gold_lookup(sample, pair):
    # sample['relationship_gold'] is a list of dicts with 'date' and 'diagnoses'
    for rel in sample['relationship_gold']:
        if rel['date'] == pair['date']:
            for diag in rel['diagnoses']:
                if diag['diagnosis'] == pair['entity_label']:
                    return True
    return False