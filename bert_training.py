import pandas as pd
from utils import get_entity_date_pairs

def make_training_pairs(samples, gold_lookup):
    """
    samples: list of dicts, each with 'note_text', 'entities_list', 'dates', etc.
    gold_lookup: function or dict to check if (entity, date) is a true relationship
    Returns: DataFrame with columns ['text', 'label']
    """
    rows = []
    for sample in samples:
        pairs = get_entity_date_pairs(sample['entities_list'], sample['dates'])
        for pair in pairs:
            # Compose input text for BERT
            text = f"{pair['entity_label']} [SEP] {pair['date']} [SEP] {sample['note_text'][:200]}"
            # Determine label (1 if in gold, else 0)
            label = 1 if gold_lookup(sample, pair) else 0
            rows.append({'text': text, 'label': label})
    return pd.DataFrame(rows)

def gold_lookup(sample, pair):
    # sample['relationship_gold'] is a list of dicts with 'date' and 'diagnoses'
    for rel in sample['relationship_gold']:
        if rel['date'] == pair['date']:
            for diag in rel['diagnoses']:
                if diag['diagnosis'] == pair['entity_label']:
                    return True
    return False
