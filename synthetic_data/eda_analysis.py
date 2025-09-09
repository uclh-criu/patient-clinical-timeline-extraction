import pandas as pd
import numpy as np
from collections import Counter

def get_doc_length_stats(df, note_col='note'):
    """
    Compute statistics on document (note) lengths.
    Returns dict with mean, std, min, max, median.
    """
    lengths = df[note_col].str.len()
    return {
        'mean': lengths.mean(),
        'std': lengths.std(),
        'min': lengths.min(),
        'max': lengths.max(),
        'median': lengths.median()
    }

def get_entity_count_stats(df, entity_col='extracted_disorders'):
    """
    Compute statistics on the number of entities per note.
    Returns dict with mean, std, min, max, median.
    """
    counts = df[entity_col].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
    return {
        'mean': counts.mean(),
        'std': counts.std(),
        'min': counts.min(),
        'max': counts.max(),
        'median': counts.median()
    }

def get_entity_frequency(df, entity_col='extracted_disorders'):
    """
    Count frequency of each entity label in the dataset.
    Returns dict: {entity_label: count}
    """
    all_entities = []
    for x in df[entity_col].dropna():
        all_entities.extend([d['label'] for d in eval(x)])
    return dict(Counter(all_entities))

def get_date_count_stats(df, date_col='formatted_dates'):
    """
    Compute statistics on the number of dates per note.
    Returns dict with mean, std, min, max, median.
    """
    counts = df[date_col].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
    return {
        'mean': counts.mean(),
        'std': counts.std(),
        'min': counts.min(),
        'max': counts.max(),
        'median': counts.median()
    }