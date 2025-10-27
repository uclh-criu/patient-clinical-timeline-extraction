import pandas as pd
import json
import ast

def parse_jsonish(s):
    """Parse a JSON-like string; handles both JSON and Python literal lists/dicts."""
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except Exception:
        return ast.literal_eval(s)

def load_data(file_path):
    """Load CSV data and parse JSON columns"""
    df = pd.read_csv(file_path)
    if 'entities_json' in df.columns:
        df['entities_json'] = df['entities_json'].apply(parse_jsonish)
    if 'dates_json' in df.columns:
        df['dates_json'] = df['dates_json'].apply(parse_jsonish)
    if 'relations_json' in df.columns:
        df['relations_json'] = df['relations_json'].apply(parse_jsonish)

    return df

def prepare_sample(row):
    """Prepare a single row for extractor input"""
    entities_list = row['entities_json']
    dates = row['dates_json']
    note_text = row['note_text']
    return note_text, entities_list, dates

def prepare_all_samples(df):
    """Prepare all samples for batch processing"""
    samples = []
    for _, row in df.iterrows():
        note_text, entities_list, dates = prepare_sample(row)
        
        # Get relative dates if available
        relative_dates = []
        if 'relative_dates_json' in row and pd.notna(row['relative_dates_json']):
            relative_dates = json.loads(row['relative_dates_json'])
        
        samples.append({
            'doc_id': row.get('doc_id'),
            'note_text': note_text,
            'entities_list': entities_list,
            'dates': dates,
            'relative_dates': relative_dates,
            'relations_json': row.get('relations_json', [])
        })
    return samples

def get_entity_date_pairs(entities_list, dates, relative_dates=None):
    """Get all possible entity-date pairs for classification, including relative dates"""
    pairs = []
    
    # Add absolute date pairs
    for entity in entities_list:
        for date_info in dates:
            pairs.append({
                'entity': entity,
                'date_info': date_info,
                'entity_label': entity['value'],
                'entity_preferred_name': entity.get('preferred_name', entity['value']),
                'date': date_info['value'],
                'distance': abs(entity['start'] - date_info['start']),
                'date_type': 'absolute'
            })
    
    # Add relative date pairs if provided
    if relative_dates:
        for entity in entities_list:
            for rel_date in relative_dates:
                pairs.append({
                    'entity': entity,
                    'date_info': rel_date,
                    'entity_label': entity['value'],
                    'entity_preferred_name': entity.get('preferred_name', entity['value']),
                    'date': rel_date['value'],
                    'distance': abs(entity['start'] - rel_date['start']),
                    'date_type': 'relative'
                })
    
    return pairs

def calculate_metrics(all_predictions, df):
    """Calculate overall metrics for all predictions"""
    
    # Convert predictions to (entity, date) pairs
    pred_pairs = set()
    for p in all_predictions:
        pred_pairs.add((p['entity_label'], p['date']))
    
    # Get all gold pairs from dataframe
    gold_pairs = set()
    for _, row in df.iterrows():
        for g in row['relations_json']:
            date = g['date']
            gold_pairs.add((g['entity'], date))
    
    # Calculate metrics
    tp = len(pred_pairs & gold_pairs)
    fp = len(pred_pairs - gold_pairs)
    fn = len(gold_pairs - pred_pairs)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }

# Function to create a boolean filter for a column, adapting to its data type
def create_non_empty_filter(series):
    # Drop NA values to prevent errors and get the first valid type
    series_no_na = series.dropna()
    if series_no_na.empty:
        # If the series is all NaNs, return a series of all False
        return pd.Series([False] * len(series), index=series.index)
    
    first_item_type = type(series_no_na.iloc[0])
    
    if first_item_type == list:
        print(f"'{series.name}' is a list column. Using len().")
        return series.apply(len) > 0
    elif first_item_type == str:
        print(f"'{series.name}' is a string column. Using str.len().")
        # An empty list as a string ('[]') has length 2.
        return series.str.len() > 2
    else:
        print(f"Warning: '{series.name}' has an unexpected type ({first_item_type}). Returning all False.")
        return pd.Series([False] * len(series), index=series.index)