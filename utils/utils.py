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
    # df['extracted_disorders'] = df['extracted_disorders'].apply(json.loads)
    # df['formatted_dates'] = df['formatted_dates'].apply(json.loads)
    # df['relationship_gold'] = df['relationship_gold'].apply(json.loads)
    if 'entities_json' in df.columns:
        df['entities_json'] = df['entities_json'].apply(parse_jsonish)
    if 'dates_json' in df.columns:
        df['dates_json'] = df['dates_json'].apply(parse_jsonish)
    if 'links_json' in df.columns:
        df['links_json'] = df['links_json'].apply(parse_jsonish)

    return df

def prepare_sample(row):
    """Prepare a single row for extractor input"""
    # entities_list = row['extracted_disorders']
    # dates = row['formatted_dates']
    # note_text = row['note']
    entities_list = row['entities_json']
    dates = row['dates_json']
    note_text = row['note_text']
    return note_text, entities_list, dates

def prepare_all_samples(df):
    """Prepare all samples for batch processing"""
    samples = []
    for _, row in df.iterrows():
        note_text, entities_list, dates = prepare_sample(row)
        samples.append({
            'note_text': note_text,
            'entities_list': entities_list,
            'dates': dates,
            # 'relationship_gold': row['relationship_gold'],
            # 'patient_id': row['patient'],
            # 'note_id': row['note_id']
            'links_json': row.get('links_json', []),
            'doc_id': row.get('doc_id')
        })
    return samples

def get_entity_date_pairs(entities_list, dates):
    """Get all possible entity-date pairs for classification"""
    pairs = []
    for entity in entities_list:
        for date_info in dates:
            pairs.append({
                'entity': entity,
                'date_info': date_info,
                # 'entity_label': entity['label'],
                # 'date': date_info['parsed'],
                'entity_label': entity['value'],
                'date': date_info['value'],
                'distance': abs(entity['start'] - date_info['start'])
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
        # for g in row['relationship_gold']:
        for g in row['links_json']:
            date = g['date']
            # for diagnosis in g['diagnoses']:
            #     gold_pairs.add((diagnosis['diagnosis'], date))
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
