import pandas as pd
import json

def load_data(file_path):
    """Load CSV data and parse JSON columns"""
    df = pd.read_csv(file_path)
    df['extracted_disorders'] = df['extracted_disorders'].apply(json.loads)
    df['formatted_dates'] = df['formatted_dates'].apply(json.loads)
    df['relationship_gold'] = df['relationship_gold'].apply(json.loads)
    return df

def prepare_sample(row):
    """Prepare a single row for extractor input"""
    entities_list = row['extracted_disorders']
    dates = row['formatted_dates']
    note_text = row['note']
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
            'relationship_gold': row['relationship_gold'],  # Add this line!
            'patient_id': row['patient'],
            'note_id': row['note_id']
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
                'entity_label': entity['label'],
                'date': date_info['parsed'],
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
        for g in row['relationship_gold']:
            date = g['date']
            for diagnosis in g['diagnoses']:
                gold_pairs.add((diagnosis['diagnosis'], date))
    
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
