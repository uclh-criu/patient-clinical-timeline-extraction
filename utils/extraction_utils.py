# utils/common_utils.py
import re
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import json

# Parse date string (Moved from data_preparation)
def parse_date_string(date_str):
    """
    Parse a date string in various formats and return a standard YYYY-MM-DD format.
    Returns None if parsing fails.
    """
    date_str = date_str.strip()
    
    try:
        # Try common formats using datetime.strptime
        formats = [
            # DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY
            '%d-%m-%Y', '%d/%m/%Y', '%d.%m.%Y',
            # DD-MM-YY, DD/MM/YY, DD.MM.YY
            '%d-%m-%y', '%d/%m/%y', '%d.%m.%y',
            # YYYY-MM-DD
            '%Y-%m-%d',
            # Month names
            '%d %b %Y', '%d %B %Y',
            '%d %b\'%y', '%dst %b %Y', '%dnd %b %Y', '%drd %b %Y', '%dth %b %Y'
        ]
        
        for fmt in formats:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                continue
                
        # If still not parsed, try more complex regex patterns
        # 1. Match patterns like "3rd Feb'23" or "3rd February 2023"
        match = re.search(r'(\d+)(?:st|nd|rd|th)?\s+([a-zA-Z]+)[\']*\s*[\']*(\d{2,4})', date_str)
        if match:
            day, month_str, year = match.groups()
            
            month_names = {
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
            }
            month = month_names.get(month_str.lower()[:3])
            if month:
                if len(year) == 2: year = '20' + year
                date_obj = datetime(int(year), month, int(day))
                return date_obj.strftime('%Y-%m-%d')
        
        # 2. Try to match date with time like "02/02/2023 @1445"
        match = re.search(r'(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})\s*[@at]*\s*\d+', date_str)
        if match:
            # Recursive call to handle the extracted date part
            return parse_date_string(match.group(1))
            
        return None
    except Exception:
        return None

# Extract diagnoses and dates with their positions from text
def extract_entities(text):
    """
    Extract underscore-formatted diagnoses and parenthesized dates with their positions.
    Parses dates into YYYY-MM-DD format.
    Returns diagnoses as [(diag_str, pos)] and dates as [(parsed_date_str, raw_date_str, pos)].
    """
    diagnoses = []
    # Regex captures one or more word characters (alphanumeric + underscore)
    # followed by [dx], [diagnosis], or [diagno sis]
    for match in re.finditer(r'(\w+)\[(?:dx|diagnosis|diagno\s*sis)\]', text):
        diagnosis = match.group(1).lower() # Capture group 1 (the diagnosis), ensure lowercase
        position = match.start(1) # Start position of the captured diagnosis word
        diagnoses.append((diagnosis, position))

    dates = []
    # Regex to capture content within parentheses followed by [date]
    for match in re.finditer(r'\(([^)]+)\)\[date\]', text):
        raw_date_str = match.group(1).strip()
        position = match.start(1) # Position of the content inside parentheses
        
        # Parse the date string here
        parsed_date = parse_date_string(raw_date_str)
        
        # Only add if parsing was successful
        if parsed_date:
            dates.append((parsed_date, raw_date_str, position))

    return diagnoses, dates

# Helper function to load dataset, select samples, prepare gold standard, and extract entities
def load_and_prepare_data(dataset_path, num_samples):
    """
    Loads dataset, selects samples, prepares gold standard, and pre-extracts entities.
    Uses the last 20% of the dataset for evaluation to avoid overlap with training.

    Args:
        dataset_path (str): Path to the JSON dataset file.
        num_samples (int): Maximum number of samples to use (if provided).

    Returns:
        tuple: (prepared_test_data, gold_standard) or (None, None) if loading fails.
               prepared_test_data is a list of dicts {'text': ..., 'entities': ...}.
               gold_standard is a list of dicts {'note_id': ..., 'diagnosis': ..., 'date': ...}.
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return None, None

    print(f"Loading dataset from {dataset_path}...")
    try:
        with open(dataset_path, 'r') as f:
            full_dataset = json.load(f)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

    # Calculate the 80/20 split point
    split_point = int(len(full_dataset) * 0.8)
    
    # Use the last 20% of the dataset for evaluation
    dataset = full_dataset[split_point:]
    print(f"Using last {len(dataset)}/{len(full_dataset)} samples (20%) for evaluation.")
    
    # If a specific number of samples is requested, limit to that
    if num_samples and num_samples < len(dataset):
        test_data = dataset[:num_samples]
        print(f"Limiting to {num_samples} evaluation samples.")
    else:
        test_data = dataset
        print(f"Using all {len(test_data)} evaluation samples.")

    # Prepare gold standard list
    gold_standard = []
    for i, entry in enumerate(test_data):
        # Check if 'ground_truth' exists and is iterable
        if 'ground_truth' in entry and isinstance(entry['ground_truth'], list):
             for section in entry['ground_truth']:
                # Check if 'date' and 'diagnoses' exist
                if 'date' in section and 'diagnoses' in section and isinstance(section['diagnoses'], list):
                    for diag in section['diagnoses']:
                        # Check if 'diagnosis' exists
                        if 'diagnosis' in diag:
                             gold_standard.append({
                                'note_id': i,
                                'diagnosis': str(diag['diagnosis']).lower(), # Ensure string and lower
                                'date': section['date'] # Assume date is already YYYY-MM-DD
                            })
                        else:
                             print(f"Warning: Missing 'diagnosis' key in note {i}, section date {section.get('date', 'N/A')}")
                else:
                     print(f"Warning: Missing 'date' or 'diagnoses' key, or 'diagnoses' not a list in note {i}, section date {section.get('date', 'N/A')}")
        else:
             print(f"Warning: Missing or invalid 'ground_truth' in note {i}")

    print(f"Prepared gold standard with {len(gold_standard)} relationships.")


    # Pre-extract entities for efficiency
    print("Pre-extracting entities...")
    prepared_test_data = []
    for entry in test_data:
        text = entry.get('clinical_note', '') # Handle missing 'clinical_note'
        entities = extract_entities(text)
        prepared_test_data.append({'text': text, 'entities': entities})

    return prepared_test_data, gold_standard

# Helper function to run extraction process for a given extractor and data
def run_extraction(extractor, prepared_test_data):
    """
    Runs the extraction process for a given extractor on prepared data.

    Args:
        extractor: An initialized and loaded extractor object (subclass of BaseExtractor).
        prepared_test_data (list): List of dicts {'text': ..., 'entities': ...}.

    Returns:
        list: List of predicted relationships [{'note_id': ..., 'diagnosis': ..., 'date': ..., 'confidence': ...}].
    """
    print(f"Generating predictions using {extractor.name}...")
    all_predictions = []
    skipped_rels = 0
    for i, note_entry in enumerate(prepared_test_data):
        try:
            # Extract relationships using the provided extractor
            relationships = extractor.extract(note_entry['text'], entities=note_entry['entities'])
            for rel in relationships:
                # Ensure required keys exist and handle potential missing 'date' or 'diagnosis'
                raw_date = rel.get('date')
                raw_diagnosis = rel.get('diagnosis')
                if raw_date is None or raw_diagnosis is None:
                    print(f"Warning: Skipping relationship in note {i} due to missing 'date' or 'diagnosis'. Rel: {rel}")
                    skipped_rels += 1
                    continue

                # Parse date and normalize diagnosis
                parsed_date = parse_date_string(str(raw_date)) # Ensure string input
                normalized_diagnosis = str(raw_diagnosis).strip().lower() # Ensure string, strip, lower

                if parsed_date and normalized_diagnosis:
                    all_predictions.append({
                        'note_id': i,
                        'diagnosis': normalized_diagnosis,
                        'date': parsed_date,
                        'confidence': rel.get('confidence', 1.0) # Default confidence to 1.0 if missing
                    })
                else:
                    # Log if parsing failed but keys were present
                    # print(f"Debug: Skipping relationship in note {i} due to parsing failure. Raw Date: '{raw_date}', Raw Diag: '{raw_diagnosis}'")
                    skipped_rels += 1
        except Exception as e:
            # Log errors during extraction for a specific note
            print(f"Extraction error on note {i} for {extractor.name}: {e}")
            # Optionally, re-raise if you want errors to halt execution: raise e
            continue # Continue with the next note

    print(f"Generated {len(all_predictions)} predictions. Skipped {skipped_rels} potentially invalid relationships.")
    return all_predictions

def calculate_and_report_metrics(all_predictions, gold_standard, extractor_name, output_dir, total_notes):
    """
    Compares predictions with gold standard, calculates metrics, prints results,
    and saves a confusion matrix plot.

    Args:
        all_predictions (list): List of predicted relationships (normalized by the caller).
                                Each dict must contain 'note_id', 'diagnosis', 'date' (YYYY-MM-DD).
        gold_standard (list): List of gold standard relationships (normalized).
                              Each dict must contain 'note_id', 'diagnosis', 'date' (YYYY-MM-DD).
        extractor_name (str): Name of the extractor being evaluated.
        output_dir (str): Directory to save evaluation outputs.
        total_notes (int): The total number of notes evaluated.

    Returns:
        dict: A dictionary containing calculated metrics.
    """
    # Convert to sets for comparison
    # Caller ensures 'diagnosis' is normalized and 'date' is parsed to YYYY-MM-DD
    pred_set = set((p['note_id'], p['diagnosis'], p['date']) for p in all_predictions)
    gold_set = set((g['note_id'], g['diagnosis'], g['date']) for g in gold_standard)

    # Calculate TP, FP, FN
    true_positives = len(pred_set & gold_set)
    false_positives = len(pred_set - gold_set)
    false_negatives = len(gold_set - pred_set)
    true_negatives = 0 # TN is ill-defined/hard to calculate accurately here

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # --- Reporting ---
    print(f"  Evaluation Results for {extractor_name}:")
    # Report length of prediction set used for metrics
    print(f"    Unique predicted relationships: {len(pred_set)}") # TP + FP
    # Report length of gold set used for metrics
    print(f"    Unique gold relationships:    {len(gold_set)}")   # TP + FN
    print(f"    True Positives:  {true_positives}")
    print(f"    False Positives: {false_positives}")
    print(f"    False Negatives: {false_negatives}")
    print(f"    Precision: {precision:.3f}")
    print(f"    Recall:    {recall:.3f}")
    print(f"    F1 Score:  {f1:.3f}")

    # --- Plotting ---
    conf_matrix_values = [true_negatives, false_positives, false_negatives, true_positives]
    plt.figure(figsize=(6, 5))
    tn, fp, fn, tp = conf_matrix_values
    conf_matrix_display_array = np.array([[tn, fp], [fn, tp]])
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_display_array, display_labels=['No Relation', 'Has Relation'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f"Confusion Matrix - {extractor_name}")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    safe_extractor_name = re.sub(r'[^\w.-]+', '_', extractor_name).lower()
    plot_filename = f"{safe_extractor_name}_confusion_matrix.png"
    plot_save_path = os.path.join(output_dir, plot_filename)
    try:
        plt.savefig(plot_save_path)
        print(f"    Confusion matrix saved to {plot_save_path}")
    except Exception as e:
        print(f"    Error saving confusion matrix: {e}")
    plt.close()

    # Return calculated metrics
    metrics_dict = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        # Keep raw values if needed, but TN is 0
        # 'confusion_matrix': conf_matrix_values
    }
    return metrics_dict