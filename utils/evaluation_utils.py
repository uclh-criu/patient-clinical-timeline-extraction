import re
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def calculate_entity_metrics(prepared_test_data, entity_gold, output_dir):
    """
    Evaluates the entity extraction (NER) performance by comparing extracted entities with gold standard.
    
    Args:
        prepared_test_data (list): List of dicts containing extracted entities.
        entity_gold (list): List of gold standard entities.
        output_dir (str): Directory to save evaluation outputs.
        
    Returns:
        dict: A dictionary containing calculated metrics.
    """
    print("\n--- Evaluating Entity Extraction (NER) ---")
    if not entity_gold:
        print("No entity_gold data provided. Skipping NER evaluation.")
        return {'precision': 0, 'recall': 0, 'f1': 0}

    # Create sets for comparison
    # Using (note_id, entity_label, entity_category, start, end) as the key for strict matching
    gold_entities_set = set(
        (g['note_id'], g['entity_label'], g['entity_category'], g['start'], g['end'])
        for g in entity_gold
    )

    # Extract predicted entities from prepared_test_data
    predicted_entities_set = set()
    for note_data in prepared_test_data:
        note_id = note_data['note_id']
        for entity in note_data['extracted_entities']:
            predicted_entities_set.add(
                (note_id, entity['label'], entity['category'], entity['start'], entity['end'])
            )

    # Calculate metrics
    true_positives = len(predicted_entities_set & gold_entities_set)
    false_positives = len(predicted_entities_set - gold_entities_set)
    false_negatives = len(gold_entities_set - predicted_entities_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"  Entity Evaluation Results:")
    print(f"  Total unique gold entities:     {len(gold_entities_set)}")
    print(f"  Total unique predicted entities: {len(predicted_entities_set)}")
    print(f"  True Positives:  {true_positives}")
    print(f"  False Positives: {false_positives}")
    print(f"  False Negatives: {false_negatives}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    
    # Plot confusion matrix
    try:
        plt.figure(figsize=(8, 6))
        conf_matrix_display_array = np.array([[0, false_positives], [false_negatives, true_positives]])
        
        # Create clearer labels
        display_labels = ['No Entity', 'Entity']
        
        # Create the confusion matrix display without automatic text
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_display_array, display_labels=display_labels)
        ax = disp.plot(cmap=plt.cm.Blues, values_format='', text_kw={'alpha': 0})
        
        # Add our own text annotations for each quadrant
        ax.text(0, 0, f'TN\n-', ha='center', va='center', fontsize=11)
        ax.text(1, 0, f'FP\n{false_positives}', ha='center', va='center', fontsize=11, color='white' if false_positives > 20 else 'black')
        ax.text(0, 1, f'FN\n{false_negatives}', ha='center', va='center', fontsize=11, color='white' if false_negatives > 20 else 'black')
        ax.text(1, 1, f'TP\n{true_positives}', ha='center', va='center', fontsize=11, color='white' if true_positives > 20 else 'black')
        
        # Set axis labels
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_xlabel('Predicted', fontsize=12)
        
        # Title with metrics
        plt.title(f"Entity Extraction Confusion Matrix\nPrec: {precision:.3f} | Rec: {recall:.3f} | F1: {f1:.3f}", 
                fontsize=12, pad=20)
        
        # Adjust layout
        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        plot_filename = os.path.join(output_dir, "entity_extraction_confusion_matrix.png")
        plt.savefig(plot_filename)
        print(f"  Entity confusion matrix saved to {plot_filename}")
        plt.close()
    except Exception as e:
        print(f"  Error saving entity confusion matrix: {e}")

    return {'precision': precision, 'recall': recall, 'f1': f1}

def calculate_and_report_metrics(all_predictions, gold_standard, extractor_name, output_dir, total_notes_processed, dataset_path=None):
    """
    Compares predictions with gold standard, calculates metrics, prints results,
    and saves a confusion matrix plot.

    Args:
        all_predictions (list): List of predicted relationships (normalized by the caller).
                                Each dict must contain 'note_id', 'diagnosis'/'entity_label', 'date' (YYYY-MM-DD).
        gold_standard (list): List of gold standard relationships (normalized).
                              Each dict must contain 'note_id', 'diagnosis'/'entity_label', 'date' (YYYY-MM-DD).
        extractor_name (str): Name of the extractor being evaluated.
        output_dir (str): Directory to save evaluation outputs.
        total_notes_processed (int): The total number of notes processed by the extractor.
        dataset_path (str, optional): Path to the dataset for display in the plot title.

    Returns:
        dict: A dictionary containing calculated metrics.
    """
    # Determine if we're in disorder_only mode by checking the keys in gold_standard
    # If any item has 'diagnosis' key, we're in disorder_only mode
    disorder_only_mode = False
    entity_key = 'entity_label'  # Default to multi_entity mode
    
    if gold_standard and len(gold_standard) > 0:
        if 'diagnosis' in gold_standard[0]:
            disorder_only_mode = True
            entity_key = 'diagnosis'
    
    # Also check the predictions to determine mode
    if all_predictions and len(all_predictions) > 0:
        if 'diagnosis' in all_predictions[0]:
            disorder_only_mode = True
            entity_key = 'diagnosis'
    
    print(f"Using {'disorder_only' if disorder_only_mode else 'multi_entity'} mode with entity key: {entity_key}")
    
    if not gold_standard:
        print(f"  No gold standard data provided for {extractor_name} (processed {total_notes_processed} notes). Skipping metric calculation.")
        # Return zeroed metrics if no gold standard
        return {
            'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0,
            'true_positives': 0, 'false_positives': 0, 'false_negatives': 0
        }

    # Identify the notes that have gold standard labels
    gold_note_ids = set(g['note_id'] for g in gold_standard)
    num_labeled_notes = len(gold_note_ids)

    if num_labeled_notes == 0:
        print(f"  No notes with gold standard labels found for {extractor_name} (processed {total_notes_processed} notes). Skipping metric calculation.")
        return {
            'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0,
            'true_positives': 0, 'false_positives': 0, 'false_negatives': 0
        }
    
    print(f"  Evaluating metrics for {extractor_name} based on {num_labeled_notes} notes with gold standard labels (out of {total_notes_processed} notes processed).")

    # Filter predictions to include only those from labeled notes
    filtered_predictions = [p for p in all_predictions if p['note_id'] in gold_note_ids]
    
    # --- DEBUG: Print predictions and gold standards for comparison ---
    print("\n--- Comparing Predictions to Gold Standard ---")
    
    # Load the dataset to get the note text if dataset_path is provided
    note_texts = {}
    if dataset_path:
        try:
            import pandas as pd
            df = pd.read_csv(dataset_path)
            if 'note' in df.columns and 'note_id' in df.columns:
                for _, row in df.iterrows():
                    note_id = row['note_id']
                    if note_id in gold_note_ids:
                        note_texts[note_id] = row['note']
        except Exception as e:
            print(f"Warning: Could not load note texts from dataset: {e}")
    
    # Print comparison for all notes with gold standard labels
    for note_id in sorted(gold_note_ids):
        # Using a simple list comprehension for clarity
        note_preds = [p for p in filtered_predictions if p.get('note_id') == note_id]
        note_golds = [g for g in gold_standard if g.get('note_id') == note_id]
        
        print(f"\n[Note ID: {note_id}]")
        
        # Print the full note text if available
        if note_id in note_texts:
            print("\nFULL NOTE TEXT:")
            print("-" * 80)
            print(note_texts[note_id])
            print("-" * 80)
        
        # To make it easier to read, convert dicts to strings and join them
        # Always use entity_label since our refactoring standardized on that field
        gold_str = '\n    - '.join([f"{g.get('entity_label', 'unknown')} @ {g.get('date', 'unknown')}" for g in note_golds])
        
        # For predictions, still respect the mode since extractors might return different fields
        if disorder_only_mode:
            pred_str = '\n    - '.join([f"{p.get('diagnosis', p.get('entity_label', 'unknown'))} @ {p.get('date', 'unknown')}" for p in note_preds])
        else:
            pred_str = '\n    - '.join([f"{p.get('entity_label', p.get('diagnosis', 'unknown'))} @ {p.get('date', 'unknown')}" for p in note_preds])
        
        print(f"  Gold Standard : \n    - {gold_str if gold_str else '[]'}")
        print(f"  Predictions   : \n    - {pred_str if pred_str else '[]'}")
    print("--------------------------------------------\n")
    
    if not filtered_predictions:
        print(f"  No predictions found for the {num_labeled_notes} labeled notes by {extractor_name}.")
        # If no predictions for labeled notes, TP and FP are 0. FN is total gold.
        true_positives = 0
        false_positives = 0
        false_negatives = len(gold_standard) # All gold items were missed
        pred_set = set()  # Empty set for reporting
        
        # Create gold_set based on the mode
        gold_set = set()
        for g in gold_standard:
            # Always use entity_label for gold standard since our refactoring standardized on that field
            gold_set.add((g['note_id'], g.get('entity_label', ''), g.get('date', '')))
    else:
        # Convert filtered predictions and gold standard to sets for comparison
        # Handle both disorder_only and multi_entity modes
        pred_set = set()
        for p in filtered_predictions:
            if disorder_only_mode:
                # In disorder_only mode, use 'diagnosis' field
                if 'diagnosis' in p:
                    pred_set.add((p['note_id'], p['diagnosis'], p['date']))
                else:
                    # If 'diagnosis' is missing but 'entity_label' exists, use that instead
                    if 'entity_label' in p:
                        pred_set.add((p['note_id'], p['entity_label'], p['date']))
                    else:
                        print(f"Warning: Prediction missing both 'diagnosis' and 'entity_label' fields: {p}")
            else:
                # In multi_entity mode, use 'entity_label' field
                if 'entity_label' in p:
                    if 'entity_category' in p:
                        pred_set.add((p['note_id'], p['entity_label'], p['entity_category'], p['date']))
                    else:
                        pred_set.add((p['note_id'], p['entity_label'], p['date']))
                else:
                    # If 'entity_label' is missing but 'diagnosis' exists, use that instead
                    if 'diagnosis' in p:
                        pred_set.add((p['note_id'], p['diagnosis'], 'disorder', p['date']))
                    else:
                        print(f"Warning: Prediction missing both 'entity_label' and 'diagnosis' fields: {p}")
        
        # Similarly for gold standard
        gold_set = set()
        for g in gold_standard:
            # Always use entity_label for gold standard since our refactoring standardized on that field
            if 'entity_label' in g:
                if 'entity_category' in g and not disorder_only_mode:
                    gold_set.add((g['note_id'], g['entity_label'], g['entity_category'], g['date']))
                else:
                    gold_set.add((g['note_id'], g['entity_label'], g['date']))
            else:
                # Fall back to diagnosis if entity_label is not available (should not happen after refactoring)
                if 'diagnosis' in g:
                    if not disorder_only_mode:
                        gold_set.add((g['note_id'], g['diagnosis'], 'disorder', g['date']))
                    else:
                        gold_set.add((g['note_id'], g['diagnosis'], g['date']))
                else:
                    print(f"Warning: Gold standard missing both 'entity_label' and 'diagnosis' fields: {g}")

        # Calculate TP, FP, FN based on filtered predictions
        true_positives = len(pred_set & gold_set)
        false_positives = len(pred_set - gold_set)
        false_negatives = len(gold_set - pred_set)
    
    true_negatives = 0 # TN is ill-defined/hard to calculate accurately here

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate accuracy: (TP + TN) / (TP + TN + FP + FN)
    # Note: TN is set to 0 since it's ill-defined for this task, so accuracy = TP / (TP + FP + FN)
    accuracy = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0

    # --- Reporting ---
    print(f"  Evaluation Results for {extractor_name} (on labeled subset):")
    print(f"    Total unique predictions for labeled notes: {len(pred_set)}")    # TP + FP for labeled notes
    print(f"    Total unique gold relationships:           {len(gold_set)}")   # TP + FN for labeled notes
    print(f"    True Positives:  {true_positives}")
    print(f"    False Positives: {false_positives}")
    print(f"    False Negatives: {false_negatives}")
    print(f"    Precision: {precision:.3f}")
    print(f"    Recall:    {recall:.3f}")
    print(f"    F1 Score:  {f1:.3f}")
    print(f"    Accuracy:  {accuracy:.3f}")

    # --- Plotting ---
    # Plotting confusion matrix based on these filtered values
    conf_matrix_values = [true_negatives, false_positives, false_negatives, true_positives]
    plt.figure(figsize=(8, 6))
    tn, fp, fn, tp = conf_matrix_values
    conf_matrix_display_array = np.array([[tn, fp], [fn, tp]])
    
    # Create clearer labels
    display_labels = ['No Relation', 'Has Relation']
    
    # Create the confusion matrix display without automatic text
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_display_array, display_labels=display_labels)
    ax = disp.plot(cmap=plt.cm.Blues, values_format='', text_kw={'alpha': 0})  # Hide automatic text
    
    # Add custom annotations to make TP/TN/FP/FN clear
    ax = plt.gca()
    
    # Add our own text annotations for each quadrant
    ax.text(0, 0, f'TN\n{tn}', ha='center', va='center', fontsize=11, color='white' if tn > 20 else 'black')
    ax.text(1, 0, f'FP\n{fp}', ha='center', va='center', fontsize=11, color='white' if fp > 20 else 'black')
    ax.text(0, 1, f'FN\n{fn}', ha='center', va='center', fontsize=11, color='white' if fn > 20 else 'black')
    ax.text(1, 1, f'TP\n{tp}', ha='center', va='center', fontsize=11, color='white' if tp > 20 else 'black')
    
    # Set axis labels
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)
    
    # Get dataset name from path for display
    dataset_name = "Unknown"
    if dataset_path:
        dataset_name = os.path.basename(dataset_path)
    
    # Title with metrics and dataset info
    plt.title(f"Confusion Matrix - {extractor_name}\nPrec: {precision:.3f} | Rec: {recall:.3f} | F1: {f1:.3f} | Acc: {accuracy:.3f}\nDataset: {dataset_name}", 
             fontsize=12, pad=20)
    
    # Adjust layout
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    safe_extractor_name = re.sub(r'[^\w.-]+', '_', extractor_name).lower()
    plot_filename = f"{safe_extractor_name}_confusion_matrix_labeled_subset.png"
    plot_save_path = os.path.join(output_dir, plot_filename)
    try:
        plt.savefig(plot_save_path)
        print(f"    Confusion matrix (labeled subset) saved to {plot_save_path}")
    except Exception as e:
        print(f"    Error saving confusion matrix: {e}")
    plt.close()

    # Return calculated metrics
    metrics_dict = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
    }
    return metrics_dict