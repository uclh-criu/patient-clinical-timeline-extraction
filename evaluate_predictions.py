import os
import pandas as pd
import json
import sys

# Import from our modules
import config

# Import from utility modules
from utils.inference_utils import (
    load_and_prepare_data,
    get_data_path
)

from utils.evaluation_utils import (
    calculate_and_report_metrics,
    calculate_entity_metrics
)

def evaluate_predictions():
    """
    Evaluate predictions stored in the dataset CSV file.
    
    This script:
    1. Loads the dataset with saved predictions
    2. Loads gold standard data for comparison
    3. Calculates metrics (precision, recall, F1) for both entity and relationship extraction
    4. Outputs summary report and confusion matrices
    """
    print(f"\n=== Evaluating Predictions for {config.EXTRACTION_METHOD} ===")
    
    # Use get_data_path to determine the dataset path
    dataset_path = get_data_path(config)
    print(f"Dataset: {dataset_path}")
    
    # Use the NUM_TEST_SAMPLES from config
    num_test_samples = config.NUM_TEST_SAMPLES

    # Define the output directory for metric results
    if 'project_root' not in locals() and 'project_root' not in globals():
        project_root = os.path.dirname(os.path.abspath(__file__))
    EXPERIMENT_OUTPUT_DIR = os.path.join(project_root, "experiment_outputs")
    os.makedirs(EXPERIMENT_OUTPUT_DIR, exist_ok=True)

    # --- 1. Load and prepare gold standard data ---
    print("Loading gold standard data...")
    if config.ENTITY_MODE == 'disorder_only':
        # In disorder_only mode, load_and_prepare_data returns only 2 values
        prepared_test_data, relationship_gold = load_and_prepare_data(dataset_path, num_test_samples, config)
        # Set entity_gold to None as it's not used in disorder_only mode
        entity_gold = None
    else:
        # In multi_entity mode, load_and_prepare_data returns 4 values
        prepared_test_data, entity_gold, relationship_gold, _ = load_and_prepare_data(dataset_path, num_test_samples, config)
    
    if prepared_test_data is None:
        print("Failed to load or prepare data. Exiting evaluation.")
        sys.exit(1)
    
    # --- 2. Load predictions from CSV ---
    try:
        # Load the CSV file with the predictions
        df = pd.read_csv(dataset_path)
        
        # Form the column name based on extraction method
        safe_extractor_name = config.EXTRACTION_METHOD.lower().replace(' ', '_')
        predictions_column = f"{safe_extractor_name}_predictions"
        
        if predictions_column not in df.columns:
            print(f"Error: Predictions column '{predictions_column}' not found in CSV.")
            print(f"Available columns: {', '.join(df.columns)}")
            print("Did you run run_inference.py first?")
            sys.exit(1)
        
        print(f"Found predictions column: {predictions_column}")
        
        # Parse predictions from CSV
        all_predictions = []
        for i, row in df.iterrows():
            # Get predictions for this note
            pred_str = row.get(predictions_column)
            if pd.isna(pred_str) or not pred_str:
                continue
                
            try:
                # Parse the JSON string to get the predictions
                note_predictions = json.loads(pred_str)
                
                # Add note_id and patient_id to each prediction
                for pred in note_predictions:
                    pred['note_id'] = i
                    
                    # If patient_id column exists, add it too
                    if hasattr(config, 'REAL_DATA_PATIENT_ID_COLUMN'):
                        patient_id_col = config.REAL_DATA_PATIENT_ID_COLUMN
                        if patient_id_col in row:
                            pred['patient_id'] = row[patient_id_col]
                    
                    all_predictions.append(pred)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse predictions for note {i}. Skipping.")
                continue
        
        print(f"Loaded {len(all_predictions)} predictions from column '{predictions_column}'")
    except Exception as e:
        print(f"Error loading predictions from CSV: {e}")
        sys.exit(1)

    # --- 3. Entity Extraction (NER) Evaluation ---
    print("\n--- Entity Extraction (NER) Evaluation ---")
    if config.ENTITY_MODE == 'disorder_only':
        # Skip entity metrics in disorder_only mode
        entity_metrics = {
            'precision': 0,
            'recall': 0,
            'f1': 0
        }
        print("Entity extraction evaluation skipped in disorder_only mode.")
    else:
        entity_metrics = calculate_entity_metrics(prepared_test_data, entity_gold, EXPERIMENT_OUTPUT_DIR)

    # --- 4. Relationship Extraction (RE) Evaluation ---
    print("\n--- Relationship Extraction (RE) Evaluation ---")
    relationship_metrics = calculate_and_report_metrics(
        all_predictions,
        relationship_gold,
        config.EXTRACTION_METHOD,
        EXPERIMENT_OUTPUT_DIR,
        len(prepared_test_data),
        dataset_path
    )
    
    # Print evaluation summary
    print("\n=== EVALUATION SUMMARY ===")
    
    print("Entity Extraction (NER):")
    print(f"  Precision: {entity_metrics['precision']:.3f}")
    print(f"  Recall:    {entity_metrics['recall']:.3f}")
    print(f"  F1 Score:  {entity_metrics['f1']:.3f}")
    
    print("\nRelationship Extraction (RE):")
    print(f"  Precision: {relationship_metrics['precision']:.3f}")
    print(f"  Recall:    {relationship_metrics['recall']:.3f}")
    print(f"  F1 Score:  {relationship_metrics['f1']:.3f}")
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    evaluate_predictions()
