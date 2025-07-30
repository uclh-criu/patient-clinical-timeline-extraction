import os
import pandas as pd
import json
import sys
from tqdm import tqdm

# Import from our modules
from extractors.extractor_factory import create_extractor
import config

# Import from utility modules
from utils.inference_eval_utils import (
    load_and_prepare_data,
    run_extraction,
    get_data_path,
    calculate_entity_metrics,
    calculate_and_report_metrics
)

def run_inference_and_evaluate():
    """
    Run the specified extraction method on the dataset, save predictions to CSV,
    and evaluate the predictions against the gold standard.
    
    This script:
    1. Loads the dataset specified in config
    2. Initializes and loads the extractor model
    3. Runs extraction to get predictions
    4. Saves predictions to a new column in the original CSV file
    5. Evaluates the predictions against the gold standard
    6. Outputs evaluation metrics and confusion matrices
    """
    print(f"\n=== Running Inference and Evaluation with method: '{config.EXTRACTION_METHOD}' ===")
    
    # Use get_data_path to determine the dataset path
    dataset_path = get_data_path(config)
    print(f"Dataset: {dataset_path}")
    
    # Use the INFERENCE_SAMPLES from config
    num_samples = config.INFERENCE_SAMPLES

    # Define the output directory for metric results
    if 'project_root' not in locals() and 'project_root' not in globals():
        project_root = os.path.dirname(os.path.abspath(__file__))
    EXPERIMENT_OUTPUT_DIR = os.path.join(project_root, "experiment_outputs")
    os.makedirs(EXPERIMENT_OUTPUT_DIR, exist_ok=True)

    # Load and prepare data using the helper function
    # Use 'test' split mode to ensure we're using the holdout test set
    print("Loading and preparing data...")
    if config.ENTITY_MODE == 'diagnosis_only':
        # In diagnosis_only mode, load_and_prepare_data returns only 2 values
        prepared_test_data, relationship_gold = load_and_prepare_data(dataset_path, num_samples, config, data_split_mode='test')
        # Set entity_gold to None as it's not used in diagnosis_only mode
        entity_gold = None
    else:
        # In multi_entity mode, load_and_prepare_data returns 4 values
        prepared_test_data, entity_gold, relationship_gold, _ = load_and_prepare_data(dataset_path, num_samples, config, data_split_mode='test')
    
    if prepared_test_data is None:
        print("Failed to load or prepare data. Exiting.")
        sys.exit(1)

    # Create and load extractor
    try:
        print(f"Creating {config.EXTRACTION_METHOD} extractor...")
        extractor = create_extractor(config.EXTRACTION_METHOD, config)
        if not extractor.load():
            print(f"Failed to load {extractor.name}. Exiting.")
            sys.exit(1)
    except Exception as e:
        print(f"Error creating or loading extractor: {e}")
        sys.exit(1)

    # Generate predictions
    print(f"Running extraction with {extractor.name}...")
    all_predictions = run_extraction(extractor, prepared_test_data, relationship_gold, dataset_path)
    print(f"Generated {len(all_predictions)} predictions.")

    # Save predictions to CSV
    print(f"Saving predictions to CSV: {dataset_path}...")
    
    try:
        # Load the original CSV file
        df = pd.read_csv(dataset_path)
        
        # Prepare column name based on extractor name
        safe_extractor_name = extractor.name.lower().replace(' ', '_')
        predictions_column = f"{safe_extractor_name}_predictions"
        
        # Determine if we're in diagnosis_only mode
        diagnosis_only_mode = (config.ENTITY_MODE == 'diagnosis_only')
        
        # Create dictionaries to hold predictions by note_id
        note_predictions = {}
        
        # Group predictions by note_id
        for pred in all_predictions:
            note_id = pred['note_id']
            if note_id not in note_predictions:
                note_predictions[note_id] = []
            
            # Create a prediction object with only the necessary fields
            pred_obj = {
                'date': pred['date'],
                'confidence': pred.get('confidence', 1.0)
            }
            
            # Handle different field names between diagnosis_only and multi_entity modes
            if diagnosis_only_mode:
                # In diagnosis_only mode, use 'diagnosis' field
                if 'diagnosis' in pred:
                    pred_obj['diagnosis'] = pred['diagnosis']
                elif 'entity_label' in pred:
                    # Fall back to entity_label if diagnosis is not available
                    pred_obj['diagnosis'] = pred['entity_label']
            else:
                # In multi_entity mode, use 'entity_label' and 'entity_category' fields
                if 'entity_label' in pred:
                    pred_obj['entity_label'] = pred['entity_label']
                    if 'entity_category' in pred:
                        pred_obj['entity_category'] = pred['entity_category']
                    else:
                        pred_obj['entity_category'] = 'unknown'
                elif 'diagnosis' in pred:
                    # Fall back to diagnosis if entity_label is not available
                    pred_obj['entity_label'] = pred['diagnosis']
                    pred_obj['entity_category'] = 'diagnosis'
            
            # Add the prediction object to the list for this note
            note_predictions[note_id].append(pred_obj)
        
        # Add or update predictions column in DataFrame
        df[predictions_column] = None
            
        # Fill in predictions column
        num_filled = 0
        note_id_column = getattr(config, 'NOTE_ID_COLUMN', None)
        
        # If we have a note_id column in the data, use that to match predictions
        if note_id_column and note_id_column in df.columns:
            for i, row in tqdm(df.iterrows(), total=len(df), desc="Saving predictions"):
                note_id = row[note_id_column]
                if note_id in note_predictions:
                    df.at[i, predictions_column] = json.dumps(note_predictions[note_id])
                    num_filled += 1
        else:
            # Fall back to using row index as note_id
            for i, row in tqdm(df.iterrows(), total=len(df), desc="Saving predictions"):
                if i in note_predictions:
                    df.at[i, predictions_column] = json.dumps(note_predictions[i])
                    num_filled += 1
        
        # Save the updated dataframe back to CSV
        df.to_csv(dataset_path, index=False)
        print(f"Successfully saved predictions to column '{predictions_column}' for {num_filled} notes.")
        
    except Exception as e:
        print(f"Error saving predictions to CSV: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\nInference completed successfully!")
    
    # Now proceed with evaluation
    print(f"\n=== Evaluating Predictions for {config.EXTRACTION_METHOD} ===")
    
    # --- 3. Entity Extraction (NER) Evaluation ---
    print("\n--- Entity Extraction (NER) Evaluation ---")
    if config.ENTITY_MODE == 'diagnosis_only':
        # Skip entity metrics in diagnosis_only mode
        entity_metrics = {
            'precision': 0,
            'recall': 0,
            'f1': 0
        }
        print("Entity extraction evaluation skipped in diagnosis_only mode.")
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
    run_inference_and_evaluate() 