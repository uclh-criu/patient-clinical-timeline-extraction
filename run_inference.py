import os
import pandas as pd
import json
import sys
from tqdm import tqdm

# Import from our modules
from extractors.extractor_factory import create_extractor
import config

# Import from utility modules
from utils.inference_utils import (
    load_and_prepare_data,
    run_extraction,
    get_data_path
)

def run_inference():
    """
    Run the specified extraction method on the dataset and save predictions to CSV.
    
    This script:
    1. Loads the dataset specified in config
    2. Initializes and loads the extractor model
    3. Runs extraction to get predictions
    4. Saves predictions to a new column in the original CSV file
    """
    print(f"\n=== Running Inference with {config.EXTRACTION_METHOD} ===")
    
    # Use get_data_path to determine the dataset path
    dataset_path = get_data_path(config)
    print(f"Dataset: {dataset_path}")
    
    # Use the NUM_TEST_SAMPLES from config
    num_test_samples = config.NUM_TEST_SAMPLES

    # Load and prepare data using the helper function
    print("Loading and preparing data...")
    if config.ENTITY_MODE == 'disorder_only':
        # In disorder_only mode, load_and_prepare_data returns only 2 values
        prepared_test_data, _ = load_and_prepare_data(dataset_path, num_test_samples, config)
    else:
        # In multi_entity mode, load_and_prepare_data returns 4 values
        prepared_test_data, _, _, _ = load_and_prepare_data(dataset_path, num_test_samples, config)
    
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
    all_predictions = run_extraction(extractor, prepared_test_data)
    print(f"Generated {len(all_predictions)} predictions.")

    # Save predictions to CSV
    print(f"Saving predictions to CSV: {dataset_path}...")
    
    try:
        # Load the original CSV file
        df = pd.read_csv(dataset_path)
        
        # Prepare column name based on extractor name
        safe_extractor_name = extractor.name.lower().replace(' ', '_')
        predictions_column = f"{safe_extractor_name}_predictions"
        
        # Determine if we're in disorder_only mode
        disorder_only_mode = (config.ENTITY_MODE == 'disorder_only')
        
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
            
            # Handle different field names between disorder_only and multi_entity modes
            if disorder_only_mode:
                # In disorder_only mode, use 'diagnosis' field
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
                    pred_obj['entity_category'] = 'disorder'
            
            # Add the prediction object to the list for this note
            note_predictions[note_id].append(pred_obj)
        
        # Add or update predictions column in DataFrame
        df[predictions_column] = None
            
        # Fill in predictions column
        num_filled = 0
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

if __name__ == "__main__":
    run_inference()
