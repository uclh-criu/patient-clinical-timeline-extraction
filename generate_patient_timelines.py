import os
import pandas as pd
import json
import sys
from tqdm import tqdm

# Import from our modules
import config

# Import from utility modules
from utils.inference_utils import get_data_path, load_and_prepare_data
from utils.post_processing_utils import (
    aggregate_predictions_by_patient,
    generate_patient_timelines,
    generate_patient_timeline_summary,
    generate_patient_timeline_visualizations
)

def create_patient_timelines():
    """
    Generate patient timeline artifacts from predictions stored in the dataset CSV file.
    
    This script:
    1. Loads the dataset with saved predictions
    2. Aggregates predictions by patient
    3. Generates three types of timeline artifacts:
       - Individual patient timeline text files
       - Patient timeline summary report
       - Visual timeline plots for each patient
    """
    print(f"\n=== Generating Patient Timelines for {config.EXTRACTION_METHOD} ===")
    
    # Use get_data_path to determine the dataset path
    dataset_path = get_data_path(config)
    print(f"Dataset: {dataset_path}")

    # Define the output directory for timelines
    if 'project_root' not in locals() and 'project_root' not in globals():
        project_root = os.path.dirname(os.path.abspath(__file__))
    timeline_output_dir = os.path.join(project_root, config.TIMELINE_OUTPUT_DIR)
    os.makedirs(timeline_output_dir, exist_ok=True)

    # --- Load the test split of the dataset ---
    # This ensures we're working with the same subset of data as inference and evaluation
    try:
        # Load the test split (without actually using the prepared data)
        if config.ENTITY_MODE == 'disorder_only':
            _, _ = load_and_prepare_data(dataset_path, config.INFERENCE_SAMPLES, config, data_split_mode='test')
        else:
            _, _, _, _ = load_and_prepare_data(dataset_path, config.INFERENCE_SAMPLES, config, data_split_mode='test')
            
        # The load_and_prepare_data function will print info about the test split
    except Exception as e:
        print(f"Warning: Could not load test split data: {e}")
        print("Proceeding with the full dataset...")

    # --- Load predictions from CSV ---
    try:
        # Load the CSV file with the predictions
        df = pd.read_csv(dataset_path)
        
        # Apply the same test split logic as in load_and_prepare_data
        if hasattr(config, 'TRAINING_SET_RATIO'):
            train_ratio = config.TRAINING_SET_RATIO
            split_idx = int(len(df) * train_ratio)
            df = df.sort_index(kind='stable')
            df = df.iloc[split_idx:]  # Use only the test portion
            print(f"Using test split: {len(df)} records (last {(1-train_ratio)*100:.0f}%)")
        
        # If a specific number of samples is requested, limit to that
        if hasattr(config, 'INFERENCE_SAMPLES') and config.INFERENCE_SAMPLES and config.INFERENCE_SAMPLES < len(df):
            df = df.iloc[:config.INFERENCE_SAMPLES]
            print(f"Limiting to {config.INFERENCE_SAMPLES} samples.")
        
        # Form the column name based on extraction method
        safe_extractor_name = config.EXTRACTION_METHOD.lower().replace(' ', '_')
        predictions_column = f"{safe_extractor_name}_predictions"
        
        if predictions_column not in df.columns:
            print(f"Error: Predictions column '{predictions_column}' not found in CSV.")
            print(f"Available columns: {', '.join(df.columns)}")
            print("Did you run run_inference.py first?")
            sys.exit(1)
        
        print(f"Found predictions column: {predictions_column}")
        
        # Check if patient ID column is available
        if not hasattr(config, 'PATIENT_ID_COLUMN'):
            print("Error: PATIENT_ID_COLUMN not defined in config.")
            sys.exit(1)
            
        patient_id_col = config.PATIENT_ID_COLUMN
        if patient_id_col not in df.columns:
            print(f"Error: Patient ID column '{patient_id_col}' not found in CSV.")
            sys.exit(1)
        
        # Parse predictions from CSV
        all_predictions = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Loading predictions"):
            # Get patient ID
            patient_id = row.get(patient_id_col)
            if pd.isna(patient_id):
                continue
                
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
                    pred['patient_id'] = patient_id
                    all_predictions.append(pred)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse predictions for note {i}. Skipping.")
                continue
        
        print(f"Loaded {len(all_predictions)} predictions for timeline generation")
    except Exception as e:
        print(f"Error loading predictions from CSV: {e}")
        sys.exit(1)

    # --- Generate Patient Timelines ---
    print("\nGenerating patient timelines...")
    
    # Aggregate predictions by patient
    patient_timelines = aggregate_predictions_by_patient(all_predictions)
    
    if not patient_timelines:
        print("No patient timelines to generate. Exiting.")
        sys.exit(0)
    
    print(f"Aggregated predictions for {len(patient_timelines)} patients")
    
    # Generate individual timeline files
    print("\n1. Generating individual patient timeline files...")
    generate_patient_timelines(patient_timelines, timeline_output_dir, config.EXTRACTION_METHOD)
    
    # Generate summary report
    print("\n2. Generating patient timeline summary report...")
    generate_patient_timeline_summary(patient_timelines, timeline_output_dir, config.EXTRACTION_METHOD)
    
    # Generate visual timeline plots
    print("\n3. Generating visual timeline plots...")
    generate_patient_timeline_visualizations(patient_timelines, timeline_output_dir, config.EXTRACTION_METHOD)
    
    print(f"\nAll patient timeline artifacts generated successfully in {timeline_output_dir}")

if __name__ == "__main__":
    create_patient_timelines()
