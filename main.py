import os
import matplotlib.pyplot as plt
import pandas as pd
import json
# Add tqdm for progress bars
from tqdm import tqdm
from datetime import datetime

# Import from our modules
from extractors.extractor_factory import create_extractor
from utils.extraction_utils import (
    extract_entities,
    calculate_and_report_metrics,
    load_and_prepare_data,
    run_extraction,
    get_data_path,
    transform_python_to_json,
    extract_relative_dates_llm
)
from data.sample_note import CLINICAL_NOTE
import config

# Define the output directory using absolute path if project_root is available
if 'project_root' not in locals() and 'project_root' not in globals():
    project_root = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_OUTPUT_DIR = os.path.join(project_root, "experiment_outputs")

def test_single_note():
    """
    Test a single clinical note using the extraction method from config.
    Outputs a list of (date, diagnosis) tuples.
    """
    print(f"Using device: {config.DEVICE}")
    print(f"Using extraction method: {config.EXTRACTION_METHOD}")
    print(f"Using data source: {config.DATA_SOURCE}")
    
    # Create appropriate extractor instance
    try:
        extractor = create_extractor(config.EXTRACTION_METHOD, config)
        print(f"Created {extractor.name} extractor")
    except ValueError as e:
        print(f"Error: {e}")
        return
    except ImportError as e:
        print(f"Error: Required package not installed: {e}")
        print("Please install the necessary packages for the selected extractor.")
        return
    
    # Load the extractor (models, API clients, etc.)
    if not extractor.load():
        print(f"Failed to load {extractor.name} extractor. Check configuration and dependencies.")
        return
    else:
        print(f"Successfully loaded {extractor.name} extractor")
    
    # Apply to clinical note
    print("\nApplying extractor to clinical note...")
    
    # Initialize variables
    clinical_note = None
    entities = None
    
    # Get the clinical note and entities based on the data source
    if hasattr(config, 'DATA_SOURCE') and config.DATA_SOURCE.lower() != 'synthetic':
        # Load the first note from the real data CSV
        try:
            import pandas as pd
            import json
            
            dataset_path = get_data_path(config)
            df = pd.read_csv(dataset_path)
            if len(df) > 0 and config.REAL_DATA_TEXT_COLUMN in df.columns:
                # Get the note text
                clinical_note = df.iloc[0][config.REAL_DATA_TEXT_COLUMN]
                
                # Check if annotation columns exist and try to load entities from them
                diagnoses_column = getattr(config, 'REAL_DATA_DIAGNOSES_COLUMN', None)
                dates_column = getattr(config, 'REAL_DATA_DATES_COLUMN', None)
                
                if (diagnoses_column and dates_column and 
                    diagnoses_column in df.columns and dates_column in df.columns):
                    
                    diagnoses_data = df.iloc[0][diagnoses_column]
                    dates_data = df.iloc[0][dates_column]
                    
                    if pd.notna(diagnoses_data) and pd.notna(dates_data):
                        try:
                            # Parse diagnoses annotations from JSON
                            diagnoses_list = []
                            # Transform Python-style string to valid JSON before parsing
                            valid_diagnoses_json = transform_python_to_json(diagnoses_data)
                            disorders = json.loads(valid_diagnoses_json)
                            for disorder in disorders:
                                label = disorder.get('label', '')
                                start_pos = disorder.get('start', 0)
                                diagnoses_list.append((label.lower(), start_pos))
                            
                            # Parse dates annotations from JSON
                            dates_list = []
                            # Transform Python-style string to valid JSON before parsing
                            valid_dates_json = transform_python_to_json(dates_data)
                            formatted_dates = json.loads(valid_dates_json)
                            for date_obj in formatted_dates:
                                parsed_date = date_obj.get('parsed', '')
                                original_date = date_obj.get('original', '')
                                start_pos = date_obj.get('start', 0)
                                dates_list.append((parsed_date, original_date, start_pos))
                            
                            # If we found entities, use them
                            if diagnoses_list or dates_list:
                                entities = (diagnoses_list, dates_list)
                                print(f"Using entities from annotation columns")
                            else:
                                # Use empty entities if found none in annotated columns
                                entities = ([], [])
                                print(f"No entities found in annotation columns, using empty entity lists")
                        except Exception as e:
                            # Use empty entities if there was an error
                            entities = ([], [])
                            print(f"Warning: Could not parse annotations, using empty entity lists: {e}")
                    else:
                        # Use empty entities if annotations are missing or NaN
                        entities = ([], [])
                        print(f"No valid annotations available, using empty entity lists")
                else:
                    # No annotation columns, only try extraction for synthetic data
                    if config.DATA_SOURCE.lower() == 'synthetic':
                        print(f"No annotation columns found, this will be extracted for synthetic data only")
                    else:
                        # For real data with no annotation columns, use empty entities
                        entities = ([], [])
                        print(f"No annotation columns found, using empty entity lists for real data")
                
                print(f"Using first note from real data CSV: {dataset_path}")
            else:
                print(f"Error: No valid notes found in {dataset_path}")
                return
        except Exception as e:
            print(f"Error loading real data CSV: {e}")
            return
    else:
        # Use the default sample note for synthetic data
        from data.sample_note import CLINICAL_NOTE
        clinical_note = CLINICAL_NOTE
        print("Using sample clinical note")
    
    # If entities weren't loaded from annotation columns, extract them from text
    # but only for synthetic data
    if entities is None:
        if not hasattr(config, 'DATA_SOURCE') or config.DATA_SOURCE.lower() == 'synthetic':
            entities = extract_entities(clinical_note)
        else:
            # For real data with no entities yet, use empty lists
            entities = ([], [])
    
    # Check if we should try to extract relative dates for real data
    if (hasattr(config, 'ENABLE_RELATIVE_DATE_EXTRACTION') and 
        config.ENABLE_RELATIVE_DATE_EXTRACTION and
        hasattr(config, 'DATA_SOURCE') and 
        config.DATA_SOURCE.lower() != 'synthetic' and
        hasattr(config, 'REAL_DATA_TIMESTAMP_COLUMN')):
        
        timestamp_column = config.REAL_DATA_TIMESTAMP_COLUMN
        
        try:
            # Get the dataset path and load the first row
            dataset_path = get_data_path(config)
            df = pd.read_csv(dataset_path, nrows=1)
            
            if timestamp_column in df.columns:
                timestamp_str = df.iloc[0][timestamp_column]
                
                if pd.notna(timestamp_str) and timestamp_str:
                    # Parse the timestamp string into a datetime object
                    # Try different formats
                    document_timestamp = None
                    timestamp_formats = [
                        '%Y-%m-%d',              # 2023-10-26
                        '%Y-%m-%d %H:%M:%S',     # 2023-10-26 15:30:45
                        '%m/%d/%Y',              # 10/26/2023
                        '%m/%d/%Y %H:%M:%S',     # 10/26/2023 15:30:45
                        '%d-%b-%Y',              # 26-Oct-2023
                        '%d %b %Y',              # 26 Oct 2023
                    ]
                    
                    for format_str in timestamp_formats:
                        try:
                            document_timestamp = datetime.strptime(timestamp_str, format_str)
                            break
                        except ValueError:
                            continue
                    
                    if document_timestamp:
                        print(f"Extracting relative dates using document timestamp: {document_timestamp.strftime('%Y-%m-%d')}")
                        # Extract relative dates using LLM
                        relative_dates = extract_relative_dates_llm(clinical_note, document_timestamp, config)
                        
                        if relative_dates:
                            # Append relative dates to existing dates list
                            diagnoses_list, dates_list = entities
                            combined_dates_list = dates_list + relative_dates
                            
                            # Update entities with combined dates
                            entities = (diagnoses_list, combined_dates_list)
                            print(f"Added {len(relative_dates)} relative dates from LLM extraction")
                        else:
                            print("No relative dates found by LLM extraction")
                    else:
                        print(f"Warning: Could not parse timestamp '{timestamp_str}'")
            else:
                print(f"Warning: Timestamp column '{timestamp_column}' not found in CSV file")
                
        except Exception as e:
            print(f"Error extracting relative dates: {e}")
    
    diagnoses, dates = entities
    print(f"Found {len(diagnoses)} diagnoses and {len(dates)} dates")
    
    relationships = extractor.extract(clinical_note, entities=entities)
    print(f"Found {len(relationships)} relationships")
    
    # Convert to list of (date, diagnosis) tuples
    output_tuples = [(rel['date'], rel['diagnosis']) for rel in relationships]
    
    # Optional: Sort by date string (simple sort, may not be perfectly chronological for all formats)
    # Consider date parsing from utils.data_preparation for robust sorting if needed.
    output_tuples.sort(key=lambda x: x[0]) 
    
    # Print the list of tuples
    print("\nExtracted Date-Diagnosis Tuples:")
    # For slightly cleaner printing:
    if output_tuples:
        for dt, dx in output_tuples:
            print(f"  ('{dt}', '{dx}')")
    else:
        print("  []")
    
    print("\nDone!")

def evaluate_on_dataset():
    """
    Evaluate the configured extraction method on the dataset specified in config.py.
    Saves predictions and correctness indicators to the original CSV file.
    """
    # Use get_data_path to determine the dataset path
    dataset_path = get_data_path(config)
    
    # Use None to use all evaluation samples (the last 20% of dataset)
    num_test_samples = None

    # Load and prepare data using the helper function, passing the config
    prepared_test_data, gold_standard = load_and_prepare_data(dataset_path, num_test_samples, config)
    if prepared_test_data is None or gold_standard is None:
        print("Failed to load or prepare data. Exiting evaluation.")
        return

    # Create and load extractor
    try:
        extractor = create_extractor(config.EXTRACTION_METHOD, config)
        if not extractor.load():
             print(f"Failed to load {extractor.name}. Exiting evaluation.")
             return
    except Exception as e:
        print(f"Error creating or loading extractor: {e}")
        return

    # Generate predictions using the helper function
    all_predictions = run_extraction(extractor, prepared_test_data)

    # Calculate and report metrics
    print("\nCalculating metrics...")
    os.makedirs(EXPERIMENT_OUTPUT_DIR, exist_ok=True)
    metrics_result = calculate_and_report_metrics(
        all_predictions,
        gold_standard,
        extractor.name,
        EXPERIMENT_OUTPUT_DIR,
        len(prepared_test_data)
    )
    
    # Only save predictions to CSV for non-synthetic data
    if hasattr(config, 'DATA_SOURCE') and config.DATA_SOURCE.lower() != 'synthetic':
        print(f"\nSaving predictions to {dataset_path}...")
        
        try:
            # Load the original CSV file
            df = pd.read_csv(dataset_path)
            
            # Prepare column names
            safe_extractor_name = extractor.name.lower().replace(' ', '_')
            predictions_column = f"{safe_extractor_name}_predictions"
            correctness_column = f"{safe_extractor_name}_is_correct"
            
            # Create dictionaries to hold predictions and correctness by note_id
            note_predictions = {}
            note_correctness = {}
            
            # Group predictions by note_id
            for pred in all_predictions:
                note_id = pred['note_id']
                if note_id not in note_predictions:
                    note_predictions[note_id] = []
                
                note_predictions[note_id].append({
                    'diagnosis': pred['diagnosis'],
                    'date': pred['date'],
                    'confidence': pred.get('confidence', 1.0)
                })
            
            # Check correctness if gold standard exists
            if gold_standard:
                # Convert gold_standard to a set of (note_id, diagnosis, date) tuples for easier comparison
                gold_set = set((g['note_id'], g['diagnosis'], g['date']) for g in gold_standard)
                
                # Check each prediction against the gold standard
                for pred in all_predictions:
                    note_id = pred['note_id']
                    is_correct = (note_id, pred['diagnosis'], pred['date']) in gold_set
                    
                    if note_id not in note_correctness:
                        note_correctness[note_id] = []
                        
                    note_correctness[note_id].append(is_correct)
            
            # Add predictions to dataframe
            df[predictions_column] = None
            if gold_standard:
                df[correctness_column] = None
                
            # Fill in predictions and correctness columns
            for i, row in df.iterrows():
                if i in note_predictions:
                    df.at[i, predictions_column] = json.dumps(note_predictions[i])
                    
                    if i in note_correctness:
                        df.at[i, correctness_column] = json.dumps(note_correctness[i])
            
            # Save the updated dataframe back to CSV
            df.to_csv(dataset_path, index=False)
            print(f"Successfully saved predictions to column '{predictions_column}'")
            if gold_standard:
                print(f"Successfully saved correctness indicators to column '{correctness_column}'")
                
        except Exception as e:
            print(f"Error saving predictions to CSV: {e}")
    
    print("\nEvaluation Done!")

def compare_all_methods():
    """
    Compare available extraction methods on the dataset.
    Saves predictions and correctness indicators from all methods to the original CSV file.
    """
    # Use get_data_path to determine the dataset path
    dataset_path = get_data_path(config)
    
    # Use None to use all evaluation samples (the last 20% of dataset)
    num_test_samples = None

    # List methods to compare
    if hasattr(config, 'COMPARISON_METHODS'):
        methods_to_try = config.COMPARISON_METHODS
    else:
        # Default to most available methods (ensure 'llama' is included)
        methods_to_try = ['custom', 'naive', 'relcat', 'llm', 'llama']

    print(f"\nComparing methods: {', '.join(methods_to_try)}")
    print(f"Using data source: {config.DATA_SOURCE}")
    print(f"Using dataset: {dataset_path}")

    # Load and prepare data once using the helper function, passing the config
    prepared_test_data, gold_standard = load_and_prepare_data(dataset_path, num_test_samples, config)
    if prepared_test_data is None or gold_standard is None:
        print("Failed to load or prepare data. Exiting comparison.")
        return

    # Load extractors
    extractors_to_compare = []
    for method in methods_to_try:
        print(f"\nAttempting to load {method} extractor...")
        try:
            extractor = create_extractor(method, config)
            if extractor.load():
                extractors_to_compare.append(extractor)
                print(f"Loaded {extractor.name}")
            else:
                print(f"Skipping {method}: load() failed")
        except Exception as e:
            print(f"Skipping {method}: Load error - {e}")

    if not extractors_to_compare:
        print("No extractors loaded successfully for comparison.")
        return

    # Evaluate each extractor
    all_method_metrics = {}
    all_method_predictions = {}
    
    # For CSV updates
    original_df = None
    if hasattr(config, 'DATA_SOURCE') and config.DATA_SOURCE.lower() != 'synthetic':
        try:
            original_df = pd.read_csv(dataset_path)
            print(f"Loaded original CSV with {len(original_df)} rows")
        except Exception as e:
            print(f"Warning: Could not load original CSV for saving predictions: {e}")
            original_df = None
    
    with tqdm(total=len(extractors_to_compare), desc="Comparing methods", unit="method") as pbar:
        for extractor in extractors_to_compare:
            print(f"\nEvaluating {extractor.name}...")
            
            # Generate predictions
            all_predictions = run_extraction(extractor, prepared_test_data)
            all_method_predictions[extractor.name] = all_predictions
            
            # Calculate metrics
            print(f"Calculating metrics for {extractor.name}...")
            metrics = calculate_and_report_metrics(
                all_predictions,
                gold_standard,
                extractor.name,
                EXPERIMENT_OUTPUT_DIR,
                len(prepared_test_data)
            )
            all_method_metrics[extractor.name] = metrics
            
            # Save predictions to CSV if applicable
            if original_df is not None:
                safe_extractor_name = extractor.name.lower().replace(' ', '_')
                predictions_column = f"{safe_extractor_name}_predictions"
                correctness_column = f"{safe_extractor_name}_is_correct"
                
                # Create dictionaries to hold predictions and correctness by note_id
                note_predictions = {}
                note_correctness = {}
                
                # Group predictions by note_id
                for pred in all_predictions:
                    note_id = pred['note_id']
                    if note_id not in note_predictions:
                        note_predictions[note_id] = []
                    
                    note_predictions[note_id].append({
                        'diagnosis': pred['diagnosis'],
                        'date': pred['date'],
                        'confidence': pred.get('confidence', 1.0)
                    })
                
                # Check correctness if gold standard exists
                if gold_standard:
                    # Convert gold_standard to a set of (note_id, diagnosis, date) tuples for easier comparison
                    gold_set = set((g['note_id'], g['diagnosis'], g['date']) for g in gold_standard)
                    
                    # Check each prediction against the gold standard
                    for pred in all_predictions:
                        note_id = pred['note_id']
                        is_correct = (note_id, pred['diagnosis'], pred['date']) in gold_set
                        
                        if note_id not in note_correctness:
                            note_correctness[note_id] = []
                            
                        note_correctness[note_id].append(is_correct)
                
                # Add predictions to dataframe
                original_df[predictions_column] = None
                if gold_standard:
                    original_df[correctness_column] = None
                    
                # Fill in predictions and correctness columns
                for i, row in original_df.iterrows():
                    if i in note_predictions:
                        original_df.at[i, predictions_column] = json.dumps(note_predictions[i])
                        
                        if i in note_correctness:
                            original_df.at[i, correctness_column] = json.dumps(note_correctness[i])
                
                print(f"Added predictions for {extractor.name} to CSV columns")
            
            pbar.update(1)
    
    # Save the updated dataframe back to CSV
    if original_df is not None:
        try:
            original_df.to_csv(dataset_path, index=False)
            print(f"Successfully saved all predictions to {dataset_path}")
        except Exception as e:
            print(f"Error saving predictions to CSV: {e}")
    
    # Generate a comparison plot
    if all_method_metrics:
        plot_comparison(all_method_metrics)

    print("\nComparison completed!")

    # Return the metrics dict for potential future use (e.g., in notebooks)
    return all_method_metrics

def plot_comparison(method_metrics):
    """
    Plot a comparison of metrics from different extraction methods.
    
    Args:
        method_metrics (dict): Dictionary mapping extractor names to their metrics.
    """
    os.makedirs(EXPERIMENT_OUTPUT_DIR, exist_ok=True)
    print("\nGenerating comparison plot...")
    
    # Create a DataFrame from the metrics
    comparison_df = pd.DataFrame(method_metrics).T  # Transpose
    
    print("\nComparison results:")
    # Select only P, R, F1 for printing summary, but keep others for potential use
    print(comparison_df[['precision', 'recall', 'f1']].round(3))

    metrics_to_plot = ['precision', 'recall', 'f1']
    try:
        plot_df = comparison_df[[col for col in metrics_to_plot if col in comparison_df.columns]]  # Ensure columns exist
        if not plot_df.empty:
            plot_df.plot(kind='bar', figsize=(10, 6), rot=0)
            plt.title(f'Comparison of Extraction Methods - {config.DATA_SOURCE.capitalize()} Data')
            plt.ylabel('Score')
            plt.xlabel('Method')
            plt.ylim(0, 1.05)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend(title='Metric')
            plt.tight_layout()
            plot_save_path = os.path.join(EXPERIMENT_OUTPUT_DIR, f"{config.DATA_SOURCE}_extractor_comparison.png")
            plt.savefig(plot_save_path)
            print(f"\nComparison plot saved to {plot_save_path}")
            plt.close()
        else:
            print("\nSkipping comparison plot: No P/R/F1 data available.")
    except Exception as e:
        print(f"\nError generating comparison plot: {e}")

if __name__ == "__main__":
    # Read mode and method directly from config
    run_mode = config.RUN_MODE.lower()
    
    print(f"--- Running Mode: {run_mode} ---")
    
    if run_mode == 'single':
        print(f"--- Method: {config.EXTRACTION_METHOD} ---")
        test_single_note()
    elif run_mode == 'evaluate':
        print(f"--- Method: {config.EXTRACTION_METHOD} ---")
        evaluate_on_dataset()
    elif run_mode == 'compare':
        compare_all_methods()
    else:
        print(f"Error: Invalid RUN_MODE '{config.RUN_MODE}' in config.py. Options are: 'single', 'evaluate', 'compare'.") 