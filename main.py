import os
import matplotlib.pyplot as plt
import pandas as pd

# Import from our modules
from extractors.extractor_factory import create_extractor
from utils.extraction_utils import (
    extract_entities,
    calculate_and_report_metrics,
    load_and_prepare_data,
    run_extraction
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
    
    diagnoses, dates = extract_entities(CLINICAL_NOTE)
    print(f"Found {len(diagnoses)} diagnoses and {len(dates)} dates")
    
    relationships = extractor.extract(CLINICAL_NOTE, entities=(diagnoses, dates))
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
    """
    dataset_path = config.DATASET_PATH
    # Use None to use all evaluation samples (the last 20% of dataset)
    num_test_samples = None

    # Load and prepare data using the helper function
    prepared_test_data, gold_standard = load_and_prepare_data(dataset_path, num_test_samples)
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
    calculate_and_report_metrics(
        all_predictions,
        gold_standard,
        extractor.name,
        EXPERIMENT_OUTPUT_DIR,
        len(prepared_test_data)
    )

    print("\nEvaluation Done!")

def compare_all_methods():
    """
    Compare available extraction methods on the dataset.
    """
    dataset_path = config.DATASET_PATH
    # Use None to use all evaluation samples (the last 20% of dataset)
    num_test_samples = None

    # Load and prepare data once using the helper function
    prepared_test_data, gold_standard = load_and_prepare_data(dataset_path, num_test_samples)
    if prepared_test_data is None or gold_standard is None:
        print("Failed to load or prepare data. Exiting comparison.")
        return

    # Load extractors
    extractors_to_compare = []
    methods_to_try = config.COMPARISON_METHODS if hasattr(config, 'COMPARISON_METHODS') else ['custom', 'naive', 'relcat', 'llm']
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
    os.makedirs(EXPERIMENT_OUTPUT_DIR, exist_ok=True)

    for extractor in extractors_to_compare:
        print(f"\n--- Evaluating: {extractor.name} ---")

        # Generate predictions using the helper function
        all_predictions = run_extraction(extractor, prepared_test_data)

        # Calculate metrics for this extractor
        metrics = calculate_and_report_metrics(
            all_predictions,
            gold_standard,
            extractor.name,
            EXPERIMENT_OUTPUT_DIR,
            len(prepared_test_data)
        )
        all_method_metrics[extractor.name] = metrics

    # Plot comparison if multiple methods were evaluated
    if len(all_method_metrics) > 1:
        print("\nGenerating comparison plot...")
        comparison_df = pd.DataFrame(all_method_metrics).T # Transpose
        print("\nComparison results:")
        # Select only P, R, F1 for printing summary, but keep others for potential use
        print(comparison_df[['precision', 'recall', 'f1']].round(3))

        metrics_to_plot = ['precision', 'recall', 'f1']
        try:
            plot_df = comparison_df[[col for col in metrics_to_plot if col in comparison_df.columns]] # Ensure columns exist
            if not plot_df.empty:
                 plot_df.plot(kind='bar', figsize=(10, 6), rot=0)
                 plt.title('Comparison of Extraction Methods')
                 plt.ylabel('Score')
                 plt.xlabel('Method')
                 plt.ylim(0, 1.05)
                 plt.grid(axis='y', linestyle='--', alpha=0.7)
                 plt.legend(title='Metric')
                 plt.tight_layout()
                 plot_save_path = os.path.join(EXPERIMENT_OUTPUT_DIR, "extractor_comparison.png")
                 plt.savefig(plot_save_path)
                 print(f"\nComparison plot saved to {plot_save_path}")
                 plt.close()
            else:
                 print("\nSkipping comparison plot: No P/R/F1 data available.")
        except Exception as e:
             print(f"\nError generating comparison plot: {e}")

    print("\nComparison Done!")

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