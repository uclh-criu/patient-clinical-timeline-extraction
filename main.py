import os
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np # Added for timeline visualization
from tqdm import tqdm
from datetime import datetime
import re

# Import from our modules
from extractors.extractor_factory import create_extractor
from utils.extraction_utils import (
    calculate_and_report_metrics,
    load_and_prepare_data,
    run_extraction,
    get_data_path,
    transform_python_to_json,
    extract_relative_dates_llm,
    aggregate_predictions_by_patient,
    generate_patient_timelines,
    generate_patient_timeline_summary
)
import config

# Define the output directory using absolute path if project_root is available
if 'project_root' not in locals() and 'project_root' not in globals():
    project_root = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_OUTPUT_DIR = os.path.join(project_root, "experiment_outputs")

def generate_patient_timeline_visualizations(patient_timelines, output_dir, extractor_name):
    """
    Generates and saves timeline visualizations for each patient based on aggregated timeline data.

    Args:
        patient_timelines (dict): Dictionary from aggregate_predictions_by_patient.
                                  Format: {patient_id: [{'diagnosis': str, 'date': str, 'confidence': float, 'note_id': int}, ...]}
        output_dir (str): Directory to save the timeline plots.
        extractor_name (str): Name of the extractor for file naming.
    """
    if not patient_timelines:
        print(f"No patient timelines to visualize for {extractor_name}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for patient_id, timeline in tqdm(patient_timelines.items(), desc="Generating timeline visualizations", unit="patient"):
        if not timeline:
            continue
            
        try:
            # Prepare data for visualization
            parsed_dates = []
            diagnoses = []
            confidences = []
            note_ids = []
            
            for entry in timeline:
                date_str = entry.get('date')
                diagnosis_str = entry.get('diagnosis')
                confidence = entry.get('confidence', 1.0)
                note_id = entry.get('note_id', '')
                
                if not date_str or not diagnosis_str:
                    continue
                    
                try:
                    # Parse the date - should already be in YYYY-MM-DD format
                    parsed_date = datetime.strptime(str(date_str), '%Y-%m-%d')
                    parsed_dates.append(parsed_date)
                    diagnoses.append(str(diagnosis_str))
                    confidences.append(confidence)
                    note_ids.append(note_id)
                except ValueError:
                    # Try other common date formats as fallback
                    date_formats = [
                        '%Y-%m-%d',           # 2024-06-30
                        '%d/%m/%Y',           # 30/06/2024  
                        '%m/%d/%Y',           # 06/30/2024
                        '%d-%m-%Y',           # 30-06-2024
                        '%d.%m.%Y',           # 30.06.2024
                        '%d.%m.%y',           # 30.06.24
                        '%d/%m/%y',           # 30/06/24
                        '%d-%m-%y',           # 30-06-24
                        '%Y.%m.%d',           # 2024.06.30
                        '%d %b %Y',           # 30 Jun 2024
                        '%d %B %Y',           # 30 June 2024
                        '%b %d %Y',           # Jun 30 2024
                        '%B %d %Y',           # June 30 2024
                        '%d %b %y',           # 30 Jun 24
                        '%d %B %y',           # 30 June 24
                    ]
                    parsed = False
                    for fmt in date_formats:
                        try:
                            parsed_date = datetime.strptime(str(date_str), fmt)
                            parsed_dates.append(parsed_date)
                            diagnoses.append(str(diagnosis_str))
                            confidences.append(confidence)
                            note_ids.append(note_id)
                            parsed = True
                            break
                        except ValueError:
                            continue
                    
                    if not parsed:
                        # Try cleaning common issues like ordinal suffixes and parentheses
                        cleaned_date = str(date_str).strip('()').strip()
                        # Remove ordinal suffixes (1st, 2nd, 3rd, 4th, etc.)
                        cleaned_date = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', cleaned_date)
                        
                        for fmt in date_formats:
                            try:
                                parsed_date = datetime.strptime(cleaned_date, fmt)
                                parsed_dates.append(parsed_date)
                                diagnoses.append(str(diagnosis_str))
                                confidences.append(confidence)
                                note_ids.append(note_id)
                                parsed = True
                                break
                            except ValueError:
                                continue
                    
                    if not parsed:
                        # Skip relative dates and other unparseable formats silently
                        # (like "3 months ago", "next week", etc.)
                        continue
                except Exception as e:
                    print(f"Warning: Could not parse date '{date_str}' for patient {patient_id}: {e}")
                    continue
            
            if not parsed_dates:
                print(f"Skipping visualization for patient {patient_id}: No valid date entries found")
                continue

            # Create a DataFrame for plotting
            timeline_df = pd.DataFrame({
                'date': parsed_dates,
                'diagnosis': diagnoses,
                'confidence': confidences,
                'note_id': note_ids
            }).sort_values(by='date')  # Sort chronologically

            # Plot the timeline
            fig, ax = plt.subplots(figsize=(14, max(6, len(timeline_df) * 0.4)))
            
            # Create y-offsets for better visibility
            if len(timeline_df) > 5:
                num_levels = min(5, len(timeline_df))  # Max 5 levels
                y_offsets = [(i % num_levels) * 0.15 - (num_levels // 2 * 0.15) for i in range(len(timeline_df))]
            else:
                y_offsets = [0.1 * (i - len(timeline_df)//2) for i in range(len(timeline_df))]

            # Color code by confidence (if available)
            colors = ['red' if c < 0.5 else 'orange' if c < 0.8 else 'green' for c in timeline_df['confidence']]
            
            # Plot points
            scatter = ax.scatter(timeline_df['date'], y_offsets, 
                               s=120, c=colors, alpha=0.7, zorder=5, edgecolors='black', linewidth=1)
            
            # Draw horizontal reference line
            ax.axhline(0, color='gray', lw=1, linestyle='--', alpha=0.5)

            # Add diagnosis labels with note information
            for i, (date_obj, diagnosis_str, confidence, note_id) in enumerate(
                zip(timeline_df['date'], timeline_df['diagnosis'], timeline_df['confidence'], timeline_df['note_id'])):
                
                vertical_pos = y_offsets[i]
                
                # Alternate text position (above/below)
                text_va = 'bottom' if i % 2 == 0 else 'top'
                text_y_offset = 0.25 if text_va == 'bottom' else -0.25
                text_y = vertical_pos + text_y_offset
                
                # Create label with diagnosis and note info
                label_text = f"{diagnosis_str}"
                if len(set(timeline_df['note_id'])) > 1:  # Only show note ID if multiple notes
                    label_text += f"\n(Note {note_id})"
                
                ax.annotate(label_text, 
                           xy=(date_obj, vertical_pos),
                           xytext=(date_obj, text_y),
                           ha='center',
                           va=text_va,
                           arrowprops=dict(arrowstyle="-", color='gray', alpha=0.6, lw=1),
                           bbox=dict(boxstyle='round,pad=0.3', fc='lightblue', alpha=0.8, 
                                   ec='black', lw=0.5),
                           fontsize=9,
                           zorder=10)
            
            # Format the plot
            ax.set_yticks([])  # Remove y-axis ticks
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_position(('outward', 10))

            # Format dates on x-axis
            plt.xticks(rotation=45, ha="right")
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            
            # Auto-format dates if many
            if len(timeline_df['date'].unique()) > 8:
                fig.autofmt_xdate()

            # Add legend for confidence colors
            legend_elements = [
                plt.scatter([], [], c='green', s=100, label='High Confidence (≥0.8)', alpha=0.7, edgecolors='black'),
                plt.scatter([], [], c='orange', s=100, label='Medium Confidence (0.5-0.8)', alpha=0.7, edgecolors='black'),
                plt.scatter([], [], c='red', s=100, label='Low Confidence (<0.5)', alpha=0.7, edgecolors='black')
            ]
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

            # Title and labels
            unique_diagnoses = len(set(timeline_df['diagnosis']))
            date_range = f"{timeline_df['date'].min().strftime('%Y-%m-%d')} to {timeline_df['date'].max().strftime('%Y-%m-%d')}"
            
            plt.title(f'Patient {patient_id} - Medical Timeline\n'
                     f'{len(timeline_df)} diagnoses, {unique_diagnoses} unique conditions\n'
                     f'({extractor_name}) | {date_range}', 
                     fontsize=12, pad=20)
            plt.xlabel('Date', fontsize=11)
            plt.ylabel('Diagnoses Timeline', fontsize=11)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the plot
            safe_extractor_name = re.sub(r'[^\w.-]+', '_', extractor_name).lower()
            plot_filename = os.path.join(output_dir, f"patient_{patient_id}_{safe_extractor_name}_visual_timeline.png")
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            print(f"Error generating visual timeline for patient {patient_id}: {e}")
            if 'fig' in locals():
                plt.close(fig)
    
    print(f"Generated {len(patient_timelines)} patient visual timeline plots in {output_dir}")

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
    
    # Only save predictions to CSV
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
    
    # Generate patient timelines if configured
    if config.GENERATE_PATIENT_TIMELINES:
        print("\nGenerating patient timelines...")
        timeline_output_dir = os.path.join(project_root, config.TIMELINE_OUTPUT_DIR)
        
        # Aggregate predictions by patient
        patient_timelines = aggregate_predictions_by_patient(all_predictions)
        
        # Generate individual timeline files
        generate_patient_timelines(patient_timelines, timeline_output_dir, extractor.name)
        
        # Generate summary report
        generate_patient_timeline_summary(patient_timelines, timeline_output_dir, extractor.name)
        
        # Generate visual timeline plots
        generate_patient_timeline_visualizations(patient_timelines, timeline_output_dir, extractor.name)
    
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
            
            # Generate patient timelines if configured
            if config.GENERATE_PATIENT_TIMELINES:
                print(f"Generating patient timelines for {extractor.name}...")
                timeline_output_dir = os.path.join(project_root, config.TIMELINE_OUTPUT_DIR)
                
                # Aggregate predictions by patient
                patient_timelines = aggregate_predictions_by_patient(all_predictions)
                
                # Generate individual timeline files
                generate_patient_timelines(patient_timelines, timeline_output_dir, extractor.name)
                
                # Generate summary report
                generate_patient_timeline_summary(patient_timelines, timeline_output_dir, extractor.name)
                
                # Generate visual timeline plots
                generate_patient_timeline_visualizations(patient_timelines, timeline_output_dir, extractor.name)
            
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
    # Select P, R, F1, and Accuracy for printing summary, but keep others for potential use
    print(comparison_df[['precision', 'recall', 'f1', 'accuracy']].round(3))

    metrics_to_plot = ['precision', 'recall', 'f1', 'accuracy']
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
            print("\nSkipping comparison plot: No precision/recall/F1/accuracy data available.")
    except Exception as e:
        print(f"\nError generating comparison plot: {e}")

if __name__ == "__main__":
    # Read mode and method directly from config
    run_mode = config.RUN_MODE.lower()
    
    print(f"--- Running Mode: {run_mode} ---")
    
    if run_mode == 'evaluate':
        print(f"--- Method: {config.EXTRACTION_METHOD} ---")
        evaluate_on_dataset()
    elif run_mode == 'compare':
        compare_all_methods()
    else:
        print(f"Error: Invalid RUN_MODE '{config.RUN_MODE}' in config.py. Options are: 'evaluate', 'compare'.") 