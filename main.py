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
import config

# Import from unified extraction_utils module
from utils.extraction_utils import (
    calculate_and_report_metrics,
    load_and_prepare_data,
    run_extraction,
    get_data_path,
    transform_python_to_json,
    extract_relative_dates_llm,
    aggregate_predictions_by_patient,
    generate_patient_timelines,
    generate_patient_timeline_summary,
    calculate_entity_metrics
)

# Define the output directory using absolute path if project_root is available
if 'project_root' not in locals() and 'project_root' not in globals():
    project_root = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_OUTPUT_DIR = os.path.join(project_root, "experiment_outputs")

def generate_patient_timeline_visualizations(patient_timelines, output_dir, extractor_name):
    """
    Generates and saves timeline visualizations for each patient based on aggregated timeline data.

    Args:
        patient_timelines (dict): Dictionary from aggregate_predictions_by_patient.
                                  Format: {patient_id: [{'entity_label': str, 'entity_category': str, 'date': str, 'confidence': float, 'note_id': int}, ...]}
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
            entity_labels = []
            entity_categories = []
            confidences = []
            note_ids = []
            
            for entry in timeline:
                date_str = entry.get('date')
                entity_label = entry.get('entity_label')
                entity_category = entry.get('entity_category', 'unknown')
                confidence = entry.get('confidence', 1.0)
                note_id = entry.get('note_id', '')
                
                if not date_str or not entity_label:
                    continue
                    
                try:
                    # Parse the date - should already be in YYYY-MM-DD format
                    parsed_date = datetime.strptime(str(date_str), '%Y-%m-%d')
                    parsed_dates.append(parsed_date)
                    entity_labels.append(str(entity_label))
                    entity_categories.append(str(entity_category))
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
                            entity_labels.append(str(entity_label))
                            entity_categories.append(str(entity_category))
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
                                entity_labels.append(str(entity_label))
                                entity_categories.append(str(entity_category))
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
                'entity_label': entity_labels,
                'entity_category': entity_categories,
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

            # Add entity labels with note information
            for i, (date_obj, entity_str, category_str, confidence, note_id) in enumerate(
                zip(timeline_df['date'], timeline_df['entity_label'], timeline_df['entity_category'], 
                    timeline_df['confidence'], timeline_df['note_id'])):
                
                vertical_pos = y_offsets[i]
                
                # Alternate text position (above/below)
                text_va = 'bottom' if i % 2 == 0 else 'top'
                text_y_offset = 0.25 if text_va == 'bottom' else -0.25
                text_y = vertical_pos + text_y_offset
                
                # Create label with entity and note info
                label_text = f"{entity_str}\n({category_str})"
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
            unique_entities = len(set(timeline_df['entity_label']))
            unique_categories = len(set(timeline_df['entity_category']))
            date_range = f"{timeline_df['date'].min().strftime('%Y-%m-%d')} to {timeline_df['date'].max().strftime('%Y-%m-%d')}"
            
            plt.title(f'Patient {patient_id} - Medical Timeline\n'
                     f'{len(timeline_df)} entities, {unique_entities} unique entities, {unique_categories} categories\n'
                     f'({extractor_name}) | {date_range}', 
                     fontsize=12, pad=20)
            plt.xlabel('Date', fontsize=11)
            plt.ylabel('Medical Timeline', fontsize=11)
            
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
    Performs three-step evaluation:
    1. Entity extraction (NER)
    2. Relationship extraction (RE)
    3. PA likelihood prediction
    """
    # Use get_data_path to determine the dataset path
    dataset_path = get_data_path(config)
    
    # Use the NUM_TEST_SAMPLES from config instead of hardcoding to None
    num_test_samples = config.NUM_TEST_SAMPLES

    # --- 1. Load and prepare data using the helper function, passing the config ---
    # The unified load_and_prepare_data function handles both modes
    if config.ENTITY_MODE == 'disorder_only':
        # In disorder_only mode, load_and_prepare_data returns only 2 values
        prepared_test_data, relationship_gold = load_and_prepare_data(dataset_path, num_test_samples, config)
        # Set these to None as they're not used in disorder_only mode
        entity_gold = None
    else:
        # In multi_entity mode, load_and_prepare_data returns 4 values
        prepared_test_data, entity_gold, relationship_gold, _ = load_and_prepare_data(dataset_path, num_test_samples, config)
    
    if prepared_test_data is None:
        print("Failed to load or prepare data. Exiting evaluation.")
        return

    # --- 2. Entity Extraction (NER) Evaluation ---
    if config.ENTITY_MODE == 'disorder_only':
        # Skip entity metrics in disorder_only mode
        entity_metrics = {
            'precision': 0,
            'recall': 0,
            'f1': 0
        }
    else:
        entity_metrics = calculate_entity_metrics(prepared_test_data, entity_gold, EXPERIMENT_OUTPUT_DIR)

    # --- 3. Relationship Extraction (RE) Evaluation ---
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

    # Calculate and report relationship metrics
    print("\nCalculating relationship metrics...")
    os.makedirs(EXPERIMENT_OUTPUT_DIR, exist_ok=True)
    
    # Use calculate_and_report_metrics for both modes
    relationship_metrics = calculate_and_report_metrics(
        all_predictions,
        relationship_gold,
        extractor.name,
        EXPERIMENT_OUTPUT_DIR,
        len(prepared_test_data),
        dataset_path
    )
    
    # --- 4. Timeline Generation ---
    # Aggregate predictions by patient
    patient_timelines = aggregate_predictions_by_patient(all_predictions)
    
    # Save predictions to CSV
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
        
        # Determine if we're in disorder_only mode
        disorder_only_mode = (config.ENTITY_MODE == 'disorder_only')
        
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
        
        # Check correctness if relationship gold standard exists
        if relationship_gold:
            # Convert gold_standard to a set for easier comparison
            gold_set = set()
            for g in relationship_gold:
                if disorder_only_mode:
                    # In disorder_only mode, gold standard has 'diagnosis' field
                    if 'diagnosis' in g:
                        gold_set.add((g['note_id'], g['diagnosis'], g['date']))
                    elif 'entity_label' in g:
                        # Fall back to entity_label if diagnosis is not available
                        gold_set.add((g['note_id'], g['entity_label'], g['date']))
                else:
                    # In multi_entity mode, gold standard has 'entity_label' and 'entity_category' fields
                    if 'entity_label' in g and 'entity_category' in g:
                        gold_set.add((g['note_id'], g['entity_label'], g['entity_category'], g['date']))
                    elif 'entity_label' in g:
                        gold_set.add((g['note_id'], g['entity_label'], g['date']))
                    elif 'diagnosis' in g:
                        # Fall back to diagnosis if entity_label is not available
                        gold_set.add((g['note_id'], g['diagnosis'], 'disorder', g['date']))
            
            # Check each prediction against the gold standard
            for pred in all_predictions:
                note_id = pred['note_id']
                
                if disorder_only_mode:
                    # In disorder_only mode
                    if 'diagnosis' in pred:
                        is_correct = (note_id, pred['diagnosis'], pred['date']) in gold_set
                    elif 'entity_label' in pred:
                        is_correct = (note_id, pred['entity_label'], pred['date']) in gold_set
                    else:
                        is_correct = False
                else:
                    # In multi_entity mode
                    if 'entity_label' in pred and 'entity_category' in pred:
                        is_correct = (note_id, pred['entity_label'], pred['entity_category'], pred['date']) in gold_set
                    elif 'entity_label' in pred:
                        is_correct = (note_id, pred['entity_label'], pred['date']) in gold_set
                    elif 'diagnosis' in pred:
                        is_correct = (note_id, pred['diagnosis'], 'disorder', pred['date']) in gold_set or \
                                     (note_id, pred['diagnosis'], pred['date']) in gold_set
                    else:
                        is_correct = False
                
                if note_id not in note_correctness:
                    note_correctness[note_id] = []
                    
                note_correctness[note_id].append(is_correct)
        
        # Add predictions to dataframe
        df[predictions_column] = None
        if relationship_gold:
            df[correctness_column] = None
            
        # Fill in predictions and correctness columns
        for i, row in df.iterrows():
            if i in note_predictions:
                df.at[i, predictions_column] = json.dumps(note_predictions[i])
                
                if i in note_correctness:
                    df.at[i, correctness_column] = json.dumps(note_correctness[i])
            
            # Get patient_id for reference (may be used elsewhere)
            patient_id = row.get(config.REAL_DATA_PATIENT_ID_COLUMN)
        
        # Save the updated dataframe back to CSV
        df.to_csv(dataset_path, index=False)
        print(f"Successfully saved predictions to column '{predictions_column}'")
        if relationship_gold:
            print(f"Successfully saved correctness indicators to column '{correctness_column}'")

            
    except Exception as e:
        print(f"Error saving predictions to CSV: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate patient timelines if configured
    if config.GENERATE_PATIENT_TIMELINES:
        print("\nGenerating patient timelines...")
        timeline_output_dir = os.path.join(project_root, config.TIMELINE_OUTPUT_DIR)
        
        # Generate individual timeline files
        generate_patient_timelines(patient_timelines, timeline_output_dir, extractor.name)
        
        # Generate summary report
        generate_patient_timeline_summary(patient_timelines, timeline_output_dir, extractor.name)
        
        # Generate visual timeline plots
        generate_patient_timeline_visualizations(patient_timelines, timeline_output_dir, extractor.name)
    
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
    
    print("\nEvaluation Done!")

def compare_all_methods():
    """
    Compare available extraction methods on the dataset.
    Performs three-step evaluation for each method:
    1. Entity extraction (NER)
    2. Relationship extraction (RE) 
    3. PA likelihood prediction
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
    # The unified load_and_prepare_data function handles both modes
    if config.ENTITY_MODE == 'disorder_only':
        # In disorder_only mode, load_and_prepare_data returns only 2 values
        prepared_test_data, relationship_gold = load_and_prepare_data(dataset_path, num_test_samples, config)
        # Set these to None as they're not used in disorder_only mode
        entity_gold = None
    else:
        # In multi_entity mode, load_and_prepare_data returns 4 values
        prepared_test_data, entity_gold, relationship_gold, _ = load_and_prepare_data(dataset_path, num_test_samples, config)
    
    if prepared_test_data is None:
        print("Failed to load or prepare data. Exiting comparison.")
        return

    # --- 1. Entity Extraction (NER) Evaluation ---
    # This only needs to be done once as it's the same for all methods
    print("\n=== ENTITY EXTRACTION EVALUATION ===")
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
    df = None
    try:
        df = pd.read_csv(dataset_path)
        print(f"Loaded original CSV with {len(df)} rows")
    except Exception as e:
        print(f"Warning: Could not load original CSV for saving predictions: {e}")
        df = None
    
    # Determine if we're in disorder_only mode
    disorder_only_mode = (config.ENTITY_MODE == 'disorder_only')
    
    with tqdm(total=len(extractors_to_compare), desc="Comparing methods", unit="method") as pbar:
        for extractor in extractors_to_compare:
            print(f"\n=== EVALUATING {extractor.name} ===")
            
            # --- 2. Relationship Extraction (RE) Evaluation ---
            # Generate predictions
            all_predictions = run_extraction(extractor, prepared_test_data)
            all_method_predictions[extractor.name] = all_predictions
            
            # Calculate relationship metrics
            print(f"Calculating relationship metrics for {extractor.name}...")
            relationship_metrics = calculate_and_report_metrics(
                all_predictions,
                relationship_gold,
                extractor.name,
                EXPERIMENT_OUTPUT_DIR,
                len(prepared_test_data),
                dataset_path
            )
            
            # Aggregate predictions by patient
            patient_timelines = aggregate_predictions_by_patient(all_predictions)
            
            # Store metrics
            all_method_metrics[extractor.name] = {
                'relationship': relationship_metrics
            }
            
            # Generate patient timelines if configured
            if config.GENERATE_PATIENT_TIMELINES:
                print(f"Generating patient timelines for {extractor.name}...")
                timeline_output_dir = os.path.join(project_root, config.TIMELINE_OUTPUT_DIR)
                
                # Generate individual timeline files
                generate_patient_timelines(patient_timelines, timeline_output_dir, extractor.name)
                
                # Generate summary report
                generate_patient_timeline_summary(patient_timelines, timeline_output_dir, extractor.name)
                
                # Generate visual timeline plots
                generate_patient_timeline_visualizations(patient_timelines, timeline_output_dir, extractor.name)
            
            # Save predictions to CSV if applicable
            if df is not None:
                safe_extractor_name = extractor.name.lower().replace(' ', '_')
                predictions_column = f"{safe_extractor_name}_predictions"
                correctness_column = f"{safe_extractor_name}_is_correct"
                
                # Add columns to dataframe if they don't exist
                if predictions_column not in df.columns:
                    df[predictions_column] = None
                if correctness_column not in df.columns:
                    df[correctness_column] = None
                
                # Create dictionaries to hold predictions and correctness by note_id
                note_predictions = {}
                note_correctness = {}
                
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
                
                # Check correctness if gold standard exists
                if relationship_gold:
                    # Convert gold_standard to a set for easier comparison
                    gold_set = set()
                    for g in relationship_gold:
                        if disorder_only_mode:
                            # In disorder_only mode, gold standard has 'diagnosis' field
                            if 'diagnosis' in g:
                                gold_set.add((g['note_id'], g['diagnosis'], g['date']))
                            elif 'entity_label' in g:
                                # Fall back to entity_label if diagnosis is not available
                                gold_set.add((g['note_id'], g['entity_label'], g['date']))
                        else:
                            # In multi_entity mode, gold standard has 'entity_label' and 'entity_category' fields
                            if 'entity_label' in g and 'entity_category' in g:
                                gold_set.add((g['note_id'], g['entity_label'], g['entity_category'], g['date']))
                            elif 'entity_label' in g:
                                gold_set.add((g['note_id'], g['entity_label'], g['date']))
                            elif 'diagnosis' in g:
                                # Fall back to diagnosis if entity_label is not available
                                gold_set.add((g['note_id'], g['diagnosis'], 'disorder', g['date']))
                    
                    # Check each prediction against the gold standard
                    for pred in all_predictions:
                        note_id = pred['note_id']
                        
                        if disorder_only_mode:
                            # In disorder_only mode
                            if 'diagnosis' in pred:
                                is_correct = (note_id, pred['diagnosis'], pred['date']) in gold_set
                            elif 'entity_label' in pred:
                                is_correct = (note_id, pred['entity_label'], pred['date']) in gold_set
                            else:
                                is_correct = False
                        else:
                            # In multi_entity mode
                            if 'entity_label' in pred and 'entity_category' in pred:
                                is_correct = (note_id, pred['entity_label'], pred['entity_category'], pred['date']) in gold_set
                            elif 'entity_label' in pred:
                                is_correct = (note_id, pred['entity_label'], pred['date']) in gold_set
                            elif 'diagnosis' in pred:
                                is_correct = (note_id, pred['diagnosis'], 'disorder', pred['date']) in gold_set or \
                                             (note_id, pred['diagnosis'], pred['date']) in gold_set
                            else:
                                is_correct = False
                        
                        if note_id not in note_correctness:
                            note_correctness[note_id] = []
                            
                        note_correctness[note_id].append(is_correct)
                
                # Add predictions to dataframe
                for i, row in df.iterrows():
                    note_id = row.get('note_id', i)
                    
                    # Add predictions
                    if note_id in note_predictions:
                        df.at[i, predictions_column] = json.dumps(note_predictions[note_id])
                    
                    # Add correctness indicators
                    if note_id in note_correctness:
                        df.at[i, correctness_column] = json.dumps(note_correctness[note_id])
                
                # Save the updated dataframe
                try:
                    df.to_csv(dataset_path, index=False)
                    print(f"Successfully saved predictions to {dataset_path}")
                except Exception as e:
                    print(f"Error saving predictions to CSV: {e}")
            
            pbar.update(1)
    
    # Print final comparison summary
    print("\n=== FINAL COMPARISON SUMMARY ===")
    
    # Entity metrics (same for all methods)
    print("\nEntity Extraction (NER) Metrics:")
    print(f"  Precision: {entity_metrics['precision']:.3f}")
    print(f"  Recall:    {entity_metrics['recall']:.3f}")
    print(f"  F1 Score:  {entity_metrics['f1']:.3f}")
    
    # Relationship metrics comparison
    print("\nRelationship Extraction (RE) Metrics:")
    print(f"{'Method':<20} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
    print("-" * 50)
    for method_name, metrics in all_method_metrics.items():
        rel_metrics = metrics['relationship']
        print(f"{method_name:<20} {rel_metrics['precision']:.3f}{'':>5} {rel_metrics['recall']:.3f}{'':>5} {rel_metrics['f1']:.3f}{'':>5}")
    
    # Generate comparison plots
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
    print("\nGenerating comparison plots...")
    
    # Extract relationship metrics into a DataFrame
    relationship_data = {}
    for method_name, metrics in method_metrics.items():
        rel_metrics = metrics['relationship']
        relationship_data[method_name] = {
            'precision': rel_metrics['precision'],
            'recall': rel_metrics['recall'],
            'f1': rel_metrics['f1'],
            'accuracy': rel_metrics.get('accuracy', 0)
        }
    
    relationship_df = pd.DataFrame(relationship_data).T
    
    print("\nRelationship Extraction Comparison Results:")
    print(relationship_df[['precision', 'recall', 'f1', 'accuracy']].round(3))
    
    try:
        plt.figure(figsize=(10, 6))
        relationship_df[['precision', 'recall', 'f1', 'accuracy']].plot(kind='bar', rot=0)
        plt.title(f'Relationship Extraction Comparison - {config.DATA_SOURCE.capitalize()} Data')
        plt.ylabel('Score')
        plt.xlabel('Method')
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Metric')
        plt.tight_layout()
        
        plot_save_path = os.path.join(EXPERIMENT_OUTPUT_DIR, f"{config.DATA_SOURCE}_relationship_comparison.png")
        plt.savefig(plot_save_path)
        print(f"Relationship comparison plot saved to {plot_save_path}")
        plt.close()
    except Exception as e:
        print(f"Error generating relationship comparison plot: {e}")

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