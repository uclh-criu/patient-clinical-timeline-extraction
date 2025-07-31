#!/usr/bin/env python3
# eda_analysis.py - Exploratory Data Analysis for clinical notes
# 
# This script performs exploratory data analysis on clinical notes,
# focusing on entities, dates, relationships, and text characteristics.

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import seaborn as sns
from tqdm import tqdm
import datetime
from sklearn.model_selection import train_test_split

# Add parent directory to path to allow importing from other modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
import config
from utils.inference_eval_utils import load_and_prepare_data, preprocess_note_for_prediction

# ===== CONFIGURABLE VARIABLES =====
# Path to the dataset to analyze - change this to point to your data file
DATA_PATH = 'data/imaging.csv'

# Entity mode to use for analysis - must match the structure of your data
# Options: 'multi_entity' (uses SNOMED_COLUMN and UMLS_COLUMN) or 'diagnosis_only' (uses DIAGNOSES_COLUMN)
ENTITY_MODE = 'diagnosis_only'

# Maximum distance between entity and date to consider for relationship analysis
MAX_DISTANCE = 500

# Number of top entities/concepts to display in reports
TOP_N = 20

# Number of samples to analyze (set to None to analyze all)
NUM_SAMPLES = None

# Output directory for saving plots and reports
OUTPUT_DIR = 'experiment_outputs/eda'

# Whether to save outputs to files
SAVE_OUTPUT = True

# Whether to perform comparative analysis between train/val/test splits
ENABLE_SPLIT_COMPARISON = True

# Whether validation data should be a separate split (True) or a subset of training data (False)
# When False, the training data used for analysis will be the full training set including validation data
# When True, the training data used for analysis will exclude validation data
VAL_AS_SEPARATE_SPLIT = True

# Plot generation flags - set to True for plots you want to generate
PLOT_CONFIG = {
    'document_length': False,      # Document length distribution plots
    'entity_distribution': False,   # Entity count and distribution plots
    'entity_category': False,       # Entity category distribution plots (multi_entity mode only)
    'class_imbalance': False,       # Class imbalance plots
    'distance_distribution': False, # Distance distribution plots for positive pairs
    'comparative': False            # Comparative plots across train/val/test splits
}

# Get data source name from the path
DATA_SOURCE = os.path.basename(DATA_PATH).split('.')[0]

# ===== END CONFIGURABLE VARIABLES =====

def analyze_document_lengths(prepared_data, split_name=None, verbose=True):
    """
    Analyzes the distribution of document lengths.
    
    Args:
        prepared_data: List of dictionaries containing preprocessed notes
        split_name: Name of the data split (train, val, test) or None
        verbose: Whether to print detailed analysis output
        
    Returns:
        dict: Statistics about document lengths
    """
    split_label = f" ({split_name})" if split_name else ""
    if verbose:
        print(f"\n===== DOCUMENT LENGTH ANALYSIS{split_label} =====")
    
    # Extract document lengths
    doc_lengths = [len(note['note']) for note in prepared_data]
    token_lengths = [len(note['note'].split()) for note in prepared_data]
    
    # Calculate statistics
    stats = {
        'document_count': len(doc_lengths),  # Store the document count directly
        'char_count': {
            'min': min(doc_lengths),
            'max': max(doc_lengths),
            'mean': np.mean(doc_lengths),
            'median': np.median(doc_lengths),
            'std': np.std(doc_lengths)
        },
        'token_count': {
            'min': min(token_lengths),
            'max': max(token_lengths),
            'mean': np.mean(token_lengths),
            'median': np.median(token_lengths),
            'std': np.std(token_lengths)
        }
    }
    
    # Print statistics if verbose
    if verbose:
        print(f"Document count: {len(doc_lengths)}")
        print("\nCharacter count statistics:")
        print(f"  Min: {stats['char_count']['min']}")
        print(f"  Max: {stats['char_count']['max']}")
        print(f"  Mean: {stats['char_count']['mean']:.2f}")
        print(f"  Median: {stats['char_count']['median']:.2f}")
        print(f"  Std Dev: {stats['char_count']['std']:.2f}")
        
        print("\nToken count statistics:")
        print(f"  Min: {stats['token_count']['min']}")
        print(f"  Max: {stats['token_count']['max']}")
        print(f"  Mean: {stats['token_count']['mean']:.2f}")
        print(f"  Median: {stats['token_count']['median']:.2f}")
        print(f"  Std Dev: {stats['token_count']['std']:.2f}")
    
    # Plot distributions if enabled
    if PLOT_CONFIG['document_length']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Character count histogram
        sns.histplot(doc_lengths, kde=True, ax=ax1)
        ax1.set_title('Document Length Distribution (Characters)')
        ax1.set_xlabel('Character Count')
        ax1.set_ylabel('Frequency')
        
        # Token count histogram
        sns.histplot(token_lengths, kde=True, ax=ax2)
        ax2.set_title('Document Length Distribution (Tokens)')
        ax2.set_xlabel('Token Count')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if SAVE_OUTPUT:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            filename = f"document_length_distribution{f'_{split_name}' if split_name else ''}.png"
            plt.savefig(os.path.join(OUTPUT_DIR, filename))
            print(f"Document length distribution plot saved to {os.path.join(OUTPUT_DIR, filename)}")
        
        plt.show()
    
    return stats

def analyze_entity_distribution(prepared_data, entity_mode, split_name=None, verbose=True):
    """
    Analyzes the distribution of entities across documents.
    
    Args:
        prepared_data: List of dictionaries containing preprocessed notes
        entity_mode: The entity mode ('diagnosis_only' or 'multi_entity')
        split_name: Name of the data split (train, val, test) or None
        verbose: Whether to print detailed analysis output
        
    Returns:
        dict: Statistics about entity distribution
    """
    split_label = f" ({split_name})" if split_name else ""
    if verbose:
        print(f"\n===== ENTITY DISTRIBUTION ANALYSIS{split_label} =====")
    
    # Extract entity counts per document
    entity_counts = []
    date_counts = []
    all_entities = []
    
    for note in prepared_data:
        entities, dates = note['entities']
        entity_counts.append(len(entities))
        date_counts.append(len(dates))
        
        # Collect all entities for frequency analysis
        for entity in entities:
            if entity_mode == 'multi_entity' and len(entity) >= 3:
                # In multi_entity mode, entity is (label, start_pos, category)
                entity_label = entity[0].lower()
                entity_category = entity[2].lower()
                all_entities.append((entity_label, entity_category))
            else:
                # In diagnosis_only mode, entity is (label, start_pos)
                entity_label = entity[0].lower()
                all_entities.append(entity_label)
    
    # Calculate statistics
    stats = {
        'entity_counts': {
            'min': min(entity_counts),
            'max': max(entity_counts),
            'mean': np.mean(entity_counts),
            'median': np.median(entity_counts),
            'std': np.std(entity_counts),
            'total': sum(entity_counts)
        },
        'date_counts': {
            'min': min(date_counts),
            'max': max(date_counts),
            'mean': np.mean(date_counts),
            'median': np.median(date_counts),
            'std': np.std(date_counts),
            'total': sum(date_counts)
        }
    }
    
    # Print statistics if verbose
    if verbose:
        print(f"Total entities: {stats['entity_counts']['total']}")
        print(f"Total dates: {stats['date_counts']['total']}")
        
        print("\nEntities per document:")
        print(f"  Min: {stats['entity_counts']['min']}")
        print(f"  Max: {stats['entity_counts']['max']}")
        print(f"  Mean: {stats['entity_counts']['mean']:.2f}")
        print(f"  Median: {stats['entity_counts']['median']:.2f}")
        print(f"  Std Dev: {stats['entity_counts']['std']:.2f}")
        
        print("\nDates per document:")
        print(f"  Min: {stats['date_counts']['min']}")
        print(f"  Max: {stats['date_counts']['max']}")
        print(f"  Mean: {stats['date_counts']['mean']:.2f}")
        print(f"  Median: {stats['date_counts']['median']:.2f}")
        print(f"  Std Dev: {stats['date_counts']['std']:.2f}")
    
    # Analyze entity frequencies
    if entity_mode == 'multi_entity':
        # Count entities by (label, category)
        entity_counter = Counter(all_entities)
        top_entities = entity_counter.most_common(TOP_N)
        
        # Count entities by category
        category_counter = Counter([entity[1] for entity in all_entities])
        
        if verbose:
            print("\nEntity category distribution:")
            for category, count in category_counter.most_common():
                percentage = (count / len(all_entities)) * 100
                print(f"  {category}: {count} ({percentage:.2f}%)")
            
            print(f"\nTop {TOP_N} most common entities:")
            for (entity, category), count in top_entities:
                print(f"  {entity} ({category}): {count}")
        
        # Plot category distribution if enabled
        if PLOT_CONFIG['entity_category']:
            plt.figure(figsize=(12, 6))
            categories = [cat for cat, _ in category_counter.most_common()]
            counts = [count for _, count in category_counter.most_common()]
            
            plt.bar(categories, counts)
            plt.title('Entity Category Distribution')
            plt.xlabel('Category')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            if SAVE_OUTPUT:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                filename = f"entity_category_distribution{f'_{split_name}' if split_name else ''}.png"
                plt.savefig(os.path.join(OUTPUT_DIR, filename))
                print(f"Entity category distribution plot saved to {os.path.join(OUTPUT_DIR, filename)}")
            
            plt.show()
    else:
        # Count entities by label only
        entity_counter = Counter(all_entities)
        top_entities = entity_counter.most_common(TOP_N)
        
        if verbose:
            print(f"\nTop {TOP_N} most common entities:")
            for entity, count in top_entities:
                print(f"  {entity}: {count}")
    
    # Plot entity and date count distributions if enabled
    if PLOT_CONFIG['entity_distribution']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Entity count histogram
        sns.histplot(entity_counts, kde=True, ax=ax1)
        ax1.set_title('Entities per Document')
        ax1.set_xlabel('Entity Count')
        ax1.set_ylabel('Frequency')
        
        # Date count histogram
        sns.histplot(date_counts, kde=True, ax=ax2)
        ax2.set_title('Dates per Document')
        ax2.set_xlabel('Date Count')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if SAVE_OUTPUT:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            filename = f"entity_date_count_distribution{f'_{split_name}' if split_name else ''}.png"
            plt.savefig(os.path.join(OUTPUT_DIR, filename))
            print(f"Entity and date count distribution plot saved to {os.path.join(OUTPUT_DIR, filename)}")
        
        plt.show()
    
    return stats

def analyze_relationship_distribution(prepared_data, relationship_gold, max_distance, split_name=None, verbose=True):
    """
    Analyzes the distribution of entity-date relationships.
    
    Args:
        prepared_data: List of dictionaries containing preprocessed notes
        relationship_gold: List of gold standard relationships
        max_distance: Maximum distance between entity and date to consider
        split_name: Name of the data split (train, val, test) or None
        verbose: Whether to print detailed analysis output
        
    Returns:
        dict: Statistics about relationship distribution
    """
    split_label = f" ({split_name})" if split_name else ""
    if verbose:
        print(f"\n===== RELATIONSHIP ANALYSIS{split_label} =====")
    
    # Create a set of gold standard relationships for quick lookup
    gold_relationships = set()
    for rel in relationship_gold:
        entity_label = rel['entity_label'].lower() if 'entity_label' in rel else rel['diagnosis'].lower()
        date = rel['date']
        gold_relationships.add((entity_label, date))
    
    # Generate all potential entity-date pairs within max_distance
    all_pairs = []
    positive_pairs = []
    negative_pairs = []
    distances = []
    positive_distances = []
    
    for note in tqdm(prepared_data, desc="Analyzing relationships"):
        note_text = note['note']
        disorders, dates = note['entities']
        
        # Generate features using preprocess_note_for_prediction
        note_features = preprocess_note_for_prediction(note_text, disorders, dates, max_distance)
        
        for feature in note_features:
            diagnosis = feature['diagnosis']
            parsed_date = feature['date']
            distance = feature['distance']
            
            # Check if this is a gold standard relationship
            key = (diagnosis.strip().lower(), parsed_date)
            is_gold = key in gold_relationships
            
            all_pairs.append((diagnosis, parsed_date, distance, is_gold))
            distances.append(distance)
            
            if is_gold:
                positive_pairs.append((diagnosis, parsed_date, distance))
                positive_distances.append(distance)
            else:
                negative_pairs.append((diagnosis, parsed_date, distance))
    
    # Calculate statistics
    stats = {
        'total_pairs': len(all_pairs),
        'positive_pairs': len(positive_pairs),
        'negative_pairs': len(negative_pairs),
        'class_imbalance_ratio': len(negative_pairs) / max(1, len(positive_pairs))
    }
    
    if positive_distances:
        stats['positive_distances'] = {
            'min': min(positive_distances),
            'max': max(positive_distances),
            'mean': np.mean(positive_distances),
            'median': np.median(positive_distances),
            'std': np.std(positive_distances)
        }
    
    # Print statistics if verbose
    if verbose:
        print(f"Total potential entity-date pairs within {max_distance} characters: {stats['total_pairs']}")
        print(f"Positive pairs (in gold standard): {stats['positive_pairs']}")
        print(f"Negative pairs: {stats['negative_pairs']}")
        print(f"Class imbalance ratio (negative:positive): {stats['class_imbalance_ratio']:.2f}:1")
        
        if positive_distances:
            print("\nDistance statistics for positive pairs:")
            print(f"  Min: {stats['positive_distances']['min']}")
            print(f"  Max: {stats['positive_distances']['max']}")
            print(f"  Mean: {stats['positive_distances']['mean']:.2f}")
            print(f"  Median: {stats['positive_distances']['median']:.2f}")
            print(f"  Std Dev: {stats['positive_distances']['std']:.2f}")
    
    # Plot distance distributions if enabled
    if PLOT_CONFIG['distance_distribution'] and positive_distances:
        plt.figure(figsize=(10, 6))
        
        # Plot histogram of distances for positive pairs
        sns.histplot(positive_distances, kde=True, label='Positive Pairs')
        plt.title('Distance Distribution for Positive Entity-Date Pairs')
        plt.xlabel('Character Distance')
        plt.ylabel('Frequency')
        plt.legend()
        
        if SAVE_OUTPUT:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            filename = f"positive_pair_distance_distribution{f'_{split_name}' if split_name else ''}.png"
            plt.savefig(os.path.join(OUTPUT_DIR, filename))
            print(f"Positive pair distance distribution plot saved to {os.path.join(OUTPUT_DIR, filename)}")
        
        plt.show()
    
    # Plot class imbalance if enabled
    if PLOT_CONFIG['class_imbalance']:
        plt.figure(figsize=(8, 6))
        plt.bar(['Positive', 'Negative'], [stats['positive_pairs'], stats['negative_pairs']])
        plt.title('Class Distribution of Entity-Date Pairs')
        plt.ylabel('Count')
        plt.yscale('log')  # Log scale to better visualize imbalance
        
        # Add count labels on top of bars
        for i, count in enumerate([stats['positive_pairs'], stats['negative_pairs']]):
            plt.text(i, count, str(count), ha='center', va='bottom')
        
        if SAVE_OUTPUT:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            filename = f"class_imbalance{f'_{split_name}' if split_name else ''}.png"
            plt.savefig(os.path.join(OUTPUT_DIR, filename))
            print(f"Class imbalance plot saved to {os.path.join(OUTPUT_DIR, filename)}")
        
        plt.show()
    
    return stats

def analyze_and_report_stats(prepared_data, relationship_gold, config, split_name=None, verbose=True):
    """
    Main function to analyze and report statistics on the dataset.
    
    Args:
        prepared_data: List of dictionaries containing preprocessed notes
        relationship_gold: List of gold standard relationships
        config: Configuration object
        split_name: Name of the data split (train, val, test) or None
        verbose: Whether to print detailed analysis output
        
    Returns:
        dict: Combined statistics from all analyses
    """
    if verbose:
        split_label = f" - {split_name.upper()} SET" if split_name else ""
        print(f"\n{'='*50}")
        print(f"EXPLORATORY DATA ANALYSIS REPORT{split_label}")
        print(f"Dataset: {DATA_PATH}")
        print(f"Entity Mode: {ENTITY_MODE}")
        print(f"{'='*50}")
    
    # Document length analysis
    doc_length_stats = analyze_document_lengths(prepared_data, split_name, verbose)
    
    # Entity distribution analysis
    entity_stats = analyze_entity_distribution(prepared_data, ENTITY_MODE, split_name, verbose)
    
    # Relationship analysis
    relationship_stats = analyze_relationship_distribution(prepared_data, relationship_gold, MAX_DISTANCE, split_name, verbose)
    
    # Combine all statistics
    all_stats = {
        'document_length': doc_length_stats,
        'entity_distribution': entity_stats,
        'relationship': relationship_stats
    }
    
    if verbose:
        print(f"\n{'='*50}")
        print("EDA ANALYSIS COMPLETE")
        print(f"{'='*50}")
    
    return all_stats

def compare_data_splits(prepared_data, relationship_gold):
    """
    Perform comparative analysis between train, validation, and test splits.
    
    Args:
        prepared_data: List of dictionaries containing preprocessed notes for the full dataset
        relationship_gold: List of gold standard relationships for the full dataset
    """
    print(f"\n{'='*50}")
    print("COMPARATIVE ANALYSIS OF TRAIN/VAL/TEST SPLITS")
    print(f"{'='*50}")
    
    # Create output directory if it doesn't exist
    if SAVE_OUTPUT:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Split the full dataset into train and test sets (80/20 split)
    # First, get a list of unique note_ids
    note_ids = list(set([note['note_id'] for note in prepared_data]))
    
    # Split the note_ids into train and test sets based on config.TRAINING_SET_RATIO
    train_test_split_ratio = 1 - config.TRAINING_SET_RATIO
    all_train_note_ids, test_note_ids = train_test_split(
        note_ids, 
        test_size=train_test_split_ratio, 
        random_state=config.DATA_SPLIT_RANDOM_SEED
    )
    
    # Split the data based on note_ids
    train_data = [note for note in prepared_data if note['note_id'] in all_train_note_ids]
    test_data = [note for note in prepared_data if note['note_id'] in test_note_ids]
    
    # Split the gold standard relationships based on note_ids
    train_rel_gold = [rel for rel in relationship_gold if rel['note_id'] in all_train_note_ids]
    test_rel_gold = [rel for rel in relationship_gold if rel['note_id'] in test_note_ids]
    
    # Create validation split from training data
    print("\nCreating validation split from training data...")
    
    # Split the all_train_note_ids into train and validation sets (80/20 split)
    train_note_ids, val_note_ids = train_test_split(
        all_train_note_ids, 
        test_size=0.2, 
        random_state=config.DATA_SPLIT_RANDOM_SEED
    )
    
    # Split the data based on note_ids
    val_data = [note for note in train_data if note['note_id'] in val_note_ids]
    
    # If validation is a separate split, use only the training portion for train analysis
    # Otherwise, use the full training data (including validation) for train analysis
    if VAL_AS_SEPARATE_SPLIT:
        new_train_data = [note for note in train_data if note['note_id'] in train_note_ids]
        # Split the gold standard relationships based on note_ids
        val_rel_gold = [rel for rel in train_rel_gold if rel['note_id'] in val_note_ids]
        new_train_rel_gold = [rel for rel in train_rel_gold if rel['note_id'] in train_note_ids]
    else:
        # Use full training data for analysis (validation is just a subset used for reporting)
        new_train_data = train_data
        new_train_rel_gold = train_rel_gold
        # Still create validation split for comparative analysis
        val_rel_gold = [rel for rel in train_rel_gold if rel['note_id'] in val_note_ids]
    
    print(f"Training set: {len(new_train_data)} notes, {len(new_train_rel_gold)} gold relationships")
    print(f"Validation set: {len(val_data)} notes, {len(val_rel_gold)} gold relationships")
    print(f"Test set: {len(test_data)} notes, {len(test_rel_gold)} gold relationships")
    
    # Run analysis on each split without verbose output
    print("\nAnalyzing data splits...")
    train_stats = analyze_and_report_stats(new_train_data, new_train_rel_gold, config, 'train', verbose=False)
    val_stats = analyze_and_report_stats(val_data, val_rel_gold, config, 'val', verbose=False)
    test_stats = analyze_and_report_stats(test_data, test_rel_gold, config, 'test', verbose=False)
    
    # Generate comparative visualizations
    generate_comparative_visualizations(train_stats, val_stats, test_stats)
    
    return {
        'train': train_stats,
        'val': val_stats,
        'test': test_stats
    }

def generate_comparative_visualizations(train_stats, val_stats, test_stats):
    """Generate visualizations comparing metrics across train, validation, and test sets."""
    print(f"\n{'='*50}")
    print("COMPARATIVE ANALYSIS ACROSS DATA SPLITS")
    print(f"{'='*50}")
    
    # Always generate the summary table regardless of plot settings
    # Extract token counts from each split for the summary table
    train_tokens = train_stats['document_length']['token_count']
    val_tokens = val_stats['document_length']['token_count']
    test_tokens = test_stats['document_length']['token_count']
    
    # Extract entity counts from each split for the summary table
    train_entities = train_stats['entity_distribution']['entity_counts']
    val_entities = val_stats['entity_distribution']['entity_counts']
    test_entities = test_stats['entity_distribution']['entity_counts']
    
    # Extract class imbalance from each split for the summary table
    train_imbalance = train_stats['relationship']['class_imbalance_ratio']
    val_imbalance = val_stats['relationship']['class_imbalance_ratio']
    test_imbalance = test_stats['relationship']['class_imbalance_ratio']
    
    # Create summary table of key metrics
    print("\nSUMMARY TABLE OF KEY METRICS ACROSS SPLITS")
    print("-" * 80)
    print(f"{'Metric':<30} {'Train':>15} {'Validation':>15} {'Test':>15}")
    print("-" * 80)
    
    # Document metrics
    print(f"{'Document count':<30} {train_stats['document_length']['document_count']:>15} {val_stats['document_length']['document_count']:>15} {test_stats['document_length']['document_count']:>15}")
    print(f"{'Avg document length (tokens)':<30} {train_tokens['mean']:>15.2f} {val_tokens['mean']:>15.2f} {test_tokens['mean']:>15.2f}")
    
    # Entity metrics
    print(f"{'Total entities':<30} {train_entities['total']:>15} {val_entities['total']:>15} {test_entities['total']:>15}")
    print(f"{'Avg entities per document':<30} {train_entities['mean']:>15.2f} {val_entities['mean']:>15.2f} {test_entities['mean']:>15.2f}")
    
    # Relationship metrics
    print(f"{'Total potential pairs':<30} {train_stats['relationship']['total_pairs']:>15} {val_stats['relationship']['total_pairs']:>15} {test_stats['relationship']['total_pairs']:>15}")
    print(f"{'Positive pairs':<30} {train_stats['relationship']['positive_pairs']:>15} {val_stats['relationship']['positive_pairs']:>15} {test_stats['relationship']['positive_pairs']:>15}")
    print(f"{'Negative pairs':<30} {train_stats['relationship']['negative_pairs']:>15} {val_stats['relationship']['negative_pairs']:>15} {test_stats['relationship']['negative_pairs']:>15}")
    print(f"{'Class imbalance ratio':<30} {train_imbalance:>15.2f} {val_imbalance:>15.2f} {test_imbalance:>15.2f}")
    
    print("-" * 80)
    
    # Save summary table to file
    if SAVE_OUTPUT:
        summary_file = os.path.join(OUTPUT_DIR, 'comparative_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("SUMMARY TABLE OF KEY METRICS ACROSS SPLITS\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Metric':<30} {'Train':>15} {'Validation':>15} {'Test':>15}\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Document count':<30} {train_stats['document_length']['document_count']:>15} {val_stats['document_length']['document_count']:>15} {test_stats['document_length']['document_count']:>15}\n")
            f.write(f"{'Avg document length (tokens)':<30} {train_tokens['mean']:>15.2f} {val_tokens['mean']:>15.2f} {test_tokens['mean']:>15.2f}\n")
            f.write(f"{'Total entities':<30} {train_entities['total']:>15} {val_entities['total']:>15} {test_entities['total']:>15}\n")
            f.write(f"{'Avg entities per document':<30} {train_entities['mean']:>15.2f} {val_entities['mean']:>15.2f} {test_entities['mean']:>15.2f}\n")
            f.write(f"{'Total potential pairs':<30} {train_stats['relationship']['total_pairs']:>15} {val_stats['relationship']['total_pairs']:>15} {test_stats['relationship']['total_pairs']:>15}\n")
            f.write(f"{'Positive pairs':<30} {train_stats['relationship']['positive_pairs']:>15} {val_stats['relationship']['positive_pairs']:>15} {test_stats['relationship']['positive_pairs']:>15}\n")
            f.write(f"{'Negative pairs':<30} {train_stats['relationship']['negative_pairs']:>15} {val_stats['relationship']['negative_pairs']:>15} {test_stats['relationship']['negative_pairs']:>15}\n")
            f.write(f"{'Class imbalance ratio':<30} {train_imbalance:>15.2f} {val_imbalance:>15.2f} {test_imbalance:>15.2f}\n")
            f.write("-" * 80)
        
        print(f"Summary table saved to {summary_file}")
    
    # Skip visualization generation if comparative plots are disabled
    if not PLOT_CONFIG['comparative']:
        print("\nComparative plot generation is disabled. Set PLOT_CONFIG['comparative'] = True to enable.")
        return
    
    # Compare document lengths
    plt.figure(figsize=(12, 6))
    
    # Extract token counts from each split
    train_tokens = train_stats['document_length']['token_count']
    val_tokens = val_stats['document_length']['token_count']
    test_tokens = test_stats['document_length']['token_count']
    
    # Create bar chart comparing document length statistics
    metrics = ['min', 'max', 'mean', 'median', 'std']
    x = np.arange(len(metrics))
    width = 0.25
    
    plt.bar(x - width, [train_tokens[m] for m in metrics], width, label='Train')
    plt.bar(x, [val_tokens[m] for m in metrics], width, label='Validation')
    plt.bar(x + width, [test_tokens[m] for m in metrics], width, label='Test')
    
    plt.xlabel('Metric')
    plt.ylabel('Token Count')
    plt.title('Document Length Comparison Across Splits')
    plt.xticks(x, metrics)
    plt.legend()
    plt.tight_layout()
    
    if SAVE_OUTPUT:
        plt.savefig(os.path.join(OUTPUT_DIR, 'comparative_document_length.png'))
        print(f"Comparative document length plot saved to {os.path.join(OUTPUT_DIR, 'comparative_document_length.png')}")
    
    plt.show()
    
    # Compare entity counts
    plt.figure(figsize=(12, 6))
    
    # Extract entity counts from each split
    train_entities = train_stats['entity_distribution']['entity_counts']
    val_entities = val_stats['entity_distribution']['entity_counts']
    test_entities = test_stats['entity_distribution']['entity_counts']
    
    # Create bar chart comparing entity count statistics
    metrics = ['min', 'max', 'mean', 'median', 'std']
    x = np.arange(len(metrics))
    width = 0.25
    
    plt.bar(x - width, [train_entities[m] for m in metrics], width, label='Train')
    plt.bar(x, [val_entities[m] for m in metrics], width, label='Validation')
    plt.bar(x + width, [test_entities[m] for m in metrics], width, label='Test')
    
    plt.xlabel('Metric')
    plt.ylabel('Entity Count')
    plt.title('Entity Count Comparison Across Splits')
    plt.xticks(x, metrics)
    plt.legend()
    plt.tight_layout()
    
    if SAVE_OUTPUT:
        plt.savefig(os.path.join(OUTPUT_DIR, 'comparative_entity_count.png'))
        print(f"Comparative entity count plot saved to {os.path.join(OUTPUT_DIR, 'comparative_entity_count.png')}")
    
    plt.show()
    
    # Compare class imbalance
    plt.figure(figsize=(10, 6))
    
    # Extract class imbalance from each split
    train_imbalance = train_stats['relationship']['class_imbalance_ratio']
    val_imbalance = val_stats['relationship']['class_imbalance_ratio']
    test_imbalance = test_stats['relationship']['class_imbalance_ratio']
    
    # Create bar chart comparing class imbalance
    splits = ['Train', 'Validation', 'Test']
    imbalance = [train_imbalance, val_imbalance, test_imbalance]
    
    plt.bar(splits, imbalance)
    plt.xlabel('Data Split')
    plt.ylabel('Negative:Positive Ratio')
    plt.title('Class Imbalance Comparison Across Splits')
    
    # Add value labels on top of bars
    for i, v in enumerate(imbalance):
        plt.text(i, v, f'{v:.2f}:1', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if SAVE_OUTPUT:
        plt.savefig(os.path.join(OUTPUT_DIR, 'comparative_class_imbalance.png'))
        print(f"Comparative class imbalance plot saved to {os.path.join(OUTPUT_DIR, 'comparative_class_imbalance.png')}")
    
    plt.show()
    
    # Create summary table of key metrics
    print("\nSUMMARY TABLE OF KEY METRICS ACROSS SPLITS")
    print("-" * 80)
    print(f"{'Metric':<30} {'Train':>15} {'Validation':>15} {'Test':>15}")
    print("-" * 80)
    
    # Document metrics - use length of token_count which has the correct document count
    print(f"{'Document count':<30} {train_stats['document_length']['document_count']:>15} {val_stats['document_length']['document_count']:>15} {test_stats['document_length']['document_count']:>15}")
    print(f"{'Avg document length (tokens)':<30} {train_tokens['mean']:>15.2f} {val_tokens['mean']:>15.2f} {test_tokens['mean']:>15.2f}")
    
    # Entity metrics
    print(f"{'Total entities':<30} {train_entities['total']:>15} {val_entities['total']:>15} {test_entities['total']:>15}")
    print(f"{'Avg entities per document':<30} {train_entities['mean']:>15.2f} {val_entities['mean']:>15.2f} {test_entities['mean']:>15.2f}")
    
    # Relationship metrics
    print(f"{'Total potential pairs':<30} {train_stats['relationship']['total_pairs']:>15} {val_stats['relationship']['total_pairs']:>15} {test_stats['relationship']['total_pairs']:>15}")
    print(f"{'Positive pairs':<30} {train_stats['relationship']['positive_pairs']:>15} {val_stats['relationship']['positive_pairs']:>15} {test_stats['relationship']['positive_pairs']:>15}")
    print(f"{'Negative pairs':<30} {train_stats['relationship']['negative_pairs']:>15} {val_stats['relationship']['negative_pairs']:>15} {test_stats['relationship']['negative_pairs']:>15}")
    print(f"{'Class imbalance ratio':<30} {train_imbalance:>15.2f} {val_imbalance:>15.2f} {test_imbalance:>15.2f}")
    
    print("-" * 80)
    
    # The summary table will be included in the consolidated JSON at the end of the analysis

def main():
    """Main function to run the EDA analysis."""
    print(f"Starting EDA analysis on {DATA_PATH} (data source: {DATA_SOURCE})...")
    
    # Create output directory if it doesn't exist
    if SAVE_OUTPUT:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Override config.ENTITY_MODE with our local setting to ensure consistent behavior
    original_entity_mode = config.ENTITY_MODE
    config.ENTITY_MODE = ENTITY_MODE
    
    # Initialize the consolidated stats dictionary
    consolidated_stats = {
        'data_source': DATA_SOURCE,
        'data_path': DATA_PATH,
        'entity_mode': ENTITY_MODE,
        'analysis_timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Load and prepare the full dataset first
    print("Loading and preparing full dataset...")
    result = load_and_prepare_data(DATA_PATH, NUM_SAMPLES, config)
    
    # Handle different return signatures based on ENTITY_MODE
    if ENTITY_MODE.lower() in ['diagnosis_only', 'disorder_only']:
        prepared_data, relationship_gold = result
        entity_gold = None
    else:
        prepared_data, entity_gold, relationship_gold, _ = result
    
    # Run the full dataset analysis (verbose=True for the full dataset)
    print("\nAnalyzing full dataset...")
    full_stats = analyze_and_report_stats(prepared_data, relationship_gold, config, verbose=True)
    consolidated_stats['full_dataset_analysis'] = full_stats
    
    if ENABLE_SPLIT_COMPARISON:
        # Perform comparative analysis between train/val/test splits using the already loaded data
        print("\nPerforming comparative analysis across data splits...")
        all_stats = compare_data_splits(prepared_data, relationship_gold)
        consolidated_stats['split_analysis'] = all_stats
        
        # Add comparative summary metrics
        comparative_summary = {
            'document_count': {
                'train': all_stats['train']['document_length']['document_count'],
                'val': all_stats['val']['document_length']['document_count'],
                'test': all_stats['test']['document_length']['document_count']
            },
            'avg_document_length': {
                'train': all_stats['train']['document_length']['token_count']['mean'],
                'val': all_stats['val']['document_length']['token_count']['mean'],
                'test': all_stats['test']['document_length']['token_count']['mean']
            },
            'total_entities': {
                'train': all_stats['train']['entity_distribution']['entity_counts']['total'],
                'val': all_stats['val']['entity_distribution']['entity_counts']['total'],
                'test': all_stats['test']['entity_distribution']['entity_counts']['total']
            },
            'avg_entities_per_document': {
                'train': all_stats['train']['entity_distribution']['entity_counts']['mean'],
                'val': all_stats['val']['entity_distribution']['entity_counts']['mean'],
                'test': all_stats['test']['entity_distribution']['entity_counts']['mean']
            },
            'positive_pairs': {
                'train': all_stats['train']['relationship']['positive_pairs'],
                'val': all_stats['val']['relationship']['positive_pairs'],
                'test': all_stats['test']['relationship']['positive_pairs']
            },
            'class_imbalance_ratio': {
                'train': all_stats['train']['relationship']['class_imbalance_ratio'],
                'val': all_stats['val']['relationship']['class_imbalance_ratio'],
                'test': all_stats['test']['relationship']['class_imbalance_ratio']
            }
        }
        consolidated_stats['comparative_summary'] = comparative_summary
    # The full dataset analysis is now always performed at the beginning
    
    # Save consolidated stats to a single JSON file
    if SAVE_OUTPUT:
        # Convert numpy values to Python native types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Recursively convert all stats to JSON-serializable format
        def convert_dict_for_json(d):
            result = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    result[k] = convert_dict_for_json(v)
                else:
                    result[k] = convert_for_json(v)
            return result
        
        json_stats = convert_dict_for_json(consolidated_stats)
        
        # Use data source name in the filename
        filename = f"eda_statistics_{DATA_SOURCE}.json"
        with open(os.path.join(OUTPUT_DIR, filename), 'w') as f:
            json.dump(json_stats, f, indent=2)
        
        print(f"\nAll statistics saved to {os.path.join(OUTPUT_DIR, filename)}")
    
    # Restore the original entity mode in config
    config.ENTITY_MODE = original_entity_mode
    
    return consolidated_stats

if __name__ == "__main__":
    main()