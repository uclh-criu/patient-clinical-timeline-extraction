import re
import os
import ast
from datetime import datetime
import json
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Get the appropriate data path based on the config
def get_data_path(config):
    """
    Get the correct dataset path based on the config.DATA_SOURCE setting.
    
    Args:
        config: Configuration object with DATA_SOURCE and path attributes.
        
    Returns:
        str: Path to the dataset file.
    """
    data_source = config.DATA_SOURCE.lower()
    
    if data_source == 'synthetic':
        return config.SYNTHETIC_DATA_PATH
    elif data_source == 'synthetic_updated':
        return config.SYNTHETIC_UPDATED_DATA_PATH
    elif data_source == 'sample':
        return config.SAMPLE_DATA_PATH
    elif data_source == 'imaging':
        return config.IMAGING_DATA_PATH
    elif data_source == 'notes':
        return config.NOTES_DATA_PATH
    elif data_source == 'letters':
        return config.LETTERS_DATA_PATH
    elif data_source == 'nph':
        return config.NPH_DATA_PATH
    else:
        raise ValueError(f"Unknown data source: {data_source}. Valid options are: 'synthetic', 'synthetic_updated', 'sample', 'imaging', 'notes', 'letters', 'nph'")

def safe_json_loads(data_string):
    """
    Safely parses a JSON-like string.
    Tries json.loads first, then ast.literal_eval for single quotes and Python literals.
    Handles JSON's true/false/null for ast.literal_eval.

    Args:
        data_string (str): The string to parse.

    Returns:
        A Python object (e.g., dict, list).

    Raises:
        json.JSONDecodeError: if parsing fails with all methods.
    """
    if not isinstance(data_string, str):
        # To mimic json.loads, which raises TypeError for non-str/bytes/bytearray
        raise TypeError(f"the JSON object must be str, bytes or bytearray, not {type(data_string).__name__}")
    
    try:
        return json.loads(data_string)
    except json.JSONDecodeError as e:
        try:
            # ast.literal_eval can't handle 'true', 'false', 'null'. Let's replace them.
            s = re.sub(r'\btrue\b', 'True', data_string, flags=re.IGNORECASE)
            s = re.sub(r'\bfalse\b', 'False', s, flags=re.IGNORECASE)
            s = re.sub(r'\bnull\b', 'None', s)
            return ast.literal_eval(s)
        except (ValueError, SyntaxError, MemoryError) as e2:
            # If all else fails, raise the original error to be caught by the caller.
            raise json.JSONDecodeError(f"Failed to parse with all methods: {e2}", data_string, 0) from e

def transform_python_to_json(python_string):
    """
    Transform a Python-style string with single quotes and datetime objects 
    into a valid JSON string with double quotes.
    
    Args:
        python_string: A string representing Python objects (like a list of dicts with single quotes)
        
    Returns:
        A valid JSON string with all strings double-quoted
    """
    if not python_string or pd.isna(python_string):
        return "[]"  # Return empty JSON array if input is empty or NaN
        
    # Handle datetime.date objects by converting them to ISO format strings
    # Example: datetime.date(2019, 4, 18) -> "2019-04-18"
    date_pattern = r"datetime\.date\((\d+),\s*(\d+),\s*(\d+)\)"
    def date_replacer(match):
        year, month, day = map(int, match.groups())
        return f'"{year:04d}-{month:02d}-{day:02d}"'
    
    # Apply the datetime replacement
    python_string_cleaned = re.sub(date_pattern, date_replacer, python_string)
    
    try:
        # Use ast.literal_eval to safely parse the Python literal
        python_obj = ast.literal_eval(python_string_cleaned)
        
        # Convert to valid JSON string with double quotes
        return json.dumps(python_obj)
    except (SyntaxError, ValueError) as e:
        # If ast.literal_eval fails, return an empty JSON array
        print(f"Warning: Failed to parse Python-style string: {e}")
        return "[]"

# Clean and preprocess text for model input
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Replace special characters
    text = re.sub(r'[^\w\s\.]', ' ', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def _parse_entity_column(data_string):
    """Helper to parse a JSON-like string from an entity column."""
    if not data_string or pd.isna(data_string):
        return []
    try:
        # First, ensure it's valid JSON, as it may be a Python literal string
        valid_json_string = transform_python_to_json(data_string)
        # Then, safely load the JSON
        entities = safe_json_loads(valid_json_string)
        
        # Process each entity to add a 'category' field based on 'categories'
        for entity in entities:
            # If entity has 'categories' but not 'category', add 'category' field
            if 'categories' in entity and not 'category' in entity:
                # Use the first category as the main category
                if isinstance(entity['categories'], list) and len(entity['categories']) > 0:
                    entity['category'] = entity['categories'][0].lower()
                else:
                    entity['category'] = 'unknown'
        
        return entities
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Warning: Could not parse entity data string. Error: {e}. Data: '{str(data_string)[:100]}...'")
        return []

def load_and_prepare_data(dataset_path, num_samples, config=None, data_split_mode='all'):
    """
    Loads dataset, selects samples, prepares gold standard, and pre-extracts entities.
    Supports real data from CSV files.
    
    Args:
        dataset_path (str): Path to the dataset file (CSV). If None, will use config.DATA_SOURCE to determine path.
        num_samples (int): Maximum number of samples to use (if provided).
        config: Configuration object containing paths and column names.
        data_split_mode (str): How to split the data. Options:
            - 'train': Return only the training portion (first TRAINING_SET_RATIO)
            - 'test': Return only the testing portion (remaining 1-TRAINING_SET_RATIO)
            - 'all': Return all data without splitting (default)
        
    Returns:
        tuple: In multi_entity mode: (prepared_test_data, entity_gold, relationship_gold, pa_likelihood_gold) or (None, None, None, None) if loading fails.
               In disorder_only mode: (prepared_test_data, relationship_gold) with entity_gold and pa_likelihood_gold set to None.
               
               prepared_test_data is a list of dicts {'patient_id': ..., 'note_id': ..., 'note': ..., 'entities': ...}.
               entity_gold is a list of dicts {'note_id': ..., 'entity_label': ..., 'entity_category': ..., 'start': ..., 'end': ...}.
               relationship_gold is a list of dicts {'note_id': ..., 'patient_id': ..., 'entity_label': ..., 'entity_category': ..., 'date': ...}.
               pa_likelihood_gold is a dict mapping patient_id to likelihood value.
    """
    # Determine if we're in disorder_only mode
    entity_mode = getattr(config, 'ENTITY_MODE', 'multi_entity')
    disorder_only_mode = (entity_mode == 'disorder_only' or entity_mode == 'diagnosis_only')
    
    # Print diagnostic information
    print(f"\n===== LOAD_AND_PREPARE_DATA =====")
    print(f"Entity mode: {entity_mode}")
    print(f"Data split mode: {data_split_mode}")
    
        # If dataset_path is not provided and config is available, use config to determine path
    if dataset_path is None and config:
        if hasattr(config, 'DATA_SOURCE'):
            dataset_path = get_data_path(config)
        elif hasattr(config, 'DATA_PATH'):
            dataset_path = config.DATA_PATH
            
    if not dataset_path:
        print(f"Error: No dataset path provided and could not determine path from config")
        return (None, None) if disorder_only_mode else (None, None, None, None)
    
    text_column = config.TEXT_COLUMN
    patient_id_column = getattr(config, 'PATIENT_ID_COLUMN', None)
    note_id_column = getattr(config, 'NOTE_ID_COLUMN', None)
    
    # Get column names for gold standards
    entity_gold_col = getattr(config, 'ENTITY_GOLD_COLUMN', None)
    relationship_gold_col = getattr(config, 'RELATIONSHIP_GOLD_COLUMN', None)
    pa_likelihood_gold_col = getattr(config, 'PA_LIKELIHOOD_GOLD_COLUMN', None)
    legacy_gold_col = getattr(config, 'RELATIONSHIP_GOLD_COLUMN', None)
    
    # Determine which columns to use for entity extraction based on mode
    if disorder_only_mode:
        # In disorder_only mode, we use diagnoses_column for disorders
        diagnoses_column = getattr(config, 'DIAGNOSES_COLUMN', None)
        dates_column = getattr(config, 'DATES_COLUMN', None)
        print(f"Using disorder_only mode with columns: diagnoses={diagnoses_column}, dates={dates_column}")
    else:
        # In multi_entity mode, we use snomed_column and umls_column for entities
        snomed_column = getattr(config, 'SNOMED_COLUMN', None)
        umls_column = getattr(config, 'UMLS_COLUMN', None)
        dates_column = getattr(config, 'DATES_COLUMN', None)
        print(f"Using multi_entity mode with columns: snomed={snomed_column}, umls={umls_column}, dates={dates_column}")
    
    timestamp_column = getattr(config, 'TIMESTAMP_COLUMN', None)
    
    # Print available columns for debugging
    print(f"Dataset path: {dataset_path}")
    print(f"Text column: {text_column}")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return (None, None) if disorder_only_mode else (None, None, None, None)
    
    print(f"Loading dataset from {dataset_path}...")
    try:
        # Read the CSV file
        df = pd.read_csv(dataset_path)
        
        # Print available columns for debugging
        print(f"Available columns in CSV: {list(df.columns)}")
        
        # Check if the text column exists
        if text_column not in df.columns:
            print(f"Error: Text column '{text_column}' not found in CSV. Available columns: {list(df.columns)}")
            return (None, None) if disorder_only_mode else (None, None, None, None)
            
        print(f"Found {len(df)} records in CSV.")
        
        # Apply train/test split if requested
        if data_split_mode != 'all' and hasattr(config, 'TRAINING_SET_RATIO') and hasattr(config, 'DATA_SPLIT_RANDOM_SEED'):
            # Get the split ratio and random seed from config
            train_ratio = config.TRAINING_SET_RATIO
            random_seed = config.DATA_SPLIT_RANDOM_SEED
            
            # Calculate the split index
            split_idx = int(len(df) * train_ratio)
            
            # Sort the dataframe to ensure consistent splits
            # We use the index as the sort key to maintain reproducibility
            df = df.sort_index(kind='stable')
            
            # Apply the split
            if data_split_mode == 'train':
                df = df.iloc[:split_idx]
                print(f"Using training split: {len(df)} records (first {train_ratio*100:.0f}%)")
            elif data_split_mode == 'test':
                df = df.iloc[split_idx:]
                print(f"Using testing split: {len(df)} records (last {(1-train_ratio)*100:.0f}%)")
        
        # If a specific number of samples is requested, limit to that
        # This happens AFTER the train/test split, so INFERENCE_SAMPLES limits the samples from the selected split
        if num_samples and num_samples < len(df):
            df = df.iloc[:num_samples]
            print(f"Limiting to {num_samples} samples from the {'training' if data_split_mode == 'train' else 'testing' if data_split_mode == 'test' else 'entire'} dataset.")
        
        # Note: Relative date extraction is now handled by extract_relative_dates.py
        relative_date_extraction_enabled = False
            
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return (None, None) if disorder_only_mode else (None, None, None, None)
    
    # Initialize gold standard data structures
    entity_gold = []
    relationship_gold = []
    pa_likelihood_gold = {}
    
    # --- Process Relationship Gold Standard ---
    # First check for the preferred column name
    gold_column = relationship_gold_col if relationship_gold_col and relationship_gold_col in df.columns else legacy_gold_col
    
    if gold_column and gold_column in df.columns:
        print(f"Found relationship gold standard column: {gold_column}. Processing gold standard data...")
        
        with tqdm(total=len(df), desc="Preparing relationship gold standard", unit="note") as pbar:
            for i, row in df.iterrows():
                # Check if the gold standard cell is not empty
                gold_data = row.get(gold_column)
                if pd.notna(gold_data) and gold_data:
                    try:
                        # Parse the JSON string in the gold standard column
                        gold_json = safe_json_loads(gold_data)
                        
                        # Check if this is the enhanced format (array of objects) or original format (object with 'relationships')
                        if isinstance(gold_json, list):
                            # Enhanced format - could have different structures
                            for entry in gold_json:
                                # Handle both formats: direct entity-date pairs or date-diagnoses structure
                                if 'entity_label' in entry and 'date' in entry:
                                    # Direct entity-date pair
                                    note_id = row.get(note_id_column) if note_id_column and note_id_column in df.columns else i
                                    relationship_gold.append({
                                        'note_id': note_id,
                                        'patient_id': row.get(patient_id_column) if patient_id_column else None,
                                        'entity_label': str(entry['entity_label']).lower(),
                                        'entity_category': str(entry.get('entity_category', 'disorder')).lower(),
                                        'date': entry['date']
                                    })
                                elif 'diagnosis' in entry and 'date' in entry:
                                    # Legacy direct diagnosis-date pair
                                    note_id = row.get(note_id_column) if note_id_column and note_id_column in df.columns else i
                                    relationship_gold.append({
                                        'note_id': note_id,
                                        'patient_id': row.get(patient_id_column) if patient_id_column else None,
                                        'entity_label': str(entry['diagnosis']).lower(),
                                        'entity_category': 'disorder',  # Default for legacy format
                                        'date': entry['date']
                                    })
                                elif 'date' in entry and 'diagnoses' in entry:
                                    # Date with multiple diagnoses
                                    date_str = entry.get('date')
                                    if not date_str:
                                        continue
                                    
                                    try:
                                        # Parse date string which might have non-padded month/day
                                        date_parts = str(date_str).split('-')
                                        if len(date_parts) == 3:
                                            year, month, day = date_parts
                                            normalized_date = f"{year}-{int(month):02d}-{int(day):02d}"
                                        else:
                                            normalized_date = date_str  # Keep original if not in expected format
                                    except (ValueError, TypeError):
                                        print(f"Warning: Could not normalize date '{date_str}' in gold standard for row {i}. Using as-is.")
                                        normalized_date = date_str
                                
                                # Process each diagnosis associated with this date
                                diagnoses = entry.get('diagnoses', [])
                                for diag in diagnoses:
                                    if 'diagnosis' in diag:
                                        note_id = row.get(note_id_column) if note_id_column and note_id_column in df.columns else i
                                        relationship_gold.append({
                                            'note_id': note_id,
                                            'patient_id': row.get(patient_id_column) if patient_id_column else None,
                                            'entity_label': str(diag['diagnosis']).lower(),
                                                'entity_category': 'disorder',  # Default for legacy format
                                                'date': normalized_date
                                        })
                        # Original format with 'relationships' key
                        elif isinstance(gold_json, dict) and 'relationships' in gold_json and isinstance(gold_json['relationships'], list):
                            for rel in gold_json['relationships']:
                                if 'diagnosis' in rel and 'date' in rel:
                                    # Normalize the date from the gold standard
                                    date_str = rel.get('date')
                                    try:
                                        date_parts = str(date_str).split('-')
                                        if len(date_parts) == 3:
                                            year, month, day = date_parts
                                            normalized_date = f"{year}-{int(month):02d}-{int(day):02d}"
                                        else:
                                            normalized_date = date_str  # Keep original if not in expected format
                                    except (ValueError, TypeError):
                                        print(f"Warning: Could not normalize date '{date_str}' in legacy gold standard for row {i}. Using as-is.")
                                        normalized_date = date_str

                                    note_id = row.get(note_id_column) if note_id_column and note_id_column in df.columns else i
                                    relationship_gold.append({
                                        'note_id': note_id,
                                        'patient_id': row.get(patient_id_column) if patient_id_column else None,
                                        'entity_label': str(rel['diagnosis']).lower(),
                                        'entity_category': 'disorder',  # Default for legacy format
                                        'date': normalized_date # Normalized YYYY-MM-DD format
                                    })
                        else:
                            print(f"Warning: Unrecognized gold standard format for row {i}")
                    except (json.JSONDecodeError, TypeError, KeyError) as e:
                        print(f"Warning: Could not parse relationship gold standard for row {i}: {e}")
                
                pbar.update(1)
        
        print(f"Prepared relationship gold standard with {len(relationship_gold)} relationships.")
    else:
        print("No relationship gold standard column found or specified. Evaluation metrics will not be calculated.")
    
    # --- Process Entity Gold Standard (only in multi_entity mode) ---
    if not disorder_only_mode and entity_gold_col and entity_gold_col in df.columns:
        print("Found entity gold standard column. Processing entity gold data...")
        
        with tqdm(total=len(df), desc="Preparing entity gold standard", unit="note") as pbar:
            for i, row in df.iterrows():
                # Check if the gold standard cell is not empty
                gold_data = row.get(entity_gold_col)
                if pd.notna(gold_data) and gold_data:
                    try:
                        # Parse the JSON string in the gold standard column
                        gold_json = safe_json_loads(gold_data)
                        
                        # Process each entity in the gold standard
                        for entity in gold_json:
                            note_id = row.get(note_id_column) if note_id_column and note_id_column in df.columns else i
                            entity_gold.append({
                                'note_id': note_id,
                                'patient_id': row.get(patient_id_column) if patient_id_column else None,
                                'entity_label': str(entity.get('entity_label', '')).lower(),
                                'entity_category': str(entity.get('entity_category', 'unknown')).lower(),
                                'start': entity.get('start', 0),
                                'end': entity.get('end', 0)
                            })
                    except (json.JSONDecodeError, TypeError, KeyError) as e:
                        print(f"Warning: Could not parse entity gold standard for row {i}: {e}")
                
                pbar.update(1)
        
        print(f"Prepared entity gold standard with {len(entity_gold)} entities.")
    elif not disorder_only_mode:
        print("No entity gold standard column found or specified.")
    
    # Pre-extract entities from the text column
    print("Pre-extracting entities...")
    prepared_test_data = []
    
    # Diagnostic counters
    notes_with_entities = 0
    total_entities = 0
    total_dates = 0
    
    with tqdm(total=len(df), desc="Processing annotations", unit="note") as pbar:
        for i, row in df.iterrows():
            # Get the text from the specified column
            text = str(row.get(text_column, ''))
            
            # This will hold the full dictionaries of extracted entities for NER evaluation
            all_extracted_entities = []
            
            # Initialize entities as empty lists in case we can't get valid annotations
            entities = ([], [])
            
            # Determine which entity extraction approach to use based on mode
            if disorder_only_mode:
                # --- DISORDER ONLY MODE ENTITY EXTRACTION ---
                # Check if we have pre-annotated entities in the CSV
                use_annotations = diagnoses_column and dates_column and diagnoses_column in df.columns and dates_column in df.columns
                
                # Comment out Row diagnostic information
                """
                if i < 3:  # Just for the first few rows
                    print(f"\nRow {i} - Using disorder_only annotations: {use_annotations}")
                    if use_annotations:
                        print(f"  Diagnoses column: {diagnoses_column}")
                        print(f"  Dates column: {dates_column}")
                """
                
                if use_annotations:
                    # Get the annotations from the CSV
                    diagnoses_data = row.get(diagnoses_column)
                    dates_data = row.get(dates_column)
                    
                    # Comment out raw data prints
                    """
                    # Print raw data for first few rows
                    if i < 3:
                        print(f"Row {i} - Raw diagnoses data: {str(diagnoses_data)[:100]}...")
                        print(f"Row {i} - Raw dates data: {str(dates_data)[:100]}...")
                    """
                    
                    if pd.notna(diagnoses_data) and pd.notna(dates_data):
                        try:
                            # Parse diagnoses annotations from JSON format
                            diagnoses_list = []
                            # Transform Python-style string to valid JSON before parsing
                            valid_diagnoses_json = transform_python_to_json(diagnoses_data)
                            disorders = safe_json_loads(valid_diagnoses_json)
                            for disorder in disorders:
                                label = disorder.get('label', '')
                                start_pos = disorder.get('start', 0)
                                diagnoses_list.append((label.lower(), start_pos))
                            
                            # Parse dates annotations from JSON format
                            dates_list = []
                            # Transform Python-style string to valid JSON before parsing
                            valid_dates_json = transform_python_to_json(dates_data)
                            formatted_dates = safe_json_loads(valid_dates_json)
                            for date_obj in formatted_dates:
                                parsed_date = date_obj.get('parsed', '')
                                original_date = date_obj.get('original', '')
                                start_pos = date_obj.get('start', 0)
                                dates_list.append((parsed_date, original_date, start_pos))
                            
                            # Check for LLM-extracted dates (from extract_relative_dates.py)
                            if 'llm_extracted_dates' in df.columns:
                                llm_dates_data = row.get('llm_extracted_dates')
                                if pd.notna(llm_dates_data) and llm_dates_data:
                                    try:
                                        # Parse the LLM-extracted dates
                                        llm_dates = safe_json_loads(llm_dates_data)
                                        for date_obj in llm_dates:
                                            parsed_date = date_obj.get('parsed', '')
                                            original_date = date_obj.get('original', '')
                                            start_pos = date_obj.get('start', 0)
                                            dates_list.append((parsed_date, original_date, start_pos))
                                            
                                        print(f"Added {len(llm_dates)} LLM-extracted dates for row {i}")
                                    except Exception as e:
                                        print(f"Error parsing LLM-extracted dates for row {i}: {e}")
                            
                            # If we found entities, use them
                            if diagnoses_list or dates_list:
                                entities = (diagnoses_list, dates_list)
                                
                                # Update counters
                                if diagnoses_list and dates_list:
                                    notes_with_entities += 1
                                total_entities += len(diagnoses_list)
                                total_dates += len(dates_list)
                                
                                # Comment out entity debugging prints
                                """
                                if i < 3:  # Just for debugging, show first few entities
                                    print(f"Row {i} diagnoses: {diagnoses_list[:2]}...")
                                    print(f"Row {i} dates: {dates_list[:2]}...")
                                """
                        except Exception as e:
                            # Print detailed error
                            if i < 10:
                                print(f"ERROR parsing entities for row {i}: {str(e)}")
                                print(f"  diagnoses_data type: {type(diagnoses_data)}")
                                print(f"  dates_data type: {type(dates_data)}")
                                
                                # Try to show the problematic part of the data
                                if isinstance(diagnoses_data, str):
                                    print(f"  diagnoses_data sample: {diagnoses_data[:100]}...")
                                if isinstance(dates_data, str):
                                    print(f"  dates_data sample: {dates_data[:100]}...")
            else:
                # --- MULTI ENTITY MODE ENTITY EXTRACTION ---
                # In multi_entity mode, we load pre-annotated entities from CSV columns
                entities_list_for_rel_extraction = []
                dates_list = []
                
                # Initialize column variables
                snomed_column = config.SNOMED_COLUMN if hasattr(config, 'SNOMED_COLUMN') else None
                umls_column = config.UMLS_COLUMN if hasattr(config, 'UMLS_COLUMN') else None
                
                # Process SNOMED entities if available
                if snomed_column and snomed_column in df.columns:
                    snomed_data = row.get(snomed_column)
                    if pd.notna(snomed_data) and snomed_data:
                        try:
                            # Parse entities from the SNOMED column
                            snomed_entities = _parse_entity_column(snomed_data)
                            all_extracted_entities.extend(snomed_entities)
                        except Exception as e:
                            print(f"Error parsing SNOMED entities for row {i}: {e}")
                
                # Process UMLS entities if available
                if umls_column and umls_column in df.columns:
                    umls_data = row.get(umls_column)
                    if pd.notna(umls_data) and umls_data:
                        try:
                            # Parse entities from the UMLS column
                            umls_entities = _parse_entity_column(umls_data)
                            all_extracted_entities.extend(umls_entities)
                        except Exception as e:
                            print(f"Error parsing UMLS entities for row {i}: {e}")
                
                # Convert the full entity dicts to the (label, start_pos, category) tuple format
                # required by the downstream relationship extraction models.
                for entity in all_extracted_entities:
                    label = entity.get('label', '')
                    start_pos = entity.get('start', 0)
                    category = entity.get('category', 'disorder').lower()  # Get category, default to 'disorder' if missing
                    if label:
                        entities_list_for_rel_extraction.append((label.lower(), start_pos, category))
                
                # Process dates
                if dates_column and dates_column in df.columns:
                    dates_data = row.get(dates_column)
                    if pd.notna(dates_data) and dates_data:
                        try:
                            # Transform Python-style string to valid JSON before parsing
                            valid_dates_json = transform_python_to_json(dates_data)
                            formatted_dates = safe_json_loads(valid_dates_json)
                            for date_obj in formatted_dates:
                                parsed_date = date_obj.get('parsed', '')
                                original_date = date_obj.get('original', '')
                                start_pos = date_obj.get('start', 0)
                                dates_list.append((parsed_date, original_date, start_pos))
                        except Exception as e:
                            print(f"Error parsing dates for row {i}: {e}")
                
                # Check for LLM-extracted dates (from extract_relative_dates.py)
                if 'llm_extracted_dates' in df.columns:
                    llm_dates_data = row.get('llm_extracted_dates')
                    if pd.notna(llm_dates_data) and llm_dates_data:
                        try:
                            # Parse the LLM-extracted dates
                            llm_dates = safe_json_loads(llm_dates_data)
                            for date_obj in llm_dates:
                                parsed_date = date_obj.get('parsed', '')
                                original_date = date_obj.get('original', '')
                                start_pos = date_obj.get('start', 0)
                                dates_list.append((parsed_date, original_date, start_pos))
                                
                            print(f"Added {len(llm_dates)} LLM-extracted dates for row {i}")
                        except Exception as e:
                            print(f"Error parsing LLM-extracted dates for row {i}: {e}")
                
                # Update entities tuple with the combined data
                entities = (entities_list_for_rel_extraction, dates_list)
                
                # Update counters
                if entities_list_for_rel_extraction or dates_list:
                    notes_with_entities += 1
                total_entities += len(entities_list_for_rel_extraction)
                total_dates += len(dates_list)
            
            # Note: Relative date extraction is now handled by extract_relative_dates.py
            # Add the prepared data entry - MOVED OUTSIDE THE IF/ELSE BLOCK
            # Use the note_id from the data if available, otherwise fall back to the row index
            note_id = row.get(note_id_column) if note_id_column and note_id_column in df.columns else i
            prepared_test_data.append({
                'note_id': note_id,
                'patient_id': row.get(patient_id_column) if patient_id_column else None,
                'note': text,
                'entities': entities,
                'extracted_entities': all_extracted_entities  # Add this for NER evaluation
            })
            
            pbar.update(1)
    
    # Print summary of entity extraction
    print("\n===== ENTITY EXTRACTION SUMMARY =====")
    print(f"Total notes processed: {len(prepared_test_data)}")
    print(f"Notes with entities: {notes_with_entities}/{len(prepared_test_data)} ({notes_with_entities/len(prepared_test_data)*100:.1f}%)")
    print(f"Total entities extracted: {total_entities}")
    print(f"Total dates extracted: {total_dates}")
    print(f"Average entities per note: {total_entities/max(1, len(prepared_test_data)):.2f}")
    print(f"Average dates per note: {total_dates/max(1, len(prepared_test_data)):.2f}")
    print("===== END ENTITY EXTRACTION SUMMARY =====\n")
    
    # Note: Relative date extraction is now handled by extract_relative_dates.py
    
    # Return appropriate values based on mode
    if disorder_only_mode:
        return prepared_test_data, relationship_gold
    else:
        return prepared_test_data, entity_gold, relationship_gold, pa_likelihood_gold

def preprocess_note_for_prediction(note, diagnoses, dates, MAX_DISTANCE=500):
    """Preprocess features from a clinical note for prediction using pre-extracted entities.
    
    Args:
        note (str): The clinical note text
        diagnoses (list): List of (diagnosis, position) tuples or (diagnosis, position, category) tuples
        dates (list): List of (parsed_date, original_date, position) tuples
        MAX_DISTANCE (int): Maximum distance between diagnosis and date to consider
        
    Returns:
        list: List of feature dictionaries for each diagnosis-date pair
    """
    # Import config here to avoid circular imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    
    # Check if debug mode is enabled
    debug_mode = getattr(config, 'MODEL_DEBUG_MODE', False)
    
    # Build features for each diagnosis-date pair
    features = []
    pairs_considered = 0
    pairs_within_distance = 0
    
    for entity_tuple in diagnoses:
        # Handle both 2-part (diagnosis, position) and 3-part (diagnosis, position, category) tuples
        if len(entity_tuple) == 3:
            diagnosis, diag_pos, entity_category = entity_tuple
        else:
            diagnosis, diag_pos = entity_tuple
            entity_category = 'diagnosis'  # Default category
        
        # Correctly unpack the 3-element tuple from the dates list
        for parsed_date, date_str, date_pos in dates:
            pairs_considered += 1
            distance = abs(diag_pos - date_pos)
            if distance > MAX_DISTANCE:
                continue
                
            pairs_within_distance += 1
            start_pos = max(0, min(diag_pos, date_pos) - 50)
            end_pos = min(len(note), max(diag_pos, date_pos) + 100)
            context = note[start_pos:end_pos]
            context = preprocess_text(context)
            
            # DIAGNOSTIC: Print detailed information about this candidate pair
            # Comment out the detailed candidate pair diagnostics to reduce verbosity
            """
            if debug_mode and pairs_within_distance <= 3:  # Only print first 3 for brevity
                print(f"\n--- Candidate Pair {pairs_within_distance} ---")
                print(f"Diagnosis: '{diagnosis}' at position {diag_pos}")
                print(f"Entity Category: '{entity_category}'")
                print(f"Date: Original='{date_str}', Parsed='{parsed_date}' at position {date_pos}")
                print(f"Distance: {distance} words")
                print(f"Diagnosis before date: {'Yes' if diag_pos < date_pos else 'No'}")
                print(f"Context window: [{start_pos}:{end_pos}] (length: {end_pos-start_pos})")
                print(f"Raw context: '{note[start_pos:end_pos][:100]}...'")
                print(f"Preprocessed context: '{context[:100]}...'")
            """
            
            feature = {
                'diagnosis': diagnosis,
                'date': parsed_date, # Use parsed date instead of raw date string
                'context': context,
                'distance': distance,
                'diag_pos_rel': diag_pos - start_pos,
                'date_pos_rel': date_pos - start_pos,
                'diag_before_date': 1 if diag_pos < date_pos else 0,
                'entity_category': entity_category  # Add entity category to features
            }
            features.append(feature)
    
    # DIAGNOSTIC: Print combined input and summary statistics
    #if debug_mode:
    #    print("\n===== DIAGNOSTIC: PREPROCESS_NOTE_FOR_PREDICTION =====")
    #    print(f"Number of diagnoses: {len(diagnoses)}")
    #    print(f"Number of dates: {len(dates)}")
    #    print(f"Max distance setting: {MAX_DISTANCE} words")
    #    print(f"Total diagnosis-date pairs considered: {pairs_considered}")
    #    print(f"Pairs within max distance ({MAX_DISTANCE} words): {pairs_within_distance}")
    #    print(f"Total features generated: {len(features)}")
    #    print("==========================================================\n")
    
    return features

def create_prediction_dataset(features, vocab, device, max_distance, max_context_len):
    """Convert preprocessed features into model-ready tensors
    
    Args:
        features (list): List of feature dictionaries
        vocab (Vocabulary): Vocabulary object with word2idx mapping
        device (torch.device): Device to place tensors on
        max_distance (int): Maximum distance value for normalization
        max_context_len (int): Maximum context length for padding/truncation
        
    Returns:
        list: List of tensor dictionaries ready for model input
    """
    # Import config here to avoid circular imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    
    # Check if debug mode is enabled
    debug_mode = getattr(config, 'MODEL_DEBUG_MODE', False)
    
    test_data = []
    unknown_words_count = 0
    total_words_count = 0
    
    # Process first 3 features in detail for diagnostics
    detailed_diagnostics_count = min(3, len(features))
    
    for i, feature in enumerate(features):
        # For the first few features, print detailed diagnostics
        show_details = debug_mode and i < detailed_diagnostics_count
        
        # Comment out the detailed feature vectorization diagnostics to reduce verbosity
        """
        if show_details:
            print(f"\n--- Feature {i+1} Vectorization ---")
            print(f"Diagnosis: '{feature['diagnosis']}'")
            print(f"Date: '{feature['date']}'")
            print(f"First 50 words of context: '{' '.join(feature['context'].split()[:50])}...'")
        """
        
        # Convert words to indices
        context_indices = []
        feature_unknown_words = 0
        feature_total_words = 0
        
        for word in feature['context'].split():
            feature_total_words += 1
            total_words_count += 1
            
            if word in vocab.word2idx:
                context_indices.append(vocab.word2idx[word])
            else:
                context_indices.append(vocab.word2idx['<unk>'])
                feature_unknown_words += 1
                unknown_words_count += 1
        
        """
        if show_details:
            print(f"Words in context: {feature_total_words}")
            print(f"Unknown words: {feature_unknown_words} ({feature_unknown_words/feature_total_words*100:.1f}%)")
            print(f"First 10 word indices: {context_indices[:10]}...")
        """
        
        # Pad or truncate using max_context_len
        original_length = len(context_indices)
        if len(context_indices) > max_context_len:  
            context_indices = context_indices[:max_context_len]
            """
            if show_details:
                print(f"Context truncated from {original_length} to {max_context_len} tokens")
            """
        else:
            padding = [0] * (max_context_len - len(context_indices))
            context_indices.extend(padding)
            """
            if show_details:
                print(f"Context padded from {original_length} to {max_context_len} tokens")
            """
        
        # Normalize distance using max_distance
        distance = min(feature['distance'] / max_distance, 1.0)
        
        """
        if show_details:
            print(f"Raw distance: {feature['distance']}, Normalized: {distance:.4f}")
            print(f"Diagnosis before date: {feature['diag_before_date']}")
        """
        
        # Create tensor dict
        tensor_dict = {
            'context': torch.tensor(context_indices, dtype=torch.long).unsqueeze(0).to(device),
            'distance': torch.tensor(distance, dtype=torch.float).unsqueeze(0).to(device),
            'diag_before': torch.tensor(feature['diag_before_date'], dtype=torch.float).unsqueeze(0).to(device),
            'feature': feature  # Keep original feature for reference
        }
        test_data.append(tensor_dict)
    
    # DIAGNOSTIC: Print combined vectorization parameters and summary statistics
    #if debug_mode:
        #print("\n===== DIAGNOSTIC: CREATE_PREDICTION_DATASET =====")
        #print(f"Vocabulary size: {vocab.n_words}")
        #print(f"Max context length: {max_context_len}")
        #print(f"Max distance for normalization: {max_distance}")
        #print(f"Device: {device}")
        #print(f"Total features processed: {len(features)}")
        #print(f"Total words processed: {total_words_count}")
        #print(f"Unknown words encountered: {unknown_words_count} ({unknown_words_count/total_words_count*100:.1f}% of total)")
        #print(f"Test data tensors created: {len(test_data)}")
        #print("==========================================================\n")
    
    return test_data

def predict_relationships(model, test_data):
    """Use the model to predict relationships between diagnoses and dates
    
    Args:
        model (DiagnosisDateRelationModel): Trained model
        test_data (list): List of tensor dictionaries
        
    Returns:
        list: List of relationship dictionaries with predictions
    """
    relationships = []
    model.eval()
    
    with torch.no_grad():
        for data in test_data:
            # Get prediction
            output = model(data['context'], data['distance'], data['diag_before'])
            prob = output.item()
            
            # Get original feature
            feature = data['feature']
            
            # Add to relationships if probability exceeds threshold
            if prob > 0.5:  # Can adjust threshold as needed
                relationships.append({
                    'diagnosis': feature['diagnosis'],
                    'date': feature['date'],
                    'confidence': prob
                })
    
    return relationships

# Helper function to run extraction process for a given extractor and data
def run_extraction(extractor, prepared_test_data, gold_standard=None, dataset_path=None):
    """
    Runs the extraction process for a given extractor on prepared data.

    Args:
        extractor: An initialized and loaded extractor object (subclass of BaseExtractor).
        prepared_test_data (list): List of dicts {'patient_id': ..., 'note_id': ..., 'note': ..., 'entities': ...}.
        gold_standard (list, optional): List of gold standard relationships for comparison.
        dataset_path (str, optional): Path to the dataset file to load note texts.

    Returns:
        list: List of predicted relationships [{'note_id': ..., 'patient_id': ..., 'entity_label': ..., 'entity_category': ..., 'date': ..., 'confidence': ...}]
             or [{'note_id': ..., 'patient_id': ..., 'diagnosis': ..., 'date': ..., 'confidence': ...}] in disorder_only mode.
    """
    print(f"Generating predictions using {extractor.name}...")
    all_predictions = []
    skipped_rels = 0
    
    # Determine if we're in disorder_only mode based on the entity_mode in config
    # rather than assuming based on extractor name
    if hasattr(extractor, 'config') and hasattr(extractor.config, 'ENTITY_MODE'):
        disorder_only_mode = extractor.config.ENTITY_MODE.lower() == 'diagnosis_only'
    else:
        # Fallback to the old heuristic for backward compatibility
        disorder_only_extractors = ['naive']  # Removed 'custom' and 'relcat' as they can handle multi-entity
        disorder_only_mode = any(name.lower() in extractor.name.lower() for name in disorder_only_extractors)
    
    # Load note texts if gold standard and dataset path are provided
    note_texts = {}
    gold_by_note_id = {}
    
    if gold_standard and dataset_path:
        # Group gold standard by note_id for easier lookup
        for g in gold_standard:
            note_id = g.get('note_id')
            if note_id not in gold_by_note_id:
                gold_by_note_id[note_id] = []
            gold_by_note_id[note_id].append(g)
        
        # Load note texts
        try:
            import pandas as pd
            df = pd.read_csv(dataset_path)
            if 'note' in df.columns:
                for i, row in df.iterrows():
                    note_id = row.get('note_id', i)  # Use note_id column if available, otherwise use index
                    note_texts[note_id] = row['note']
        except Exception as e:
            print(f"Warning: Could not load note texts from dataset: {e}")
    
    for note_entry in tqdm(prepared_test_data, desc=f"Processing with {extractor.name}", unit="note"):
        note_id = note_entry['note_id']
        patient_id = note_entry['patient_id']
        note_predictions = []
        
        try:
            # Extract relationships using the provided extractor
            relationships = extractor.extract(note_entry['note'], entities=note_entry['entities'], 
                                            note_id=note_id, patient_id=patient_id)
            
            for rel in relationships:
                # Check if we have a disorder_only mode extractor (using 'diagnosis')
                # or a multi_entity mode extractor (using 'entity_label')
                if 'diagnosis' in rel:
                    # disorder_only mode
                    entity_label = rel.get('diagnosis')
                    entity_category = 'disorder'  # Default for disorder_only mode
                elif 'entity_label' in rel:
                    # multi_entity mode
                    entity_label = rel.get('entity_label')
                    entity_category = rel.get('entity_category', 'unknown')
                else:
                    # Missing both required fields
                    print(f"Warning: Relationship missing both 'diagnosis' and 'entity_label' fields: {rel}")
                    skipped_rels += 1
                    continue
                
                # Get the date
                date_str = rel.get('date')
                
                if date_str is None or entity_label is None:
                    missing_fields = []
                    if date_str is None:
                        missing_fields.append("'date'")
                    if entity_label is None:
                        missing_fields.append("'diagnosis'/'entity_label'")
                    
                    print(f"Warning: Skipping relationship in note {note_id} due to missing {' and '.join(missing_fields)}. Rel: {rel}")
                    skipped_rels += 1
                    continue

                # Normalize entity label and date
                normalized_entity = str(entity_label).strip().lower() # Ensure string, strip, lower
                date_str = str(date_str).strip() # Ensure string and strip whitespace

                if date_str and normalized_entity:
                    # Apply category mapping if available
                    mapped_category = entity_category
                    if hasattr(extractor, 'category_mapping') and entity_category in extractor.category_mapping:
                        mapped_category = extractor.category_mapping[entity_category]
                        # Track the mapping for reporting
                        if hasattr(extractor, 'seen_raw_to_normalized'):
                            extractor.seen_raw_to_normalized[entity_category] = mapped_category
                    
                    # Create prediction in the appropriate format based on the mode
                    if disorder_only_mode:
                        prediction = {
                            'note_id': note_id,
                            'patient_id': patient_id,
                            'diagnosis': normalized_entity,
                            'date': date_str,
                            'confidence': rel.get('confidence', 1.0), # Default confidence to 1.0 if missing
                            'original_category': entity_category,
                            'mapped_category': mapped_category
                        }
                    else:
                        prediction = {
                            'note_id': note_id,
                            'patient_id': patient_id,
                            'entity_label': normalized_entity,
                            'entity_category': mapped_category,  # Use the mapped category
                            'original_category': entity_category,  # Store the original category for reference
                            'date': date_str,
                            'confidence': rel.get('confidence', 1.0) # Default confidence to 1.0 if missing
                        }
                    
                    all_predictions.append(prediction)
                    note_predictions.append(prediction)
                else:
                    # Log if validation failed
                    skipped_rels += 1
            
            # Print a unified output for this note if gold standard is available
            if gold_standard and note_id in gold_by_note_id and note_id in note_texts:
                note_text = note_texts[note_id]
                note_gold = gold_by_note_id[note_id]
                
                # Print a concise summary for this note
                print(f"\n[Note ID: {note_id}]")
                
                # Print note text preview
                note_preview = note_text[:50] + "..." if len(note_text) > 50 else note_text
                print(f"NOTE TEXT: {note_preview}")
                
                # Print gold standard
                print("GOLD STANDARD:")
                if note_gold:
                    for g in note_gold:
                        entity = g.get('entity_label', g.get('diagnosis', 'unknown'))
                        category = g.get('entity_category', 'unknown')
                        date = g.get('date', 'unknown')
                        print(f"  - {entity} ({category}) @ {date}")
                else:
                    print("  None")
                
                # Print predictions with confidence scores
                print("PREDICTIONS:")
                if note_predictions:
                    for p in note_predictions:
                        entity = p.get('entity_label', p.get('diagnosis', 'unknown'))
                        category = p.get('entity_category', 'unknown')
                        
                        # Apply category mapping for display
                        mapped_category = category
                        if hasattr(extractor, 'config') and hasattr(extractor.config, 'CATEGORY_MAPPINGS') and category in extractor.config.CATEGORY_MAPPINGS:
                            mapped_category = extractor.config.CATEGORY_MAPPINGS[category]
                        
                        date = p.get('date', 'unknown')
                        confidence = p.get('confidence', 0.0)
                        
                        # Truncate very long entity names to prevent line wrapping
                        max_entity_length = 40
                        display_entity = entity
                        if len(entity) > max_entity_length:
                            display_entity = entity[:max_entity_length] + "..."
                        
                        # Get the category - use entity_category which should already be mapped
                        # or fall back to the original_category if available
                        display_category = p.get('entity_category', p.get('original_category', category))
                        
                        # Use a more compact format to avoid line wrapping
                        print(f"  - {display_entity} ({display_category}) @ {date} [conf: {confidence:.4f}]")
                else:
                    print("  None")
                
                print("-" * 60)
                
        except Exception as e:
            # Log errors during extraction for a specific note
            print(f"Extraction error on note {note_id} for {extractor.name}: {e}")
            # Optionally, re-raise if you want errors to halt execution: raise e
            continue # Continue with the next note

    print(f"Generated {len(all_predictions)} predictions. Skipped {skipped_rels} potentially invalid relationships.")
    
    # Print summary of entity categories if available
    if hasattr(extractor, 'seen_categories') and extractor.seen_categories:
        print("\n===== ENTITY CATEGORY SUMMARY =====")
        print(f"Found {len(extractor.seen_categories)} unique entity categories:")
        for category in sorted(extractor.seen_categories):
            print(f"  - '{category}'")
        
        # Print the category mapping if available
        if hasattr(extractor, 'seen_raw_to_normalized') and extractor.seen_raw_to_normalized:
            print("\n===== CATEGORY MAPPING USED =====")
            print(f"Applied {len(extractor.seen_raw_to_normalized)} category mappings:")
            for raw_category, normalized in sorted(extractor.seen_raw_to_normalized.items()):
                print(f"  - '{raw_category}' → '{normalized}'")
        print("===================================\n")
    
    return all_predictions

def calculate_entity_metrics(prepared_test_data, entity_gold, output_dir, config=None):
    """
    Evaluates the entity extraction (NER) performance by comparing extracted entities with gold standard.
    
    Args:
        prepared_test_data (list): List of dicts containing extracted entities.
        entity_gold (list): List of gold standard entities.
        output_dir (str): Directory to save evaluation outputs.
        config (module, optional): Configuration module with CATEGORY_MAPPINGS.
        
    Returns:
        dict: A dictionary containing calculated metrics.
    """
    print("\n--- Evaluating Entity Extraction (NER) ---")
    if not entity_gold:
        print("No entity_gold data provided. Skipping NER evaluation.")
        return {'precision': 0, 'recall': 0, 'f1': 0}
    
    # Get category mappings if available
    category_mappings = {}
    if config and hasattr(config, 'CATEGORY_MAPPINGS'):
        category_mappings = config.CATEGORY_MAPPINGS
        print("Using category mappings for NER evaluation")
    
    # Create sets for comparison
    # Using (note_id, entity_label, normalized_category, start, end) as the key for matching
    gold_entities_set = set()
    for g in entity_gold:
        # Apply category mapping to gold standard if available
        entity_category = g['entity_category'].lower() if g['entity_category'] else 'unknown'
        if entity_category in category_mappings:
            normalized_category = category_mappings[entity_category]
        else:
            normalized_category = entity_category
            
        gold_entities_set.add(
            (g['note_id'], g['entity_label'], normalized_category, g['start'], g['end'])
        )

    # Extract predicted entities from prepared_test_data and apply mappings
    predicted_entities_set = set()
    for note_data in prepared_test_data:
        note_id = note_data['note_id']
        for entity in note_data['extracted_entities']:
            # Apply category mapping to predicted entities if available
            entity_category = entity['category'].lower() if entity['category'] else 'unknown'
            if entity_category in category_mappings:
                normalized_category = category_mappings[entity_category]
            else:
                normalized_category = entity_category
                
            predicted_entities_set.add(
                (note_id, entity['label'], normalized_category, entity['start'], entity['end'])
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
    
    # Skip detailed note-by-note comparison as we now show this during extraction
    # We'll just print a summary message
    print("\n--- Evaluation Summary ---")
    print(f"Evaluating {len(filtered_predictions)} predictions against {len(gold_standard)} gold standard relationships.")
    print(f"Notes with gold standard labels: {num_labeled_notes}")
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